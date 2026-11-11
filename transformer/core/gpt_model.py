"""
transformer/core/gpt_model.py
================================
DGBGPT - decoder-only causal language model (roadmap task T0.2).

Why this exists
---------------
The legacy encoder-decoder DGBTransformer was trained with the answer in its input
(label leakage, see tests/unit/test_training_objective.py), so it never learned language.
A decoder-only model has ONE input stream and a causal mask, so the objective is the
standard next-token prediction:

    logits = model(ids[:, :-1])            # src from the dataset loader
    loss   = cross_entropy(logits, ids[:, 1:])   # tgt from the dataset loader

Architecture (all in plain PyTorch)
-----------------------------------
- token embedding, tied with the output projection (optional)
- pre-norm blocks with RMSNorm
- rotary position embeddings (RoPE) on queries and keys
- causal self-attention via torch.nn.functional.scaled_dot_product_attention
- SwiGLU feed-forward
- attention projections are named W_q / W_k / W_v / W_o so the existing LoRA code
  (finetune/core/lora_adapter.py) finds them unchanged

Right-padded batches need no padding mask: causal attention means a real token never
looks at the padding that follows it, and the loss ignores pad targets.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.to(x.dtype) * self.weight


def build_rope_cache(head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin), each of shape (max_seq_len, head_dim // 2)."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of channels by position. x: (B, H, T, head_dim)."""
    t = x.size(-2)
    half = x.size(-1) // 2
    cos = cos[:t].to(x.dtype)[None, None]
    sin = sin[:t].to(x.dtype)[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        self.dropout = dropout
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        shape = (b, t, self.n_heads, self.head_dim)
        q = self.W_q(x).view(shape).transpose(1, 2)
        k = self.W_k(x).view(shape).transpose(1, 2)
        v = self.W_v(x).view(shape).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.W_o(out.transpose(1, 2).contiguous().view(b, t, d))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, eps: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model, eps)
        self.mlp = SwiGLU(d_model, d_ff)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), cos, sin))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class DGBGPT(nn.Module):
    """Decoder-only language model. forward(ids) -> logits of shape (B, T, vocab)."""

    architecture = "decoder_only"

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_idx: int = 0,
        tie_embeddings: bool = True,
        layer_norm_eps: float = 1e-6,
        rope_theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.config = dict(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, dropout=dropout, max_seq_len=max_seq_len, pad_idx=pad_idx,
            tie_embeddings=tie_embeddings, layer_norm_eps=layer_norm_eps, rope_theta=rope_theta,
        )

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, n_heads, d_ff, dropout, layer_norm_eps) for _ in range(n_layers)]
        )
        self.final_norm = RMSNorm(d_model, layer_norm_eps)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.output_proj.weight = self.embed.weight

        cos, sin = build_rope_cache(d_model // n_heads, max_seq_len, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style: shrink the residual-branch output projections with depth.
        res_std = 0.02 / math.sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=res_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ API
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.size(1) > self.max_seq_len:
            raise ValueError(f"sequence length {ids.size(1)} exceeds max_seq_len {self.max_seq_len}")
        x = self.drop(self.embed(ids))
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        return self.output_proj(self.final_norm(x))

    def make_padding_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """True where the token is padding (kept for compatibility with the inference engine)."""
        return ids == self.pad_idx

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ----------------------------------------------------------- generation
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        eos_id: Optional[int] = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation. prompt_ids: (B, T). Returns only the NEW tokens, shape
        (B, n_new). Rows that hit eos_id are padded with pad_idx afterwards.
        temperature == 0 means greedy. No KV cache yet (roadmap T2.3): each step re-runs
        the (cropped) context, which is fine for tests and small models.
        """
        was_training = self.training
        self.eval()
        try:
            ids = prompt_ids
            b = ids.size(0)
            done = torch.zeros(b, dtype=torch.bool, device=ids.device)
            new_tokens = []
            for _ in range(max_new_tokens):
                logits = self(ids[:, -self.max_seq_len:])[:, -1, :].float()
                nxt = self._pick(logits, temperature, top_k, top_p, generator)
                nxt = torch.where(done, torch.full_like(nxt, self.pad_idx), nxt)
                new_tokens.append(nxt)
                ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
                if eos_id is not None:
                    done = done | (nxt == eos_id)
                    if bool(done.all()):
                        break
            if not new_tokens:
                return ids[:, :0]
            return torch.stack(new_tokens, dim=1)
        finally:
            self.train(was_training)

    @staticmethod
    def _pick(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
              generator: Optional[torch.Generator]) -> torch.Tensor:
        if temperature <= 0.0:
            return logits.argmax(dim=-1)
        logits = logits / temperature
        if top_k and top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            remove = cum - probs > top_p          # keep the token that crosses the threshold
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 200,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Same call signature as the legacy model so the inference engine works unchanged.
        `src` is the encoded prompt; a trailing EOS (added by the tokenizer) is dropped so
        the model continues the text. Returns only the newly generated ids.
        """
        ids = src
        if ids.size(1) == 0:
            ids = torch.full((ids.size(0), 1), bos_id, dtype=torch.long, device=ids.device)
        elif ids.size(1) > 1 and bool((ids[:, -1] == eos_id).all()):
            ids = ids[:, :-1]
        return self.generate(ids, max_new_tokens=max_len, eos_id=eos_id, temperature=0.0)
