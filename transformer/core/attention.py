"""
transformer/core/attention.py
================================
Enterprise multi-head scaled dot-product attention.

Features
--------
- MultiHeadAttention: full encoder-decoder and self-attention
- FeedForward: position-wise FFN with configurable activation
- SinusoidalPositionalEncoding: learned-or-fixed sinusoidal PE
- CausalMask: helper for autoregressive decoder
- Dropout + Layer Norm as sub-components
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class MultiHeadAttention(nn.Module):
    """
    Scaled dot-product multi-head attention.

    Supports:
    - Self-attention (src=tgt queries)
    - Cross-attention (decoder queries over encoder memory)
    - Key/value padding mask
    - Causal (look-ahead) mask for autoregressive decoding

    Parameters
    ----------
    d_model:    Model dimensionality.
    n_heads:    Number of attention heads.
    dropout:    Attention weight dropout probability.
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.scale    = math.sqrt(self.d_head)

        self.W_q  = nn.Linear(d_model, d_model, bias=False)
        self.W_k  = nn.Linear(d_model, d_model, bias=False)
        self.W_v  = nn.Linear(d_model, d_model, bias=False)
        self.W_o  = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query:    "torch.Tensor",           # (B, Tq, D)
        key:      "torch.Tensor",           # (B, Tk, D)
        value:    "torch.Tensor",           # (B, Tv, D)
        key_mask: Optional["torch.Tensor"] = None,   # (B, Tk) bool — True=ignore
        attn_mask:Optional["torch.Tensor"] = None,   # (Tq, Tk) additive float mask
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Returns
        -------
        output:  (B, Tq, D)
        weights: (B, H, Tq, Tk) — averaged attention weights for visualisation
        """
        B, Tq, _  = query.shape
        _, Tk, _  = key.shape

        Q = self.W_q(query).view(B, Tq, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(key  ).view(B, Tk, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(value).view(B, Tk, self.n_heads, self.d_head).transpose(1, 2)
        # Q/K/V: (B, H, T, d_head)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, Tq, Tk)

        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)

        if key_mask is not None:
            # key_mask: (B, Tk) → (B, 1, 1, Tk)
            scores = scores.masked_fill(
                key_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        weights = F.softmax(scores, dim=-1)
        weights = self.drop(weights)

        # Replace NaN weights from all-masked rows with 0
        weights = weights.nan_to_num(nan=0.0)

        out = torch.matmul(weights, V)                        # (B, H, Tq, d_head)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        out = self.W_o(out)
        return out, weights


class FeedForward(nn.Module):
    """
    Position-wise two-layer FFN: Linear → GELU → Dropout → Linear → Dropout.

    Parameters
    ----------
    d_model:    Input/output dimensionality.
    d_ff:       Inner dimensionality (typically 4 × d_model).
    dropout:    Dropout probability applied after each activation.
    activation: "gelu" (default) or "relu".
    """

    def __init__(
        self,
        d_model:    int,
        d_ff:       int,
        dropout:    float = 0.1,
        activation: str   = "gelu",
    ) -> None:
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_ff)
        self.fc2  = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding as described in 'Attention Is All You Need'.

    Computes PE lazily on first forward call and caches for subsequent uses.

    Parameters
    ----------
    d_model:   Embedding dimensionality.
    max_len:   Maximum sequence length to pre-compute.
    dropout:   Dropout applied to PE-augmented embeddings.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """x: (B, T, D) → (B, T, D) with PE added."""
        return self.drop(x + self.pe[:, : x.size(1)])


class EncoderLayer(nn.Module):
    """
    One Transformer encoder layer: Pre-LN self-attention + Pre-LN FFN.

    Pre-LayerNorm (before sublayer, not after) stabilises training at
    depth and reduces need for aggressive warmup.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff:    int,
        dropout: float = 0.1,
        eps:     float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn     = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff       = FeedForward(d_model, d_ff, dropout)
        self.norm1    = nn.LayerNorm(d_model, eps=eps)
        self.norm2    = nn.LayerNorm(d_model, eps=eps)
        self.drop     = nn.Dropout(dropout)

    def forward(
        self,
        x:        "torch.Tensor",
        src_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        # Pre-LN self-attention
        residual = x
        x_norm   = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_mask=src_mask)
        x = residual + self.drop(attn_out)
        # Pre-LN FFN
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    """
    One Transformer decoder layer:
    Pre-LN masked self-attention → Pre-LN cross-attention → Pre-LN FFN.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff:    int,
        dropout: float = 0.1,
        eps:     float = 1e-6,
    ) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff         = FeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model, eps=eps)
        self.norm2      = nn.LayerNorm(d_model, eps=eps)
        self.norm3      = nn.LayerNorm(d_model, eps=eps)
        self.drop       = nn.Dropout(dropout)

    def forward(
        self,
        x:        "torch.Tensor",
        memory:   "torch.Tensor",
        tgt_mask: Optional["torch.Tensor"] = None,
        mem_mask: Optional["torch.Tensor"] = None,
        causal_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        B, T, _ = x.shape
        # Pre-LN masked self-attention (causal)
        residual = x
        x_norm   = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            key_mask=tgt_mask,
            attn_mask=causal_mask,
        )
        x = residual + self.drop(attn_out)
        # Pre-LN cross-attention over encoder memory
        residual = x
        x_norm   = self.norm2(x)
        cross_out, _ = self.cross_attn(x_norm, memory, memory, key_mask=mem_mask)
        x = residual + self.drop(cross_out)
        # Pre-LN FFN
        x = x + self.drop(self.ff(self.norm3(x)))
        return x


def make_causal_mask(seq_len: int, device: "torch.device") -> "torch.Tensor":
    """
    Build an additive causal (look-ahead) mask of shape (T, T).
    Positions that should be masked contain -inf; others contain 0.
    """
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    )
    return mask   # (T, T)
