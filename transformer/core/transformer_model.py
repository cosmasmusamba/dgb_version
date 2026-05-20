"""
transformer/core/transformer_model.py
========================================
DGBTransformer — full encoder-decoder architecture.

Built entirely from custom components in transformer/core/attention.py.
No HuggingFace, no fairseq, no third-party model libraries.

Architecture
------------
  Embedding → SinusoidalPE → N × EncoderLayer  (encoder)
  Embedding → SinusoidalPE → N × DecoderLayer  (decoder)
  → Linear head → logits over vocab
  Optional tied embeddings (encoder emb = decoder emb = output projection)
"""
from __future__ import annotations

import logging
from typing import Optional

from transformer.core.attention import (
    EncoderLayer, DecoderLayer, SinusoidalPositionalEncoding, make_causal_mask,
)

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class DGBTransformer(nn.Module):
    """
    Encoder-decoder Transformer for sequence generation.

    Parameters
    ----------
    vocab_size:       Vocabulary size.
    d_model:          Hidden dimensionality.
    n_heads:          Attention heads per layer.
    n_encoder_layers: Number of encoder stacks.
    n_decoder_layers: Number of decoder stacks.
    d_ff:             FFN inner dimensionality.
    dropout:          Dropout probability.
    max_seq_len:      Maximum sequence length.
    pad_idx:          Padding token index (ignored in loss and attention).
    tie_embeddings:   Share source/target embedding weights and output projection.
    layer_norm_eps:   Epsilon for LayerNorm stability.
    """

    def __init__(
        self,
        vocab_size:       int,
        d_model:          int   = 256,
        n_heads:          int   = 8,
        n_encoder_layers: int   = 4,
        n_decoder_layers: int   = 4,
        d_ff:             int   = 1_024,
        dropout:          float = 0.1,
        max_seq_len:      int   = 512,
        pad_idx:          int   = 0,
        tie_embeddings:   bool  = True,
        layer_norm_eps:   float = 1e-6,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.pad_idx    = pad_idx

        # ── Embeddings ────────────────────────────────────────────────
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        if tie_embeddings:
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        self.src_pe = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        self.tgt_pe = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        # ── Encoder ───────────────────────────────────────────────────
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, layer_norm_eps)
            for _ in range(n_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # ── Decoder ───────────────────────────────────────────────────
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, layer_norm_eps)
            for _ in range(n_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # ── Output projection ─────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.output_proj.weight = self.src_embed.weight

        self._init_weights()
        logger.info(
            "DGBTransformer: vocab=%d  d=%d  heads=%d  enc=%d  dec=%d  ff=%d  params=%.1fM",
            vocab_size, d_model, n_heads, n_encoder_layers, n_decoder_layers, d_ff,
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    def _init_weights(self) -> None:
        from transformer.utils.model_helpers import init_weights
        self.apply(init_weights)

    # ── Mask helpers ──────────────────────────────────────────────────

    def make_padding_mask(self, ids: "torch.Tensor") -> "torch.Tensor":
        """Return bool mask (B, T) — True where ids == pad_idx."""
        return ids == self.pad_idx

    # ── Forward ───────────────────────────────────────────────────────

    def encode(
        self,
        src:      "torch.Tensor",
        src_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Encode source sequence. Returns memory (B, Ts, D)."""
        x = self.src_pe(self.src_embed(src) * (self.d_model ** 0.5))
        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask)
        return self.encoder_norm(x)

    def decode(
        self,
        tgt:      "torch.Tensor",
        memory:   "torch.Tensor",
        tgt_mask: Optional["torch.Tensor"] = None,
        mem_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Decode target sequence attending over memory. Returns (B, Tt, D)."""
        T = tgt.size(1)
        causal = make_causal_mask(T, tgt.device)
        x = self.tgt_pe(self.tgt_embed(tgt) * (self.d_model ** 0.5))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, mem_mask=mem_mask, causal_mask=causal)
        return self.decoder_norm(x)

    def forward(
        self,
        src:      "torch.Tensor",
        tgt:      "torch.Tensor",
        src_mask: Optional["torch.Tensor"] = None,
        tgt_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Full encoder-decoder forward pass.

        Parameters
        ----------
        src:      (B, Ts) source token ids
        tgt:      (B, Tt) target token ids (shifted right — teacher forcing)
        src_mask: (B, Ts) bool padding mask
        tgt_mask: (B, Tt) bool padding mask

        Returns
        -------
        logits:   (B, Tt, vocab_size)
        """
        memory = self.encode(src, src_mask)
        dec    = self.decode(tgt, memory, tgt_mask=tgt_mask, mem_mask=src_mask)
        return self.output_proj(dec)

    # ── Generation helpers ────────────────────────────────────────────

    @torch.no_grad()
    def greedy_decode(
        self,
        src:      "torch.Tensor",
        bos_id:   int,
        eos_id:   int,
        max_len:  int = 200,
        src_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Greedy autoregressive decoding. Returns (B, T) output ids.
        Used as the fast fallback in the inference route.
        """
        B      = src.size(0)
        device = src.device
        memory = self.encode(src, src_mask)
        ys     = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            logits = self.decode(ys, memory, mem_mask=src_mask)  # (B, T, D)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == eos_id).all():
                break
        return ys
