"""
inference/sampling/top_p_sampler.py
=====================================
Nucleus (top-p) sampling, top-k sampling, and temperature scaling.

All three are combined into a single TopPSampler that can be configured
per-request via GenerationConfig.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@dataclass
class GenerationConfig:
    """Full generation configuration — passed to all samplers."""
    max_new_tokens:   int   = 256
    min_new_tokens:   int   = 1
    temperature:      float = 1.0
    top_k:            int   = 50
    top_p:            float = 0.9
    repetition_penalty: float = 1.1   # > 1 discourages repetition
    do_sample:        bool  = True    # False = greedy
    no_repeat_ngram:  int   = 3
    num_return:       int   = 1       # sequences to return per prompt


class TopPSampler:
    """
    Stateless sampler that applies temperature → top-k → top-p → multinomial sampling.

    Usage
    -----
    sampler = TopPSampler()
    next_tok = sampler.sample(logits, cfg)   # returns (B,) tensor of next token ids
    """

    def sample(
        self,
        logits:   "torch.Tensor",  # (B, V)
        cfg:      "GenerationConfig",
        prev_ids: Optional["torch.Tensor"] = None,  # (B, T) for repetition penalty
    ) -> "torch.Tensor":
        """Sample next token ids. Returns (B,) int64 tensor."""
        # Repetition penalty
        if cfg.repetition_penalty != 1.0 and prev_ids is not None:
            logits = self._apply_rep_penalty(logits, prev_ids, cfg.repetition_penalty)

        # Temperature scaling
        if cfg.temperature != 1.0:
            logits = logits / max(cfg.temperature, 1e-5)

        # Greedy
        if not cfg.do_sample:
            return logits.argmax(dim=-1)

        # Top-k filtering
        if cfg.top_k > 0:
            logits = self._top_k_filter(logits, cfg.top_k)

        # Top-p filtering
        if cfg.top_p < 1.0:
            logits = self._top_p_filter(logits, cfg.top_p)

        probs     = F.softmax(logits, dim=-1)
        next_toks = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_toks

    def _top_k_filter(self, logits: "torch.Tensor", k: int) -> "torch.Tensor":
        """Zero out all logits except the top-k."""
        values, _ = torch.topk(logits, min(k, logits.size(-1)))
        threshold  = values[:, -1].unsqueeze(-1)
        return logits.masked_fill(logits < threshold, float("-inf"))

    def _top_p_filter(self, logits: "torch.Tensor", p: float) -> "torch.Tensor":
        """Zero out logits outside the nucleus (smallest set with cumprob >= p)."""
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Shift right so the first token above threshold is kept
        remove = (cum_probs - F.softmax(sorted_logits, dim=-1)) > p
        sorted_logits[remove] = float("-inf")
        # Restore original order
        return sorted_logits.scatter(1, sorted_idx, sorted_logits)

    def _apply_rep_penalty(
        self,
        logits:  "torch.Tensor",
        prev_ids: "torch.Tensor",
        penalty:  float,
    ) -> "torch.Tensor":
        """Penalise tokens that have already appeared in prev_ids."""
        logits = logits.clone()
        for b in range(logits.size(0)):
            unique = prev_ids[b].unique()
            score  = logits[b, unique]
            # Divide positive scores, multiply negative scores
            logits[b, unique] = torch.where(score < 0, score * penalty, score / penalty)
        return logits


class StreamingSampler:
    """
    Stateful autoregressive token generator using TopPSampler.

    Maintains a running token buffer and yields one token at a time
    for token-streaming SSE endpoints.
    """

    def __init__(
        self,
        model,
        tokenizer,
        bos_id:  int,
        eos_id:  int,
        pad_id:  int,
        device:  "torch.device",
    ) -> None:
        self._model     = model
        self._tokenizer = tokenizer
        self._bos       = bos_id
        self._eos       = eos_id
        self._pad       = pad_id
        self._device    = device
        self._sampler   = TopPSampler()

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: "torch.Tensor",   # (1, T)
        cfg:        "GenerationConfig",
    ):
        """
        Async generator that yields (token_id, token_text) tuples.
        Suitable for SSE streaming from the API.
        """
        src      = prompt_ids.to(self._device)
        src_mask = self._model.make_padding_mask(src)
        memory   = self._model.encode(src, src_mask)

        generated = torch.full((1, 1), self._bos, dtype=torch.long, device=self._device)

        for step in range(cfg.max_new_tokens):
            dec_out = self._model.decode(generated, memory, mem_mask=src_mask)
            logits  = self._model.output_proj(dec_out[:, -1, :])   # (1, V)

            next_id = self._sampler.sample(logits, cfg, prev_ids=generated)
            generated = torch.cat([generated, next_id.unsqueeze(-1)], dim=1)
            tok_text  = self._tokenizer.decode([next_id.item()])

            yield next_id.item(), tok_text

            if next_id.item() == self._eos and step >= cfg.min_new_tokens:
                break
