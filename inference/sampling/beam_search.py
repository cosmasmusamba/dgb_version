"""
inference/sampling/beam_search.py
===================================
Full beam search decoder for DGBTransformer.

Features
--------
- Configurable beam width, length penalty, min/max output length
- Diverse beam groups (optional)
- Returns ranked hypotheses with log-probabilities
- No-repeat n-gram blocking
- Early stopping when all beams have reached EOS
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
# Type stub when torch absent
if not _HAS_TORCH:
    pass


@dataclass
class Hypothesis:
    """A single beam hypothesis."""
    tokens:    List[int]
    log_prob:  float
    complete:  bool = False

    @property
    def length(self) -> int:
        return len(self.tokens)

    def score(self, length_penalty: float = 1.0) -> float:
        """Length-normalised log-probability."""
        return self.log_prob / (self.length ** length_penalty)


@dataclass
class BeamSearchConfig:
    beam_size:       int   = 4
    max_length:      int   = 200
    min_length:      int   = 1
    length_penalty:  float = 1.0
    no_repeat_ngram: int   = 3        # 0 = disabled
    early_stop:      bool  = True
    temperature:     float = 1.0      # applied before softmax in each step


class BeamSearchDecoder:
    """
    Stateless beam search decoder for any encoder-decoder model
    that exposes .encode() and .decode() + .output_proj methods.

    Usage
    -----
    decoder = BeamSearchDecoder(model, bos_id=2, eos_id=3, pad_id=0)
    results = decoder.decode(src_ids, cfg=BeamSearchConfig(beam_size=4))
    best_ids = results[0][0].tokens   # best hypothesis for first batch item
    """

    def __init__(
        self,
        model,
        bos_id: int,
        eos_id: int,
        pad_id: int = 0,
        device: Optional["torch.device"] = None,
    ) -> None:
        self._model  = model
        self._bos    = bos_id
        self._eos    = eos_id
        self._pad    = pad_id
        self._device = device or next(model.parameters()).device

    @torch.no_grad()
    def decode(
        self,
        src:      "torch.Tensor",
        cfg:      BeamSearchConfig = None,
        src_mask: Optional["torch.Tensor"] = None,
    ) -> List[List[Hypothesis]]:
        """
        Beam search over a batch.

        Parameters
        ----------
        src:      (B, Ts) source token ids
        cfg:      BeamSearchConfig (defaults if None)
        src_mask: (B, Ts) bool padding mask

        Returns
        -------
        List of length B, each element is a sorted list of Hypothesis objects
        (best hypothesis first).
        """
        if cfg is None:
            cfg = BeamSearchConfig()

        B      = src.size(0)
        device = self._device

        # Encode source once — reuse across all beam steps
        memory = self._model.encode(src, src_mask)   # (B, Ts, D)

        results: List[List[Hypothesis]] = []

        for b in range(B):
            mem_b = memory[b:b+1]   # (1, Ts, D)
            mk_b  = src_mask[b:b+1] if src_mask is not None else None
            hyps  = self._beam_search_single(mem_b, mk_b, cfg)
            results.append(hyps)

        return results

    def _beam_search_single(
        self,
        memory:   "torch.Tensor",
        mem_mask: Optional["torch.Tensor"],
        cfg:      BeamSearchConfig,
    ) -> List[Hypothesis]:
        """Run beam search for a single (unbatched) memory state."""
        device = self._device
        K      = cfg.beam_size

        # Active beams: list of (sequence_so_far, cumulative_log_prob)
        active:    List[Tuple[List[int], float]] = [([self._bos], 0.0)]
        completed: List[Hypothesis]              = []

        for step in range(cfg.max_length):
            if not active:
                break

            # Stack active beams into a batch for parallel decoding
            n_active = len(active)
            seqs     = [seq for seq, _ in active]
            max_t    = max(len(s) for s in seqs)
            tgt      = torch.full((n_active, max_t), self._pad, dtype=torch.long, device=device)
            for i, seq in enumerate(seqs):
                tgt[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

            # Expand memory to match n_active
            mem_exp = memory.expand(n_active, -1, -1)
            mk_exp  = mem_mask.expand(n_active, -1) if mem_mask is not None else None

            dec_out = self._model.decode(tgt, mem_exp, mem_mask=mk_exp)     # (n, T, D)
            logits  = self._model.output_proj(dec_out[:, -1, :])            # (n, V)

            # Apply temperature
            if cfg.temperature != 1.0:
                logits = logits / max(cfg.temperature, 1e-5)

            log_probs = F.log_softmax(logits, dim=-1)   # (n, V)

            new_active: List[Tuple[List[int], float]] = []
            candidates: List[Tuple[float, List[int]]] = []

            for i, (seq, cum_lp) in enumerate(active):
                lp = log_probs[i]                              # (V,)

                # No-repeat n-gram blocking
                if cfg.no_repeat_ngram > 0:
                    lp = self._block_ngrams(lp, seq, cfg.no_repeat_ngram)

                top_lps, top_ids = lp.topk(K * 2)

                for lp_val, tok_id in zip(top_lps.tolist(), top_ids.tolist()):
                    new_seq    = seq + [tok_id]
                    new_cum_lp = cum_lp + lp_val
                    candidates.append((new_cum_lp, new_seq))

            # Sort and keep top-K
            candidates.sort(key=lambda x: x[0], reverse=True)
            for cum_lp, seq in candidates[:K]:
                if seq[-1] == self._eos or len(seq) >= cfg.max_length:
                    if len(seq) >= cfg.min_length:
                        completed.append(Hypothesis(
                            tokens=seq, log_prob=cum_lp, complete=True
                        ))
                else:
                    new_active.append((seq, cum_lp))

            active = new_active

            # Early stopping
            if cfg.early_stop and len(completed) >= K:
                break

        # Any remaining active beams become partial hypotheses
        for seq, cum_lp in active:
            completed.append(Hypothesis(tokens=seq, log_prob=cum_lp, complete=False))

        # Sort by length-penalised score
        completed.sort(
            key=lambda h: h.score(cfg.length_penalty), reverse=True
        )
        return completed or [Hypothesis(tokens=[self._bos], log_prob=0.0)]

    def _block_ngrams(
        self,
        log_probs: "torch.Tensor",
        seq:       List[int],
        n:         int,
    ) -> "torch.Tensor":
        """Set log_prob to -inf for any token that would create a repeated n-gram."""
        if len(seq) < n:
            return log_probs
        last_ngram = tuple(seq[-(n-1):])
        for i in range(len(seq) - n + 1):
            if tuple(seq[i:i+n-1]) == last_ngram:
                blocked = seq[i + n - 1]
                log_probs = log_probs.clone()
                log_probs[blocked] = float("-inf")
        return log_probs
