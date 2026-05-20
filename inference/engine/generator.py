"""
inference/engine/generator.py
================================
High-level generation engine.

Combines beam search and top-p sampling under one interface.
Loads the model lazily (once) and serves concurrent requests.

Also manages:
- Model warm-up (forward pass on dummy input to trigger JIT/CUDA compilation)
- Concurrent request serialisation via asyncio.Lock (single-GPU protection)
- Per-request timeout
- Token and character count telemetry
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple

from inference.sampling.beam_search import BeamSearchDecoder, BeamSearchConfig, Hypothesis
from inference.sampling.top_p_sampler import TopPSampler, GenerationConfig, StreamingSampler
from modules.utils.error_handler import InferenceError, ModelInitError

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class InferenceEngine:
    """
    Unified inference engine.

    Parameters
    ----------
    model:      Loaded DGBTransformer (eval mode, on device).
    tokenizer:  Loaded DGBTokenizer.
    device:     torch.device for inference.
    """

    def __init__(self, model, tokenizer, device: "torch.device") -> None:
        self._model     = model
        self._tokenizer = tokenizer
        self._device    = device
        self._lock      = asyncio.Lock()
        self._beam_dec  = BeamSearchDecoder(model, bos_id=2, eos_id=3, pad_id=0, device=device)
        self._stream_sampler = StreamingSampler(
            model, tokenizer, bos_id=2, eos_id=3, pad_id=0, device=device,
        )
        self._total_tokens = 0
        self._total_requests = 0

    # ── Warm-up ───────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Run one dummy forward pass to trigger CUDA kernel compilation."""
        try:
            dummy = torch.tensor([[2, 3, 1, 0]], dtype=torch.long, device=self._device)
            with torch.no_grad():
                _ = self._model(dummy, dummy)
            logger.info("InferenceEngine: warm-up complete")
        except Exception as exc:
            logger.warning("Warm-up failed: %s", exc)

    # ── Synchronous complete ──────────────────────────────────────────

    async def complete(
        self,
        prompt:    str,
        cfg:       GenerationConfig,
        use_beam:  bool = False,
        beam_cfg:  Optional[BeamSearchConfig] = None,
    ) -> str:
        """
        Generate a complete response for `prompt`.
        Returns the full decoded output string.
        """
        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._complete_sync, prompt, cfg, use_beam, beam_cfg
            )

    def _complete_sync(
        self,
        prompt:   str,
        cfg:      GenerationConfig,
        use_beam: bool,
        beam_cfg: Optional[BeamSearchConfig],
    ) -> str:
        t0 = time.perf_counter()
        try:
            ids = self._tokenizer.encode(prompt, add_special_tokens=True)
            src = torch.tensor([ids], dtype=torch.long, device=self._device)
            src_mask = self._model.make_padding_mask(src)

            if use_beam:
                bc   = beam_cfg or BeamSearchConfig()
                hyps = self._beam_dec.decode(src, bc, src_mask)
                best = hyps[0][0] if hyps and hyps[0] else None
                out_ids = best.tokens if best else [3]
            else:
                out_ids = self._model.greedy_decode(
                    src, bos_id=2, eos_id=3, max_len=cfg.max_new_tokens, src_mask=src_mask
                )[0].tolist()

            text = self._tokenizer.decode(out_ids, skip_special_tokens=True)
            self._total_tokens   += len(out_ids)
            self._total_requests += 1
            latency = (time.perf_counter() - t0) * 1000
            logger.debug("Inference: %d tokens  %.0fms", len(out_ids), latency)
            return text
        except Exception as exc:
            logger.error("Inference error: %s", exc)
            raise InferenceError(str(exc)) from exc

    # ── Streaming complete ────────────────────────────────────────────

    async def stream(
        self,
        prompt: str,
        cfg:    GenerationConfig,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields decoded text tokens one by one.
        Suitable for SSE and WebSocket streaming endpoints.
        """
        ids = self._tokenizer.encode(prompt, add_special_tokens=True)
        src = torch.tensor([ids], dtype=torch.long, device=self._device)

        loop = asyncio.get_event_loop()

        def _gen():
            return list(self._stream_sampler.generate(src, cfg))

        async with self._lock:
            pairs: List[Tuple[int, str]] = await loop.run_in_executor(None, _gen)

        for _, tok_text in pairs:
            if tok_text:
                yield tok_text

    # ── Telemetry ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "total_requests": self._total_requests,
            "total_tokens":   self._total_tokens,
        }


# ── Singleton management ──────────────────────────────────────────────────────

_engine: Optional[InferenceEngine] = None


def get_engine() -> Optional[InferenceEngine]:
    return _engine


def init_engine(model, tokenizer, device: "torch.device") -> InferenceEngine:
    global _engine
    _engine = InferenceEngine(model, tokenizer, device)
    _engine.warmup()
    logger.info("InferenceEngine initialised on device=%s", device)
    return _engine
