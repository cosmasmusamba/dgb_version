"""
inference/engine/generator.py
================================
High-level generation engine.

Combines beam search and top-p sampling under one interface.
Loads the model lazily (once) and serves concurrent requests.

Standard DGB utilities used
----------------------------
- configure_logging / set_log_stage   (consistent logging format)
- get_config / get_path_resolver      (centralised config & path resolution)
- init_path_resolver                  (dynamic path discovery)
- UnifiedLogWriter / get_unified_log  (structured inference records)
- DeviceMonitor                       (hardware telemetry during warmup)
- StreamEvent / get_training_hub      (SSE fan-out for the API layer)
- error_handler.InferenceError        (typed exception hierarchy)
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
from modules.utils.unified_log import get_unified_log

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
    run_id:     RunContext run_id for unified log attribution.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: "torch.device",
        run_id: str = "",
    ) -> None:
        self._model     = model
        self._tokenizer = tokenizer
        self._device    = device
        self._run_id    = run_id
        self._lock      = asyncio.Lock()
        self._beam_dec  = BeamSearchDecoder(model, bos_id=2, eos_id=3, pad_id=0, device=device)
        self._stream_sampler = StreamingSampler(
            model, tokenizer, bos_id=2, eos_id=3, pad_id=0, device=device,
        )
        self._total_tokens   = 0
        self._total_requests = 0

    # ── Warm-up ───────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Run one dummy forward pass to trigger CUDA kernel compilation."""
        if not _HAS_TORCH:
            return
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
        """Generate a complete response for `prompt`."""
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
        method = "beam" if use_beam else "greedy"
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
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.debug("Inference: %d tokens  %.0fms  %s", len(out_ids), latency_ms, method)

            # Emit structured inference record to unified log
            ulog = get_unified_log()
            if ulog is not None:
                ulog.inference(
                    prompt_tokens=len(ids),
                    output_tokens=len(out_ids),
                    latency_ms=latency_ms,
                    method=method,
                )

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
        """Async generator that yields decoded text tokens one by one."""
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


def init_engine(
    model,
    tokenizer,
    device: "torch.device",
    run_id: str = "",
) -> InferenceEngine:
    global _engine
    _engine = InferenceEngine(model, tokenizer, device, run_id=run_id)
    _engine.warmup()
    logger.info("InferenceEngine initialised on device=%s", device)
    return _engine


# ── Entry point ───────────────────────────────────────────────────────────────

def run_inference(prompt: Optional[str] = None):
    """
    Entry point for quick inference testing.
    Dynamically loads latest checkpoint and tokenizer using PathResolver.
    """
    import torch
    from tokenizer.dgb_tokenizer import DGBTokenizer
    from transformer.utils.model_helpers import load_model
    from inference.sampling.top_p_sampler import GenerationConfig
    from configs.loader import get_config
    from modules.logging_config import configure_logging, set_log_stage
    from modules.utils.run_context import get_run_context, create_run_context
    from modules.utils.path_resolver import init_path_resolver
    from modules.utils.unified_log import init_unified_log

    # ── Logging & run context ─────────────────────────────────────────
    cfg = get_config()
    configure_logging(level=cfg.logging.level)
    set_log_stage("inference")

    ctx = get_run_context() if True else create_run_context(cfg.project.model_id)

    # ── Path resolution ───────────────────────────────────────────────
    model_id    = cfg.project.model_id
    res         = init_path_resolver(model_id=model_id, cfg=cfg)
    models_dir  = res.models_dir(create=False)
    tok_dir     = res.tokenizer_dir(create=False)
    log_dir     = res.logs_dir()

    # ── Unified log ───────────────────────────────────────────────────
    ulog = init_unified_log(
        log_dir / ctx.prefix("inference.jsonl"),
        run_id=ctx.run_id,
        model_id=model_id,
    )
    ulog.pipeline("Inference session started", stage="inference")

    # ── Model discovery ───────────────────────────────────────────────
    model_files = sorted(models_dir.glob("*_best_model.pt"))
    if not model_files:
        model_files = sorted(models_dir.glob("*_epoch_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model checkpoint found in {models_dir}")
    latest_model = model_files[-1]
    logger.info("Using model: %s", latest_model.name)

    # ── Tokenizer ─────────────────────────────────────────────────────
    if not tok_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tok_dir}")
    tokenizer = DGBTokenizer.from_pretrained(tok_dir)
    logger.info("Tokenizer: vocab_size=%d", tokenizer.vocab_size)

    # ── Device & model ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(str(latest_model), device, vocab_size=tokenizer.vocab_size)

    # ── Engine ────────────────────────────────────────────────────────
    engine = init_engine(model, tokenizer, device, run_id=ctx.run_id)

    # ── Generate ──────────────────────────────────────────────────────
    gen_cfg = GenerationConfig(
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    test_prompt = prompt or "Hi there!"
    print(f"\n📝 Prompt: {test_prompt}\n")
    print("🤖 Generating...\n")

    output = asyncio.run(engine.complete(test_prompt, gen_cfg))
    print(f"\n✨ Generated Response:\n{output}\n")

    ulog.pipeline("Inference session complete", stage="inference")
    ulog.close()


if __name__ == "__main__":
    import sys
    user_prompt = sys.argv[1] if len(sys.argv) > 1 else None
    run_inference(user_prompt)
