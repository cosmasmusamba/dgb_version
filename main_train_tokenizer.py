"""
main_train_tokenizer.py
========================
Full tokenizer training pipeline with:
  - DynamicResourceManager     (adapts chunk size to live RAM)
  - ProgressTracker + save()   (granular file-offset resume)
  - UnifiedLogWriter           (structured event stream)
  - PipelineState              (stage-level resume + KV sub-stage offsets)
  - DeviceMonitor              (hardware telemetry)

Output artefacts carry the RunContext datetime prefix:
    <run_id>_vocabulary.json
    <run_id>_vocab_meta.json
    <run_id>_bpe_merges.json
    <run_id>_tokenizer_progress.json   ← granular resume offset

Input cleaned files are discovered sorted by name so they are processed
in chronological order across multiple cleaning runs.

Run AFTER main_dataset_clean.py:
    python main_train_tokenizer.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_logging_configured = False


def _bootstrap_logging(log_file: Optional[Path] = None, level: str = "INFO") -> None:
    global _logging_configured
    if _logging_configured:
        return
    from modules.logging_config import configure_logging, set_log_stage
    configure_logging(level=level, log_file=log_file)
    set_log_stage("train_tokenizer")
    _logging_configured = True


_bootstrap_logging()   # early console-only boot

# ── Normal imports ─────────────────────────────────────────────────────────────
from configs.loader import get_config
from modules.logging_config import LogStage, set_log_stage
from modules.utils.error_handler import DataError, log_exception
from modules.utils.file_handler import list_files, ensure_dir
from modules.utils.streaming import get_training_hub, StreamEvent
from modules.utils.run_context import RunContext, get_run_context
from modules.utils.path_resolver import PathResolver, get_path_resolver
from modules.utils.pipeline_state import PipelineState
from modules.utils.progress_tracking import ProgressTracker
from modules.utils.dynamic_resource_manager import DynamicResourceManager
from modules.utils.device_monitor import DeviceMonitor
from modules.utils.system_detector import get_system_profile
from modules.utils.unified_log import get_unified_log, init_unified_log
from tokenizer.core.trainer import TokenizerTrainer

logger = logging.getLogger(__name__)


def run_tokenizer_training(
    cleaned_dir:      Optional[Path]                     = None,
    save_dir:         Optional[Path]                     = None,
    ctx:              Optional[RunContext]                = None,
    resolver:         Optional[PathResolver]             = None,
    resource_manager: Optional[DynamicResourceManager]  = None,
    state:            Optional[PipelineState]            = None,
) -> dict:
    """
    Train BPE tokenizer from all cleaned corpus files and save artefacts.

    Files in cleaned_dir are sorted by name so all runs' cleaned files
    are processed in chronological order.

    Granular resume
    ---------------
    A ProgressCheckpoint is written after each file is tokenized so the
    pipeline can resume from the last successfully processed file on crash.

    Adaptive resources
    ------------------
    When a DynamicResourceManager is provided (or created standalone),
    chunk_size for the tokenizer trainer is read from the live resource
    handle at each file boundary.

    Output files:
        <save_dir>/<run_id>_vocabulary.json
        <save_dir>/<run_id>_vocab_meta.json
        <save_dir>/<run_id>_bpe_merges.json
    """
    with LogStage("train_tokenizer"):
        cfg         = get_config()
        ctx         = ctx or get_run_context()

        try:
            res = resolver or get_path_resolver()
        except RuntimeError:
            from modules.utils.path_resolver import PathResolver, init_path_resolver
            res = init_path_resolver(cfg.project.model_id, cfg)

        cleaned_dir = cleaned_dir or res.cleaned_dir()
        save_dir    = save_dir    or res.tokenizer_dir()
        log_dir     = res.logs_dir()

        ensure_dir(save_dir)
        ensure_dir(log_dir)

        # Unified log (use existing or init)
        ulog = get_unified_log()
        if ulog is None:
            ulog = init_unified_log(
                log_dir / ctx.prefix("tokenizer.jsonl"),
                run_id=ctx.run_id,
                model_id=cfg.project.model_id,
            )

        # Pipeline state record
        if state is None:
            state = PipelineState.load_latest(log_dir, model_id=cfg.project.model_id)
        rec = state.get("train_tokenizer")

        # File discovery (sorted = chronological across cleaning runs)
        files = sorted(list_files(cleaned_dir, glob="*.txt"))
        if not files:
            raise DataError(
                f"No cleaned .txt files in {cleaned_dir}.\n"
                "Run  python main_dataset_clean.py  first."
            )

        # Granular resume: skip already-tokenized files
        progress_path    = log_dir / ctx.prefix("tokenizer_progress.json")
        offset_ckpt      = ProgressTracker.load_checkpoint(progress_path)
        files_done       = offset_ckpt.line_offset   # reuse line_offset as file_idx
        if files_done > 0:
            logger.info(
                "Tokenizer resume: skipping first %d already-processed file(s)",
                files_done,
            )
            files = files[files_done:]

        hub = get_training_hub()
        hub.publish(StreamEvent.status(
            f"Tokenizer training started — run_id={ctx.run_id}"
        ))
        ulog.tokenizer(
            f"Training started  run_id={ctx.run_id}  files={len(files)}",
            vocab_size=cfg.tokenizer.vocab_size,
        )

        logger.info(
            "Tokenizer training: %d file(s)  run_id=%s", len(files), ctx.run_id
        )
        logger.info(
            "Config: vocab_size=%d  num_merges=%d  min_freq=%d",
            cfg.tokenizer.vocab_size, cfg.tokenizer.num_merges, cfg.tokenizer.min_freq,
        )

        # System profile → seed resource manager
        profile = get_system_profile()

        # Standalone resource manager if not injected
        standalone_rm = False
        if resource_manager is None:
            resource_manager = DynamicResourceManager(
                interval=15,
                initial_batch=profile.recommended_batch_size,
                initial_chunk=profile.recommended_chunk_size_chars,
                initial_workers=profile.recommended_num_workers,
                gpu_device=profile.recommended_device,
            )
            resource_manager.start()
            standalone_rm = True

        # Device monitor
        monitor = DeviceMonitor(interval=30, profile=profile)
        monitor.start()

        tracker = ProgressTracker(
            total=len(files),
            label="train_tokenizer",
            log_every=1,
        )

        try:
            rec.mark_running()
            state.save(log_dir)

            trainer = TokenizerTrainer(ctx=ctx)

            for file_idx, f in enumerate(files):
                # Adaptive chunk size from live resource manager
                chunk_size = resource_manager.handle.chunk_size_chars
                pressure   = resource_manager.handle.pressure

                logger.info(
                    "Tokenizing file %d/%d: %s  chunk=%d  pressure=%s",
                    file_idx + 1, len(files), f.name, chunk_size, pressure,
                )
                hub.publish(StreamEvent.status(
                    f"Tokenizing {f.name} ({file_idx + 1}/{len(files)})"
                ))
                ulog.tokenizer(
                    f"file_start  {f.name}",
                    file_idx=file_idx,
                    chunk_size=chunk_size,
                )

                try:
                    trainer.add_file(f, chunk_size=chunk_size)
                except Exception as exc:
                    log_exception(exc, context=f"TokenizerTrainer.add_file {f.name}")
                    ulog.error(f"add_file failed: {f.name}: {exc}", stage="train_tokenizer")
                    raise

                # Atomic file-offset checkpoint
                tracker.save(
                    progress_path,
                    epoch=0,
                    batch_idx=0,
                    global_step=file_idx + 1,
                    line_offset=files_done + file_idx + 1,   # absolute file index
                    note=f"after_{f.name}",
                )
                state.set("tokenizer_files_done", files_done + file_idx + 1)
                state.save_checkpoint(log_dir, note=f"tokenizer_file_{file_idx}")

                tracker.update()
                ulog.tokenizer(f"file_done  {f.name}", file_idx=file_idx)

            # Build and save vocabulary
            vocab = trainer.build_and_save(save_dir=save_dir)

            result = {
                "run_id":     ctx.run_id,
                "vocab_size": vocab.size,
                "merges":     len(trainer.bpe.merges),
                "files":      len(files) + files_done,
            }

            rec.mark_completed(**result)
            state.save(log_dir)
            tracker.done()

            logger.info("Tokenizer complete — %s", result)
            hub.publish(StreamEvent.done("Tokenizer training complete"))
            ulog.tokenizer("complete", **result)

            # Verify artefacts
            for fname in (ctx.vocab_name(), ctx.vocab_meta_name(), ctx.bpe_merges_name()):
                path   = save_dir / fname
                status = "✓" if path.exists() else "✗ MISSING"
                logger.info("  %s %s", status, fname)

            return result

        except Exception as exc:
            rec.mark_failed(str(exc))
            state.save(log_dir)
            ulog.error(f"Tokenizer failed: {exc}", stage="train_tokenizer")
            raise

        finally:
            monitor.stop()
            if standalone_rm:
                resource_manager.stop()


if __name__ == "__main__":
    from modules.utils.run_context import create_run_context
    from modules.utils.path_resolver import init_path_resolver

    cfg = get_config()
    ctx = create_run_context(cfg.project.model_id)
    res = init_path_resolver(cfg.project.model_id, cfg)

    log_file = res.logs_dir() / ctx.training_log_name()
    _bootstrap_logging(log_file=log_file, level=cfg.logging.level)

    run_tokenizer_training(ctx=ctx, resolver=res)
