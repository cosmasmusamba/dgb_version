"""
finetune/core/finetune_loop.py
================================
Finetuning training loop with full granular resumability.

All sub-stage offsets are stored atomically via the PipelineState KV API
and ProgressTracker.save() so exact line/batch/epoch resume is possible
after any crash or interruption.

Resume strategy
---------------
On start, the loop reads:
    global_step    → resume absolute step counter
    epoch          → resume from this epoch (0-indexed)
    batch_offset   → skip this many batches in the first resumed epoch

These values are updated AFTER every batch/checkpoint and persisted to:
    <log_dir>/<run_id>_finetune_progress.json  (ProgressTracker offset)
    <log_dir>/<run_id>_pipeline_state.json      (PipelineState KV)

Checkpoint naming follows the RunContext convention:
    <ckpt_dir>/<run_id>_finetune_step_<N>.pt
    <ckpt_dir>/<run_id>_finetune_epoch_<N>.pt
    <ckpt_dir>/<run_id>_finetune_complete.pt
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from configs.loader import DGBConfig
from modules.utils.pipeline_state import PipelineState
from modules.utils.progress_tracking import ProgressTracker
from modules.utils.metrics_logger import MetricsLogger
from modules.utils.dynamic_resource_manager import DynamicResourceManager
from modules.utils.unified_log import UnifiedLogWriter
from modules.utils.run_context import RunContext
from modules.utils.error_handler import log_exception
from finetune.core.finetune_dataset_loader import FinetuneDatasetLoader
from finetune.core.lora_adapter import LoRAAdapter

logger = logging.getLogger(__name__)


def run_finetune_loop(
    *,
    dataset_loader:   FinetuneDatasetLoader,
    cfg:              DGBConfig,
    resource_manager: DynamicResourceManager,
    metrics:          MetricsLogger,
    state:            PipelineState,
    ctx:              RunContext,
    ckpt_dir:         Path,
    log_dir:          Path,
    ulog:             Optional[UnifiedLogWriter] = None,
) -> None:
    """
    Execute the finetuning loop.

    Granular resume
    ---------------
    - epoch, global_step, batch_offset are read from PipelineState KV.
    - ProgressTracker.save() writes an atomic JSON offset after every
      checkpoint_interval steps AND at every epoch boundary.
    - LoRAAdapter.save_checkpoint() persists model/optimizer/scaler weights.

    Adaptive resources
    ------------------
    - resource_manager.handle.batch_size is re-read at every epoch boundary
      so the DynamicResourceManager can adjust without a restart.
    """
    # ── Config ────────────────────────────────────────────────────────
    # Finetune config lives either as cfg.finetune or a raw dict
    fin_cfg = getattr(cfg, "finetune", None)
    if fin_cfg is None:
        fin_cfg = {}
    if hasattr(fin_cfg, "__dict__"):
        fin_cfg = fin_cfg.__dict__

    epochs              = int(fin_cfg.get("epochs", 1))
    base_batch_size     = int(fin_cfg.get("batch_size", 8))
    checkpoint_interval = int(fin_cfg.get("checkpoint_interval_steps", 500))

    # ── Restore sub-stage offsets ─────────────────────────────────────
    global_step        = int(state.get_kv("global_step",    0))
    start_epoch        = int(state.get_kv("epoch",          0))
    start_batch_offset = int(state.get_kv("batch_offset",   0))

    # Progress offset file for line/batch-level resume
    progress_path = log_dir / ctx.prefix("finetune_progress.json")
    state_dir     = log_dir

    # Dummy total for the outer ProgressTracker (epoch-level)
    epoch_tracker = ProgressTracker(
        total=epochs,
        label="finetune_epochs",
        log_every=1,
    )
    # Align tracker with resume point
    if start_epoch > 0:
        epoch_tracker._current = start_epoch

    adapter = LoRAAdapter(cfg, run_ctx=ctx)

    if ulog:
        ulog.finetune(
            "finetune_loop_start",
            epoch=start_epoch, step=global_step,
            start_batch_offset=start_batch_offset,
        )
    logger.info(
        "Finetune loop start  epochs=%d  resume: epoch=%d  step=%d  batch_offset=%d",
        epochs, start_epoch, global_step, start_batch_offset,
    )

    last_loss: float = 0.0

    for epoch in range(start_epoch, epochs):
        # ── Adaptive batch size at epoch boundary ──────────────────────
        handle     = resource_manager.handle
        batch_size = min(base_batch_size, handle.batch_size)
        pressure   = handle.pressure

        if ulog:
            ulog.finetune(
                "epoch_start",
                epoch=epoch, step=global_step,
                batch_size=batch_size, pressure=pressure,
            )
        logger.info("Epoch %d/%d  batch_size=%d  pressure=%s", epoch + 1, epochs, batch_size, pressure)

        batch_tracker = ProgressTracker(
            total=0,       # unknown — dataset may be a stream
            label=f"finetune_epoch_{epoch}",
            log_every=checkpoint_interval,
        )

        for batch_idx, batch in enumerate(
            dataset_loader.stream_batches(batch_size, start_offset=start_batch_offset)
        ):
            try:
                loss, extra = adapter.train_step(batch, epoch=epoch, step=global_step)
                last_loss   = float(loss)
            except Exception as exc:
                log_exception(exc, context=f"train_step epoch={epoch} step={global_step}")
                if ulog:
                    ulog.error(
                        f"train_step failed: {exc}",
                        stage="finetune", epoch=epoch, step=global_step,
                    )
                raise

            # ── Metrics ────────────────────────────────────────────────
            metrics.log({
                "epoch":        epoch,
                "step":         global_step,
                "loss":         last_loss,
                **(extra or {}),
            })

            if ulog:
                ulog.finetune(
                    "batch",
                    epoch=epoch, step=global_step,
                    loss=last_loss, batch_size=batch_size,
                )

            batch_tracker.update()

            # ── Advance counters ───────────────────────────────────────
            global_step += 1
            state.set("global_step",  global_step)
            state.set("batch_offset", batch_idx + 1)

            # ── Interval checkpoint ────────────────────────────────────
            if checkpoint_interval > 0 and (global_step % checkpoint_interval) == 0:
                ckpt_name = ctx.prefix(f"finetune_step_{global_step}.pt")
                ckpt_meta = {
                    "step": global_step, "epoch": epoch,
                    "note": f"autosave_step_{global_step}",
                }
                try:
                    ckpt_path = adapter.save_checkpoint(state=ckpt_meta, name=ckpt_name)
                    state.set("last_checkpoint_path", str(ckpt_path))

                    # Atomic progress offset
                    batch_tracker.save(
                        progress_path,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=global_step,
                        note=f"step_{global_step}",
                    )
                    state.save_checkpoint(state_dir, note=f"step_{global_step}")

                    if ulog:
                        ulog.finetune(
                            "checkpoint_autosave",
                            epoch=epoch, step=global_step,
                            ckpt_path=str(ckpt_path),
                        )
                    logger.info("Checkpoint → %s", ckpt_name)
                except Exception as exc:
                    log_exception(exc, context="checkpoint_autosave")
                    if ulog:
                        ulog.error(
                            f"checkpoint_autosave failed: {exc}",
                            stage="finetune", step=global_step,
                        )

        # ── Epoch end ─────────────────────────────────────────────────
        metrics.log_epoch(epoch=epoch, avg_loss=last_loss)
        epoch_tracker.update()

        state.set("epoch",        epoch + 1)
        state.set("batch_offset", 0)

        if ulog:
            ulog.finetune(
                "epoch_end",
                epoch=epoch, step=global_step,
                loss=last_loss,
            )
        logger.info("Epoch %d complete  loss=%.4f", epoch + 1, last_loss)

        # ── Epoch checkpoint ───────────────────────────────────────────
        epoch_ckpt_name = ctx.prefix(f"finetune_epoch_{epoch + 1}.pt")
        try:
            ckpt_meta = {
                "step": global_step, "epoch": epoch + 1,
                "note": f"epoch_{epoch + 1}",
            }
            ckpt_path = adapter.save_checkpoint(state=ckpt_meta, name=epoch_ckpt_name)
            state.set("last_checkpoint_path", str(ckpt_path))

            batch_tracker.save(
                progress_path,
                epoch=epoch + 1,
                batch_idx=0,
                global_step=global_step,
                note=f"epoch_{epoch + 1}_end",
            )
            state.save_checkpoint(state_dir, note=f"epoch_{epoch + 1}_end")

            if ulog:
                ulog.finetune(
                    "checkpoint_epoch",
                    epoch=epoch + 1, step=global_step,
                    ckpt_path=str(ckpt_path),
                )
        except Exception as exc:
            log_exception(exc, context=f"epoch_checkpoint epoch={epoch}")
            if ulog:
                ulog.error(
                    f"epoch checkpoint failed: {exc}",
                    stage="finetune", epoch=epoch + 1,
                )

        # After the first resumed epoch, start batches from 0
        start_batch_offset = 0

    # ── Final checkpoint ───────────────────────────────────────────────
    final_name = ctx.prefix("finetune_complete.pt")
    try:
        final_meta = {
            "step": global_step, "epoch": epochs,
            "note": "finetune_complete",
        }
        final_path = adapter.save_checkpoint(state=final_meta, name=final_name)
        state.set("last_checkpoint_path", str(final_path))
        epoch_tracker.save(
            progress_path,
            epoch=epochs,
            batch_idx=0,
            global_step=global_step,
            note="finetune_complete",
        )
        state.save_checkpoint(state_dir, note="finetune_finished")

        if ulog:
            ulog.finetune(
                "checkpoint_final",
                epoch=epochs, step=global_step,
                ckpt_path=str(final_path),
            )
        logger.info("Final checkpoint → %s", final_name)
    except Exception as exc:
        log_exception(exc, context="final_checkpoint")
        if ulog:
            ulog.error(f"final checkpoint failed: {exc}", stage="finetune")

    epoch_tracker.done()
    if ulog:
        ulog.finetune(
            "finetune_loop_end",
            epoch=epochs, step=global_step,
        )
    logger.info("Finetune loop complete  global_step=%d", global_step)
