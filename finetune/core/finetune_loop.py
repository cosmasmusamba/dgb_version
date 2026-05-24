# finetune/core/finetune_loop.py
"""
Training loop for finetuning.
This module delegates training steps to the LoRA adapter and performs checkpointing
using the adapter.save_checkpoint method at configured intervals. It integrates
resource adjustments, progress tracking, metrics logging and unified logging.
"""

from typing import Any, Dict

from finetune.core.lora_adapter import LoRAAdapter
from modules.utils.progress_tracking import ProgressTracker
from modules.utils.dynamic_resource_manager import AdaptiveResourceManager
from modules.utils.unified_log import UnifiedLogger
from modules.utils.metrics_logger import MetricsLogger
from modules.utils.pipeline_state import PipelineState
from modules.utils.run_context import RunContext
from finetune.core.finetune_dataset_loader import FinetuneDatasetLoader


def run_finetune_loop(
    dataset_loader: FinetuneDatasetLoader,
    config: Dict[str, Any],
    resource_manager: AdaptiveResourceManager,
    logger: UnifiedLogger,
    progress: ProgressTracker,
    metrics: MetricsLogger,
    state: PipelineState,
    run_ctx: RunContext
):
    """
    Execute the finetuning loop.

    - Delegates per-batch training to LoRAAdapter.train_step
    - Uses LoRAAdapter.save_checkpoint to persist model/optimizer/scaler state
    - Uses ProgressTracker and PipelineState to persist offsets for resumability
    - Uses AdaptiveResourceManager to recommend batch sizes at epoch boundaries
    """
    fin_cfg = config["finetune"]
    adapter = LoRAAdapter(config, run_ctx=run_ctx)

    epochs = int(fin_cfg.get("epochs", 1))
    base_batch_size = int(fin_cfg.get("batch_size", 8))
    checkpoint_interval = int(fin_cfg.get("checkpoint_interval_steps", 500))

    # resume-aware counters
    global_step = int(state.get("global_step", 0))
    start_epoch = int(state.get("epoch", 0))
    start_batch_offset = int(state.get("batch_offset", 0))

    logger.log_event("finetune_loop_start", {
        "run_id": run_ctx.run_id,
        "start_epoch": start_epoch,
        "start_batch_offset": start_batch_offset,
        "global_step": global_step
    })

    for epoch in range(start_epoch, epochs):
        # allow resource manager to adjust safe parameters at epoch boundary
        resource_manager.adjust(epoch)
        batch_size = resource_manager.recommend_batch_size(base_batch_size)

        logger.log_event("epoch_start", {"epoch": epoch, "batch_size": batch_size})

        # stream batches with resume support
        for batch_idx, batch in enumerate(dataset_loader.stream_batches(batch_size, start_offset=start_batch_offset)):
            # perform a training step via the adapter
            loss, extra = adapter.train_step(batch, epoch=epoch, step=global_step)

            # persist progress and metrics
            progress.save(epoch=epoch, batch=batch_idx, step=global_step)
            metrics.log({"epoch": epoch, "step": global_step, "loss": float(loss), **(extra or {})})
            logger.log_event("finetune_batch", {"epoch": epoch, "step": global_step, "loss": float(loss)})

            # update pipeline state for resumability
            global_step += 1
            state.set("global_step", global_step)
            state.set("batch_offset", batch_idx + 1)

            # checkpoint at configured intervals using adapter.save_checkpoint
            if checkpoint_interval > 0 and (global_step % checkpoint_interval) == 0:
                ckpt_meta = {"step": global_step, "epoch": epoch, "note": f"autosave_step_{global_step}"}
                try:
                    ckpt_path = adapter.save_checkpoint(state=ckpt_meta, name=f"autosave_step_{global_step}.pt")
                    state.set("last_checkpoint_path", ckpt_path)
                    logger.log_event("checkpoint_autosave", {"path": ckpt_path, "step": global_step})
                except Exception as e:
                    logger.log_event("checkpoint_autosave_failed", {"error": str(e), "step": global_step})

        # epoch complete: reset batch offset and optionally checkpoint
        state.set("epoch", epoch + 1)
        state.set("batch_offset", 0)
        logger.log_event("epoch_end", {"epoch": epoch, "last_loss": float(loss)})

        # epoch-level checkpoint
        try:
            ckpt_meta = {"step": global_step, "epoch": epoch + 1, "note": f"epoch_{epoch+1}"}
            ckpt_path = adapter.save_checkpoint(state=ckpt_meta, name=f"epoch_{epoch+1}.pt")
            state.set("last_checkpoint_path", ckpt_path)
            logger.log_event("checkpoint_epoch", {"path": ckpt_path, "epoch": epoch + 1})
        except Exception as e:
            logger.log_event("checkpoint_epoch_failed", {"error": str(e), "epoch": epoch + 1})

        # after first epoch resume, ensure subsequent batches start from 0
        start_batch_offset = 0

    # final checkpoint at the end of training
    try:
        final_meta = {"step": global_step, "epoch": epochs, "note": "finetune_complete"}
        final_path = adapter.save_checkpoint(state=final_meta, name=f"finetune_complete_{run_ctx.run_id}.pt")
        state.set("last_checkpoint_path", final_path)
        logger.log_event("checkpoint_final", {"path": final_path})
    except Exception as e:
        logger.log_event("checkpoint_final_failed", {"error": str(e)})

    # final state save
    state.save_checkpoint(note="finetune_finished")
    logger.log_event("finetune_loop_end", {"run_id": run_ctx.run_id, "global_step": global_step})
