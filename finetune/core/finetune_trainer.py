"""
finetune/core/finetune_trainer.py
==================================
Orchestrates the finetuning workflow.

Integrates every standard DGB utility in the correct way:
  - DynamicResourceManager      (adaptive batch/chunk sizing)
  - UnifiedLogWriter            (structured JSONL event stream)
  - ProgressTracker             (ETA + atomic offset persistence)
  - MetricsLogger               (step/epoch metrics + loss curves)
  - PipelineState               (stage + KV sub-stage state)
  - locked_file                 (cross-process checkpoint locking)
  - DeviceMonitor / SleepGuard  (hardware telemetry + sleep prevention)
  - configure_logging / LogStage

This module NEVER references:
  - AdaptiveResourceManager  (does not exist — use DynamicResourceManager)
  - UnifiedLogger            (does not exist — use UnifiedLogWriter)
  - safe_write               (does not exist — use atomic_write_json)
  - handle_error             (does not exist — use log_exception / handle_errors)
  - FileLocker               (is not a class — use locked_file context manager)
  - RunContext.from_config   (does not exist — ctx is injected from main_finetune.py)
  - RunContext.ensure_dir    (does not exist — use ensure_dir() utility)
  - RunContext.path          (does not exist — use resolver.* or Path directly)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from configs.loader import get_config, DGBConfig
from modules.logging_config import set_log_stage
from modules.utils.run_context import RunContext, get_run_context
from modules.utils.path_resolver import PathResolver
from modules.utils.pipeline_state import PipelineState
from modules.utils.unified_log import UnifiedLogWriter, get_unified_log
from modules.utils.progress_tracking import ProgressTracker
from modules.utils.metrics_logger import MetricsLogger
from modules.utils.dynamic_resource_manager import DynamicResourceManager
from modules.utils.device_monitor import DeviceMonitor, SleepGuard
from modules.utils.system_detector import get_system_profile
from modules.utils.file_handler import ensure_dir
from modules.utils.file_locker import locked_file
from modules.utils.error_handler import log_exception
from finetune.core.finetune_loop import run_finetune_loop
from finetune.core.finetune_dataset_loader import FinetuneDatasetLoader
from finetune.utils.schema_validator import validate_dataset
from finetune.utils.checkpoint_lineage import record_lineage

logger = logging.getLogger(__name__)


class FinetuneTrainer:
    """
    Assembles and runs the full finetuning pipeline.

    Responsibilities
    ----------------
    - System detection and adaptive parameter seeding
    - DeviceMonitor + SleepGuard lifecycle
    - DynamicResourceManager lifecycle
    - Dataset validation
    - Cross-process file locking for checkpoints
    - Delegation to run_finetune_loop
    - Final lineage recording

    Parameters
    ----------
    cfg:      DGBConfig from configs/loader.get_config().
    state:    PipelineState (stage records + KV sub-stage offsets).
    ctx:      RunContext (datetime prefix, run_id).
    resolver: PathResolver (external path support).
    ulog:     UnifiedLogWriter (shared across the whole run).
    """

    def __init__(
        self,
        cfg:      DGBConfig,
        state:    PipelineState,
        ctx:      RunContext,
        resolver: PathResolver,
        ulog:     Optional[UnifiedLogWriter] = None,
    ) -> None:
        self._cfg      = cfg
        self._state    = state
        self._ctx      = ctx
        self._res      = resolver

        # Shared unified log (fall back to the process singleton)
        self._ulog = ulog or get_unified_log()

        set_log_stage("finetune")

        # Directories
        self._ckpt_dir  = resolver.models_dir()
        self._log_dir   = resolver.logs_dir()
        self._lock_dir  = self._log_dir / "locks"
        ensure_dir(self._ckpt_dir)
        ensure_dir(self._log_dir)
        ensure_dir(self._lock_dir)

        # System profile → seed resource manager
        self._profile = get_system_profile()
        p = self._profile

        # Metrics logger (run-id prefixed files)
        self._metrics = MetricsLogger(
            save_dir=self._log_dir,
            model_id=cfg.project.model_id,
            run_id=ctx.run_id,
        )

        # Device monitoring
        self._monitor = DeviceMonitor(interval=30, profile=p)
        self._guard   = SleepGuard(keep_display_on=False)

        # Resource manager — seeded from system profile, adapts during training
        self._rm = DynamicResourceManager(
            interval=15,
            initial_batch=p.recommended_batch_size,
            initial_chunk=getattr(p, "recommended_chunk_size_chars", 500_000),
            initial_workers=p.recommended_num_workers,
            initial_pin_mem=p.recommended_pin_memory,
            gpu_device=p.recommended_device,
            max_batch=512,
            min_batch=1,
        )

        logger.info(
            "FinetuneTrainer init  run_id=%s  model=%s  device=%s  batch=%d",
            ctx.run_id, cfg.project.model_id,
            p.recommended_device, p.recommended_batch_size,
        )

    def run(self) -> None:
        """
        Execute the full finetuning pipeline:
          1. Start monitoring and resource management.
          2. Validate a sample of the dataset.
          3. Acquire checkpoint lock and run the training loop.
          4. Record checkpoint lineage.
        """
        set_log_stage("finetune")
        self._guard.enable()
        self._monitor.start()
        self._rm.start()

        if self._ulog:
            self._ulog.finetune(
                f"Finetune started  run_id={self._ctx.run_id}",
                epoch=0, step=0,
            )

        try:
            # Dataset
            dataset_loader = FinetuneDatasetLoader(
                config=self._cfg, run_ctx=self._ctx
            )
            val_size = getattr(
                getattr(self._cfg, "finetune", None), "validation_sample_size", 10
            ) if hasattr(self._cfg, "finetune") else 10
            sample_iter = dataset_loader.load_sample(val_size)
            validate_dataset(sample_iter)
            logger.info("Dataset validation passed")

            if self._ulog:
                self._ulog.finetune("Dataset validated", epoch=0, step=0)

            # Cross-process checkpoint lock
            lock_path = self._lock_dir / "finetune.lock"
            with locked_file(lock_path):
                run_finetune_loop(
                    dataset_loader=dataset_loader,
                    cfg=self._cfg,
                    resource_manager=self._rm,
                    metrics=self._metrics,
                    state=self._state,
                    ctx=self._ctx,
                    ckpt_dir=self._ckpt_dir,
                    log_dir=self._log_dir,
                    ulog=self._ulog,
                )

            # Lineage
            record_lineage(self._cfg, self._state, run_ctx=self._ctx)
            if self._ulog:
                self._ulog.finetune(
                    "Finetune complete",
                    step=self._state.get_kv("global_step", 0),
                    epoch=self._state.get_kv("epoch", 0),
                )
            logger.info("Finetune complete  run_id=%s", self._ctx.run_id)

        except Exception as exc:
            log_exception(exc, context="FinetuneTrainer.run")
            if self._ulog:
                self._ulog.error(f"Finetune error: {exc}", stage="finetune")
            raise

        finally:
            self._monitor.stop()
            self._guard.disable()
            self._rm.stop()
