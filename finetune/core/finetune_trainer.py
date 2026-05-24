"""
Orchestrates the finetuning workflow.
Integrates dataset loader, loop, resource manager, logging, metrics, progress tracking and checkpoint lineage.
"""
from typing import Any, Dict

from finetune.core.finetune_dataset_loader import FinetuneDatasetLoader
from finetune.core.finetune_loop import run_finetune_loop
from finetune.utils.schema_validator import validate_dataset
from finetune.utils.checkpoint_lineage import record_lineage

from modules.utils.dynamic_resource_manager import AdaptiveResourceManager
from modules.utils.unified_log import UnifiedLogger
from modules.utils.progress_tracking import ProgressTracker
from modules.utils.metrics_logger import MetricsLogger
from modules.utils.error_handler import handle_error
from modules.utils.safe_writer import safe_write
from modules.utils.file_locker import FileLocker
from modules.utils.run_context import RunContext
from modules.utils.pipeline_state import PipelineState

class FinetuneTrainer:
    def __init__(self, config: Dict[str, Any], state: PipelineState, run_ctx: RunContext):
        self.config = config
        self.state = state
        self.run_ctx = run_ctx

        # Utilities from modules.utils
        self.logger = UnifiedLogger(component="finetune", run_ctx=run_ctx)
        self.resource_manager = AdaptiveResourceManager(config)
        self.progress = ProgressTracker(namespace="finetune", run_ctx=run_ctx)
        self.metrics = MetricsLogger(namespace="finetune", run_ctx=run_ctx)
        self.file_locker = FileLocker()
        # ensure checkpoint directory exists and is safe to write
        safe_write(self.run_ctx.ensure_dir("checkpoints/logs/finetune"), "{}")

    def run(self):
        """
        Main orchestration method:
        - Validate dataset
        - Run finetune loop with adaptive resource management
        - Record checkpoint lineage and final state
        """
        try:
            dataset_loader = FinetuneDatasetLoader(self.config, run_ctx=self.run_ctx)
            # Validate a small sample (stream-safe)
            sample_iter = dataset_loader.load_sample(self.config["finetune"].get("validation_sample_size", 10))
            validate_dataset(sample_iter)

            self.logger.log_event("finetune_start", {"config_snapshot": self.config["finetune"], "run_id": self.run_ctx.run_id})

            # Acquire a lock for this finetune run to avoid concurrent writes
            lock_path = self.run_ctx.path("locks/finetune.lock")
            with self.file_locker.lock(lock_path):
                run_finetune_loop(
                    dataset_loader=dataset_loader,
                    config=self.config,
                    resource_manager=self.resource_manager,
                    logger=self.logger,
                    progress=self.progress,
                    metrics=self.metrics,
                    state=self.state,
                    run_ctx=self.run_ctx
                )

            # Persist lineage and final checkpoint metadata
            record_lineage(self.config, self.state, run_ctx=self.run_ctx)
            self.logger.log_event("finetune_complete", {"status": "success", "run_id": self.run_ctx.run_id})

        except Exception as exc:
            # Centralized error handling and logging
            handle_error(component="finetune", error=exc, run_ctx=self.run_ctx)
            self.logger.log_event("finetune_failed", {"error": str(exc), "run_id": self.run_ctx.run_id})
            raise
