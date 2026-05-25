"""
model_trainer.py
=================
Thin entry point — mirrors main_train_tokenizer.py in structure.

All logic lives in trainer/core/trainer.py.
This file only:
  1. Bootstraps logging (once, via _configure_logging_once)
  2. Resolves paths and run context
  3. Creates DynamicResourceManager (if not injected by main_pipeline.py)
  4. Instantiates DGBTrainer
  5. Calls trainer.run()

Standard DGB utilities used
----------------------------
- get_config / configs.loader            (centralised config)
- configure_logging / LogStage           (consistent logging format)
- create_run_context / get_run_context   (datetime prefix, run_id)
- init_path_resolver                     (external path support)
- DynamicResourceManager                 (adaptive batch / chunk sizing)
- PipelineState                          (stage-level resume)
- init_unified_log                       (single JSONL stream for entire run)

Run with:
    python model_trainer.py

Or via the unified pipeline:
    python main_pipeline.py
"""

from __future__ import annotations

import logging
import sys

# ---------------------------------------------------------------------------
# Logging bootstrap — called ONCE, never duplicated
# ---------------------------------------------------------------------------
_logging_configured = False


def _configure_logging_once(log_file=None, level: str = "INFO") -> None:
    global _logging_configured
    if _logging_configured:
        return
    from modules.logging_config import configure_logging, set_log_stage
    configure_logging(level=level, log_file=log_file, async_file=True)
    set_log_stage("model_training")
    _logging_configured = True


_configure_logging_once()   # early console-only boot

# ---------------------------------------------------------------------------
# Normal imports (after early log setup)
# ---------------------------------------------------------------------------
from configs.loader import get_config
from modules.utils.run_context import RunContext, get_run_context, create_run_context
from modules.utils.path_resolver import PathResolver, init_path_resolver
from modules.utils.pipeline_state import PipelineState
from modules.utils.dynamic_resource_manager import DynamicResourceManager
from modules.utils.system_detector import get_system_profile
from modules.utils.unified_log import init_unified_log, get_unified_log
from trainer.core.trainer import DGBTrainer

logger = logging.getLogger(__name__)


def run_model_training(
    model_id:         str                               = "dgb1",
    *,
    ctx:              RunContext                         = None,
    resolver:         PathResolver                      = None,
    resource_manager: DynamicResourceManager            = None,
    state:            PipelineState                     = None,
) -> dict:
    """
    Public API called by main_pipeline.py.

    Parameters
    ----------
    model_id:         Model identifier.
    ctx:              RunContext (datetime prefix). Created if not provided.
    resolver:         PathResolver (external path support). Created if not provided.
    resource_manager: Live DynamicResourceManager from the pipeline (optional).
                      When absent, DGBTrainer creates its own standalone instance.
    state:            PipelineState for stage-level tracking (optional).

    Returns
    -------
    dict with best_loss and run_id.
    """
    cfg = get_config()
    ctx = ctx or get_run_context()

    if resolver is None:
        resolver = init_path_resolver(model_id, cfg)

    log_dir  = resolver.logs_dir()
    log_file = log_dir / ctx.training_log_name()
    _configure_logging_once(log_file=log_file, level=cfg.logging.level)

    # Process-wide unified log (single JSONL stream for the whole run)
    ulog = get_unified_log()
    if ulog is None:
        ulog = init_unified_log(
            log_dir / ctx.prefix("pipeline.jsonl"),
            run_id=ctx.run_id,
            model_id=model_id,
        )

    # Pipeline state record
    if state is None:
        state = PipelineState.load_latest(log_dir, model_id=model_id)
    rec = state.get("model_training")

    logger.info("model_trainer  run_id=%s  model_id=%s", ctx.run_id, model_id)
    ulog.pipeline(
        f"model_training started  run_id={ctx.run_id}  model_id={model_id}",
        stage="model_training",
    )

    rec.mark_running()
    state.save(log_dir)

    try:
        trainer = DGBTrainer(
            model_id=model_id,
            ctx=ctx,
            resolver=resolver,
            resource_manager=resource_manager,
        )
        best_loss = trainer.run()

        result = {"best_loss": round(best_loss, 4), "run_id": ctx.run_id}

        rec.mark_completed(**result)
        state.save(log_dir)
        ulog.pipeline(
            f"model_training complete  best_loss={result['best_loss']}",
            stage="model_training",
        )

        return result

    except Exception as exc:
        rec.mark_failed(str(exc))
        state.save(log_dir)
        ulog.error(f"model_training failed: {exc}", stage="model_training")
        raise


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = get_config()
    ctx = create_run_context(cfg.project.model_id)
    res = init_path_resolver(cfg.project.model_id, cfg)

    log_file = res.logs_dir() / ctx.training_log_name()
    _configure_logging_once(log_file=log_file, level=cfg.logging.level)

    result = run_model_training(
        model_id=cfg.project.model_id,
        ctx=ctx,
        resolver=res,
    )
    logger.info("Done: %s", result)
