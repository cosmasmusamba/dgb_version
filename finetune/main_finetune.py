"""
finetune/main_finetune.py
==========================
Thin CLI entry point for finetuning runs.

Bootstraps logging, resolves paths and run identity, then delegates all
work to FinetuneTrainer.  Mirrors the structure of model_trainer.py.

Usage
-----
    python finetune/main_finetune.py
    python finetune/main_finetune.py --log-level DEBUG
    python finetune/main_finetune.py --run-id 20260522133713   # resume
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Bootstrap — must happen before any DGB import ─────────────────────────────
_logging_configured = False


def _bootstrap(log_file=None, level: str = "INFO") -> None:
    global _logging_configured
    if _logging_configured:
        return
    from modules.logging_config import configure_logging, set_log_stage
    configure_logging(level=level, log_file=log_file)
    set_log_stage("finetune")
    _logging_configured = True


_bootstrap()   # early console-only boot

# ── Normal imports ─────────────────────────────────────────────────────────────
from configs.loader import get_config
from modules.utils.run_context import create_run_context, get_run_context, RunContext
from modules.utils.path_resolver import init_path_resolver
from modules.utils.pipeline_state import PipelineState
from modules.utils.unified_log import init_unified_log
from finetune.core.finetune_trainer import FinetuneTrainer

logger = logging.getLogger(__name__)


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DGB Finetune")
    p.add_argument("--run-id",    default=None, help="Resume a specific run")
    p.add_argument("--log-level", default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def run_finetuning(
    *,
    ctx:     RunContext = None,
    run_id:  str        = None,
) -> dict:
    """
    Public API — called by main_finetune.py and main_pipeline.py.

    Parameters
    ----------
    ctx:    Shared RunContext (creates a new one if absent).
    run_id: Resume an existing run (overrides ctx.run_id for state lookup).

    Returns
    -------
    dict with "run_id" and "status".
    """
    cfg = get_config()
    ctx = ctx or get_run_context()
    res = init_path_resolver(cfg.project.model_id, cfg)

    # Logging with prefixed file
    log_file = res.logs_dir() / ctx.prefix("finetune.log")
    _bootstrap(log_file=log_file, level=cfg.logging.level)

    # Unified structured log
    ulog_path = res.logs_dir() / ctx.prefix("finetune.jsonl")
    ulog = init_unified_log(ulog_path, run_id=ctx.run_id, model_id=cfg.project.model_id)

    # Pipeline state — load existing if resuming, otherwise fresh
    state_dir = res.logs_dir()
    if run_id:
        state = PipelineState.load_for_run(state_dir, run_id=run_id, model_id=cfg.project.model_id)
    else:
        state = PipelineState.load_latest(state_dir, model_id=cfg.project.model_id)
    state.run_id = ctx.run_id

    # Ensure the finetune stage record exists
    rec = state.get("finetune")

    logger.info(
        "Finetune  run_id=%s  model_id=%s  status=%s",
        ctx.run_id, cfg.project.model_id, rec.status,
    )
    ulog.pipeline(
        f"Finetune started  run_id={ctx.run_id}",
        stage="finetune",
    )

    rec.mark_running()
    state.save(state_dir)

    try:
        trainer = FinetuneTrainer(
            cfg=cfg,
            state=state,
            ctx=ctx,
            resolver=res,
            ulog=ulog,
        )
        trainer.run()

        rec.mark_completed(run_id=ctx.run_id)
        state.save(state_dir)
        ulog.pipeline("Finetune complete", stage="finetune")
        return {"run_id": ctx.run_id, "status": "completed"}

    except Exception as exc:
        rec.mark_failed(str(exc))
        state.save(state_dir)
        ulog.error(f"Finetune failed: {exc}", stage="finetune")
        raise


def main() -> int:
    args = _build_args()
    cfg  = get_config()
    level = args.log_level or cfg.logging.level

    ctx = create_run_context(cfg.project.model_id)
    _bootstrap(level=level)

    logger.info("DGB Finetune v%s  run_id=%s", cfg.project.version, ctx.run_id)

    try:
        result = run_finetuning(ctx=ctx, run_id=args.run_id)
        logger.info("Done: %s", result)
        return 0
    except KeyboardInterrupt:
        logger.info("Finetune interrupted by user")
        return 1
    except Exception as exc:
        logger.error("Finetune failed: %s", exc, exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
