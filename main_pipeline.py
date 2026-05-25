"""
main_pipeline.py
=================
DGB pipeline orchestrator.

Runs the full pipeline:
  1. dataset_clean      — clean raw Wikipedia dump files
  2. train_tokenizer    — train BPE tokenizer on cleaned data
  3. model_training     — pre-train the transformer

CLI flags
---------
  --force STAGE     Reset and re-run a completed stage
  --skip  STAGE     Skip a stage regardless of status
  --status          Print current pipeline state and exit
  --run-id ID       Resume a specific run (default: latest or new)
  --stage STAGE     Run only this one stage
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from configs.loader import get_config
from modules.logging_config import configure_logging, LogStage, TrainingLogger
from modules.utils.run_context import RunContext
from modules.utils.path_resolver import init_path_resolver
from modules.utils.pipeline_state import PipelineState, StageStatus
from modules.utils.unified_log import init_unified_log
from modules.utils.safe_writer import atomic_write_json
from modules.utils.error_handler import PipelineStageError

logger = logging.getLogger(__name__)


def run_dataset_clean(cfg, ctx: RunContext, res, unified_log, state: PipelineState) -> None:
    rec = state.get("dataset_clean")
    rec.mark_running()
    try:
        with LogStage("dataset_clean"):
            from modules.utils.data_cleaner import clean_lines, cleaning_stats
            from modules.utils.chunk_processor import chunk_file_lines
            from modules.utils.file_handler import list_files
            from modules.utils.safe_writer import atomic_write_text

            raw_dir     = res.raw_dir()
            cleaned_dir = res.cleaned_dir()
            raw_files   = list_files(raw_dir, glob="*.txt")  # natural sort: wk_0,1,2..10,11..

            if not raw_files:
                raise FileNotFoundError(
                    f"No .txt files in {raw_dir}. Add Wikipedia dump files."
                )

            logger.info(
                "Dataset cleaning: %d files  order: %s … %s",
                len(raw_files),
                raw_files[0].name,
                raw_files[-1].name,
            )

            total_raw = total_cleaned = 0
            for raw_file in raw_files:
                logger.info("Cleaning: %s", raw_file.name)
                all_cleaned = []
                for chunk in chunk_file_lines(
                    raw_file,
                    chunk_size=cfg.dataset.chunk_size_chars,
                    encoding=cfg.dataset.encoding,
                ):
                    total_raw += len(chunk)
                    cleaned    = list(clean_lines(
                        chunk,
                        min_len=cfg.dataset.min_line_length,
                        max_len=cfg.dataset.max_line_length,
                        dedup=cfg.dataset.dedup,
                    ))
                    all_cleaned.extend(cleaned)

                total_cleaned += len(all_cleaned)
                out_name  = cleaned_dir / ctx.cleaned_file_name(raw_file.name)
                atomic_write_text(out_name, "\n".join(all_cleaned), encoding=cfg.dataset.encoding)
                logger.info("Wrote %d cleaned lines → %s", len(all_cleaned), out_name.name)

            from modules.utils.data_cleaner import cleaning_stats as _stats
            summary = _stats(total_raw, total_cleaned)
            summary_path = cleaned_dir / ctx.cleaning_summary_name()
            atomic_write_json(summary_path, summary)
            unified_log.dataset(
                "Dataset cleaning complete",
                raw_lines=total_raw,
                cleaned_lines=total_cleaned,
                removed_pct=summary["removed_pct"],
            )
            rec.mark_completed(
                raw_lines=total_raw,
                cleaned_lines=total_cleaned,
                files_processed=len(raw_files),
            )
    except Exception as exc:
        rec.mark_failed(str(exc))
        raise PipelineStageError("dataset_clean", exc) from exc


def run_train_tokenizer(cfg, ctx: RunContext, res, unified_log, state: PipelineState) -> None:
    rec = state.get("train_tokenizer")
    rec.mark_running()
    try:
        with LogStage("train_tokenizer"):
            from tokenizer.core.trainer import TokenizerTrainer
            cleaned_dir = res.cleaned_dir()
            tok_dir     = res.tokenizer_dir()

            trainer = TokenizerTrainer()
            trainer.run(cleaned_dir=cleaned_dir, save_dir=tok_dir, run_id=ctx.run_id)

            unified_log.tokenizer(
                "Tokenizer training complete",
                vocab_size=trainer.vocab.size,
                merges=len(trainer.bpe.merges),
            )
            rec.mark_completed(vocab_size=trainer.vocab.size)
    except Exception as exc:
        rec.mark_failed(str(exc))
        raise PipelineStageError("train_tokenizer", exc) from exc


def run_model_training(cfg, ctx: RunContext, res, unified_log, state: PipelineState) -> None:
    rec = state.get("model_training")
    rec.mark_running()
    try:
        with LogStage("model_training"):
            from tokenizer.dgb_tokenizer import DGBTokenizer
            from transformer.core.transformer_model import DGBTransformer
            from transformer.utils.model_helpers import resolve_device, log_model_info
            from trainer.core.checkpoint_manager import CheckpointManager
            from trainer.core.dataset_loader import StreamingTextDataset, build_streaming_loader
            from trainer.core.training_loop import TrainingConfig, run_training_loop
            from modules.utils.metrics_logger import MetricsLogger
            from modules.utils.progress_tracking import ProgressTracker
            from modules.utils.system_detector import get_system_profile
            from modules.utils.dynamic_resource_manager import DynamicResourceManager

            tok_dir    = res.tokenizer_dir()
            log_dir    = res.logs_dir()
            models_dir = res.models_dir()

            # Load tokenizer
            tokenizer = DGBTokenizer.from_pretrained(tok_dir)

            # Device
            device = resolve_device(cfg.training.device)

            # Build model
            tf    = cfg.transformer
            model = DGBTransformer(
                vocab_size=tokenizer.vocab_size,
                d_model=tf.d_model,
                n_heads=tf.n_heads,
                n_encoder_layers=tf.n_encoder_layers,
                n_decoder_layers=tf.n_decoder_layers,
                d_ff=tf.d_ff,
                dropout=tf.dropout,
                max_seq_len=tf.max_seq_len,
                pad_idx=tf.pad_idx,
                tie_embeddings=tf.tie_embeddings,
                layer_norm_eps=tf.layer_norm_eps,
            ).to(device)
            log_model_info(model, "DGBTransformer")

            # Training config — all values from config (B4, T4, T6 fixed)
            train_cfg = TrainingConfig.from_cfg(cfg)

            # Resource manager
            profile = get_system_profile()
            resource = DynamicResourceManager.from_profile(profile, cfg)

            # Checkpoint manager
            ckpt_mgr = CheckpointManager(
                models_dir=models_dir, log_dir=log_dir,
                save_every_epochs=train_cfg.save_every_epochs,
                run_id=ctx.run_id,
            )

            # Metrics + progress
            metrics = MetricsLogger(save_dir=log_dir, run_id=ctx.run_id, model_id=cfg.project.model_id)
            total_est = train_cfg.epochs * 1000
            progress  = ProgressTracker(total=total_est, label="training")

            # Build a light GranularProgress-like namespace for ckpt_mgr
            class _Prog:
                epoch = 0; global_step = 0; batch_idx = 0
                best_loss = float("inf"); clean_file_idx = 0; clean_line_idx = 0
            prog = _Prog()
            ckpt_mgr.restore_progress(prog)

            best_loss = run_training_loop(
                model=model,
                tokenizer=tokenizer,
                cleaned_dir=res.cleaned_dir(),
                device=device,
                train_cfg=train_cfg,
                checkpoint_mgr=ckpt_mgr,
                resource_handle=resource,
                metrics_logger=metrics,
                unified_log=unified_log,
                progress=prog,
                run_id=ctx.run_id,
                model_id=cfg.project.model_id,
            )
            metrics.save()
            rec.mark_completed(best_loss=round(best_loss, 4))
    except Exception as exc:
        rec.mark_failed(str(exc))
        raise PipelineStageError("model_training", exc) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="DGB pipeline orchestrator")
    parser.add_argument("--force",  metavar="STAGE", help="Reset and re-run a stage")
    parser.add_argument("--skip",   metavar="STAGE", help="Skip a stage")
    parser.add_argument("--status", action="store_true", help="Print pipeline status and exit")
    parser.add_argument("--run-id", metavar="ID",    help="Resume a specific run ID")
    parser.add_argument("--stage",  metavar="STAGE", help="Run only this stage")
    args = parser.parse_args()

    cfg = get_config()
    configure_logging(level=cfg.logging.level)

    res = init_path_resolver(cfg.project.model_id, cfg)
    log_dir = res.logs_dir()

    # Handle run context properly
    if args.run_id:
        ctx = RunContext(model_id=cfg.project.model_id, run_id=args.run_id)
    else:
        latest = RunContext.latest_for_model(log_dir, cfg.project.model_id)
        if latest is not None:
            ctx = latest
            logger.info(f"Resuming previous run: {ctx.run_id}")
        else:
            ctx = RunContext(model_id=cfg.project.model_id, run_id=None)
            logger.info(f"Starting new run: {ctx.run_id}")

    state = PipelineState.load_latest(log_dir, cfg.project.model_id)
    state.run_id = ctx.run_id

    if args.status:
        print(state.summary())
        return 0

    # Apply --force / --skip
    if args.force:
        if args.force in state._stages:
            state.get(args.force).reset()
            logger.info("Stage %s reset for re-run", args.force)
        else:
            logger.warning("Unknown stage: %s", args.force)

    if args.skip:
        if args.skip in state._stages:
            state.get(args.skip).mark_skipped("--skip flag")
        else:
            logger.warning("Unknown stage: %s", args.skip)

    unified_log = init_unified_log(
        path=log_dir / ctx.pipeline_log_name(),
        run_id=ctx.run_id,
        model_id=cfg.project.model_id,
    )
    unified_log.pipeline(f"Pipeline started  run_id={ctx.run_id}", stage="pipeline")

    t0     = time.time()
    stages = {
        "dataset_clean":   run_dataset_clean,
        "train_tokenizer": run_train_tokenizer,
        "model_training":  run_model_training,
    }

    if args.stage:
        to_run = {args.stage: stages[args.stage]} if args.stage in stages else {}
    else:
        to_run = stages

    exit_code = 0
    for name, fn in to_run.items():
        rec = state.get(name)
        if rec.is_done:
            logger.info("Stage %s already done — skipping (use --force %s to re-run)", name, name)
            continue
        try:
            fn(cfg, ctx, res, unified_log, state)
            state.save(log_dir)
        except PipelineStageError as exc:
            logger.critical("Pipeline aborted at stage '%s': %s", name, exc)
            state.save(log_dir)
            exit_code = 1
            break

    elapsed = time.time() - t0
    unified_log.pipeline(
        f"Pipeline finished in {elapsed:.1f}s  status={'ok' if exit_code == 0 else 'failed'}",
        stage="pipeline",
    )
    unified_log.close()
    print(state.summary())
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
