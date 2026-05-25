"""
main_dataset_clean.py
======================
Wikipedia dump cleaning pipeline with:
  - File-level resume         (skip already-cleaned files on restart)
  - Atomic PipelineState KV   (granular sub-stage offset persistence)
  - ProgressTracker.save()    (file-offset checkpoint for exact resume)
  - DynamicResourceManager    (live RAM-adaptive chunk sizing)
  - UnifiedLogWriter          (structured JSONL event stream)
  - DeviceMonitor             (hardware telemetry)

Source file naming: wk_N.txt  (wk = Wikipedia dump folder abbreviation)
Output file naming: 20260510202040_dgb1_cleaned_wk_N.txt

Resume behaviour
----------------
On restart, the pipeline reads <run_id>_cleaning_progress.json.
Any source file whose corresponding cleaned output already exists on
disk is skipped automatically — no re-cleaning.
The ProgressTracker offset provides additional sub-file granularity
(line_offset) that TokenizerTrainer.add_file can honour on deep resume.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Callable

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_logging_configured = False


def _bootstrap(level: str = "INFO", log_file=None):
    global _logging_configured
    if _logging_configured:
        return
    from modules.logging_config import configure_logging, set_log_stage
    configure_logging(level=level, log_file=log_file)
    set_log_stage("dataset_clean")
    _logging_configured = True


_bootstrap()   # early console-only boot

# ── Normal imports ─────────────────────────────────────────────────────────────
from configs.loader import get_config
from modules.logging_config import LogStage, set_log_stage
from modules.utils.error_handler import DataError, log_exception
from modules.utils.file_handler import list_files, ensure_dir, write_text
from modules.utils.data_cleaner import clean_lines, cleaning_stats
from modules.utils.chunk_processor import chunk_file_lines
from modules.utils.progress_tracking import ProgressTracker
from modules.utils.streaming import get_training_hub, StreamEvent
from modules.utils.safe_writer import atomic_write_json
from modules.utils.run_context import RunContext, get_run_context
from modules.utils.path_resolver import PathResolver, get_path_resolver
from modules.utils.pipeline_state import PipelineState
from modules.utils.dynamic_resource_manager import DynamicResourceManager
from modules.utils.device_monitor import DeviceMonitor
from modules.utils.system_detector import get_system_profile
from modules.utils.unified_log import get_unified_log, init_unified_log

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core per-file processor
# ---------------------------------------------------------------------------

def clean_file(
    src_path:   Path,
    dst_path:   Path,
    chunk_size: int,
    *,
    min_len:    int  = 20,
    max_len:    int  = 2_000,
    dedup:      bool = True,
    encoding:   str  = "utf-8",
) -> dict:
    """
    Clean one raw Wikipedia dump text file and write the result atomically.

    chunk_size is passed in from the DynamicResourceManager so it reflects
    the current available RAM, not a value fixed at startup.
    """
    logger.debug(
        "clean_file START  src=%s  chunk_size=%d chars (%.1fMB)  dst=%s",
        src_path.name, chunk_size, chunk_size / 1e6, dst_path.name,
    )

    raw_count     = 0
    cleaned_lines: list[str] = []
    chunk_count   = 0

    for chunk in chunk_file_lines(src_path, chunk_size, encoding=encoding):
        chunk_count += 1
        raw_count   += len(chunk)
        good = list(clean_lines(chunk, min_len=min_len, max_len=max_len, dedup=dedup))
        cleaned_lines.extend(good)
        logger.debug(
            "  chunk %d: %d raw lines → %d kept",
            chunk_count, len(chunk), len(good),
        )

    content = "\n".join(cleaned_lines) + "\n" if cleaned_lines else ""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(dst_path, content, encoding=encoding)

    stats = cleaning_stats(raw_count, len(cleaned_lines))
    logger.debug(
        "clean_file END  %d/%d lines kept (%.1f%% removed)  chunks=%d",
        stats["cleaned_lines"], stats["raw_lines"],
        stats["removed_pct"], chunk_count,
    )
    return stats


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_dataset_cleaning(
    raw_dir:          Optional[Path]                    = None,
    cleaned_dir:      Optional[Path]                    = None,
    ctx:              Optional[RunContext]               = None,
    resolver:         Optional[PathResolver]            = None,
    resource_manager: Optional[DynamicResourceManager] = None,
    state:            Optional[PipelineState]           = None,
    *,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Process all Wikipedia dump .txt files → cleaned output directory.

    File-level resume
    -----------------
    Files whose cleaned output already exists are silently skipped.
    A ProgressTracker offset file records the last processed file index
    for sub-file granularity on deep resume.

    Adaptive chunk size
    -------------------
    chunk_size is read from the DynamicResourceManager handle on every
    file so it adjusts to live RAM without a restart.
    """
    with LogStage("dataset_clean"):
        cfg = get_config()
        ctx = ctx or get_run_context()

        try:
            res = resolver or get_path_resolver()
        except RuntimeError:
            from modules.utils.path_resolver import init_path_resolver
            res = init_path_resolver(cfg.project.model_id, cfg)

        actual_raw     = raw_dir     or res.raw_dir()
        actual_cleaned = cleaned_dir or res.cleaned_dir()
        log_dir        = res.logs_dir()

        ensure_dir(actual_cleaned)
        ensure_dir(log_dir)

        # Unified log
        ulog = get_unified_log()
        if ulog is None:
            ulog = init_unified_log(
                log_dir / ctx.prefix("pipeline.jsonl"),
                run_id=ctx.run_id,
                model_id=cfg.project.model_id,
            )

        # Pipeline state
        if state is None:
            state = PipelineState.load_latest(log_dir, model_id=cfg.project.model_id)
        rec = state.get("dataset_clean")

        # Progress offset (file-level resume)
        progress_path = log_dir / ctx.prefix("cleaning_progress.json")

        hub = get_training_hub()

        logger.info("Run ID          : %s", ctx.run_id)
        logger.info("Raw source      : %s", actual_raw)
        logger.info("Cleaned output  : %s", actual_cleaned)
        logger.info(
            "Config          : min_len=%d  max_len=%d  dedup=%s  encoding=%s",
            cfg.dataset.min_line_length, cfg.dataset.max_line_length,
            cfg.dataset.dedup, cfg.dataset.encoding,
        )

        hub.publish(StreamEvent.status(
            f"Dataset cleaning started — run_id={ctx.run_id}"
        ))
        ulog.dataset(f"Cleaning started  run_id={ctx.run_id}", raw_dir=str(actual_raw))

        # File discovery (wk_0.txt, wk_1.txt, … in natural sorted order)
        raw_files = sorted(list_files(actual_raw, glob="*.txt"))
        if not raw_files:
            raise DataError(
                f"No .txt files in {actual_raw}\n"
                f"Set DGB_RAW_DIR or copy Wikipedia dumps there."
            )

        logger.info(
            "Found %d raw file(s): %s … %s",
            len(raw_files),
            raw_files[0].name,
            raw_files[-1].name if len(raw_files) > 1 else raw_files[0].name,
        )

        # System profile → seed resource manager
        profile = get_system_profile()
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

        # Stage record
        rec.mark_running()
        state.save(log_dir)

        # Trackers
        tracker = ProgressTracker(
            total=len(raw_files),
            label="dataset_clean",
            log_every=1,
        )
        results: dict = {}
        totals = {"raw_lines": 0, "cleaned_lines": 0}
        skipped_count = 0
        t0 = time.time()

        try:
            for file_num, src in enumerate(raw_files, start=1):
                # ── Adaptive chunk size ────────────────────────────────
                chunk_size = resource_manager.handle.chunk_size_chars
                pressure   = resource_manager.handle.pressure

                dst_name = ctx.cleaned_name(src.stem)
                dst      = actual_cleaned / dst_name

                # ── File-level resume: skip if already cleaned ─────────
                if dst.exists() and dst.stat().st_size > 0:
                    logger.info(
                        "SKIP %d/%d: %s (cleaned output exists)",
                        file_num, len(raw_files), src.name,
                    )
                    ulog.dataset(
                        f"skip_existing  {src.name}",
                        file_num=file_num, dst=dst_name,
                    )
                    skipped_count += 1
                    tracker.update()
                    continue

                logger.info(
                    "File %d/%d: %s → %s  chunk=%d  pressure=%s",
                    file_num, len(raw_files), src.name, dst_name,
                    chunk_size, pressure,
                )
                hub.publish(StreamEvent.status(
                    f"Cleaning {src.name} ({file_num}/{len(raw_files)})"
                ))
                ulog.dataset(
                    f"clean_start  {src.name}",
                    file_num=file_num, chunk_size=chunk_size, pressure=pressure,
                )

                try:
                    stats = clean_file(
                        src, dst,
                        chunk_size=chunk_size,
                        min_len=cfg.dataset.min_line_length,
                        max_len=cfg.dataset.max_line_length,
                        dedup=cfg.dataset.dedup,
                        encoding=cfg.dataset.encoding,
                    )
                    logger.info(
                        "  ✓ %s  %d/%d lines kept (%.1f%% removed)",
                        dst_name,
                        stats["cleaned_lines"], stats["raw_lines"],
                        stats["removed_pct"],
                    )
                    ulog.dataset(
                        f"clean_done  {src.name}",
                        file_num=file_num,
                        raw_lines=stats["raw_lines"],
                        cleaned_lines=stats["cleaned_lines"],
                        removed_pct=stats["removed_pct"],
                    )
                except Exception as exc:
                    log_exception(exc, context=f"clean_file {src.name}")
                    ulog.error(f"clean_file failed: {src.name}: {exc}", stage="dataset_clean")
                    stats = {"error": str(exc)}

                results[src.name] = stats
                totals["raw_lines"]     += stats.get("raw_lines", 0)
                totals["cleaned_lines"] += stats.get("cleaned_lines", 0)

                # ── Atomic per-file progress checkpoint ───────────────
                tracker.save(
                    progress_path,
                    epoch=0,
                    batch_idx=0,
                    global_step=file_num,
                    line_offset=file_num,
                    note=f"after_{src.name}",
                )
                state.set("cleaning_files_done", file_num)
                state.save_checkpoint(log_dir, note=f"cleaning_file_{file_num}")

                prog_state = tracker.update()
                logger.info(
                    "Progress: %d/%d (%.1f%%)  ETA %.0fs",
                    prog_state.current, prog_state.total,
                    prog_state.percent, prog_state.eta_sec,
                )
                hub.publish(StreamEvent.progress(
                    phase="dataset",
                    current=prog_state.current, total=prog_state.total,
                    percent=prog_state.percent, eta_sec=prog_state.eta_sec,
                    file=src.name,
                ))
                if on_progress:
                    on_progress(prog_state.current, prog_state.total)

            # ── Summary ───────────────────────────────────────────────
            elapsed = time.time() - t0
            removed_pct = round(
                (1 - totals["cleaned_lines"] / max(totals["raw_lines"], 1)) * 100, 2
            )
            summary = {
                "run_id":          ctx.run_id,
                "files_processed": len(raw_files),
                "files_skipped":   skipped_count,
                "elapsed_sec":     round(elapsed, 2),
                **totals,
                "removed_pct":     removed_pct,
                "raw_dir":         str(actual_raw),
                "cleaned_dir":     str(actual_cleaned),
                "per_file":        results,
            }

            logger.info(
                "DONE  %d files  %d skipped  %d/%d lines kept (%.1f%% removed)  %.1fs",
                len(raw_files), skipped_count,
                totals["cleaned_lines"], totals["raw_lines"],
                removed_pct, elapsed,
            )
            hub.publish(StreamEvent.done("Dataset cleaning complete"))
            ulog.dataset("complete", **{k: v for k, v in summary.items() if k != "per_file"})

            summary_path = actual_cleaned / ctx.prefix("cleaning_summary.json")
            atomic_write_json(summary_path, summary)
            logger.info("Summary → %s", summary_path.name)

            rec.mark_completed(
                files_processed=len(raw_files),
                files_skipped=skipped_count,
                elapsed_sec=round(elapsed, 2),
            )
            state.save(log_dir)
            tracker.done()

            return summary

        except Exception as exc:
            rec.mark_failed(str(exc))
            state.save(log_dir)
            ulog.error(f"Dataset cleaning failed: {exc}", stage="dataset_clean")
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
    _bootstrap(level=cfg.logging.level, log_file=log_file)
    run_dataset_cleaning(ctx=ctx, resolver=res)
