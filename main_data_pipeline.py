"""
main_data_pipeline.py
=======================
CLI entry point for the DGB large-scale data ingestion pipeline.

Usage
-----
  # Run all enabled sources
  python main_data_pipeline.py

  # Run specific sources only
  python main_data_pipeline.py --sources wikipedia arxiv

  # Skip sources
  python main_data_pipeline.py --skip commoncrawl

  # Force re-run a completed source
  python main_data_pipeline.py --force wikipedia

  # Print pipeline status and exit
  python main_data_pipeline.py --status

  # Export processed shards to training format
  python main_data_pipeline.py --export --format plain_text
  python main_data_pipeline.py --export --format jsonl_text_only
  python main_data_pipeline.py --export --format jsonl_sft

  # Dry run (validate config, list sources, exit)
  python main_data_pipeline.py --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from configs.loader import get_config
from modules.logging_config import configure_logging, LogStage
from modules.utils.run_context import RunContext


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DGB data ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sources",  nargs="+", metavar="SOURCE",
                   help="Only run these sources (space-separated)")
    p.add_argument("--skip",     nargs="+", metavar="SOURCE",
                   help="Skip these sources")
    p.add_argument("--force",    nargs="+", metavar="SOURCE",
                   help="Reset checkpoint and re-run these sources")
    p.add_argument("--status",   action="store_true",
                   help="Print pipeline status and exit")
    p.add_argument("--dry-run",  action="store_true",
                   help="Validate config, list sources, and exit")
    p.add_argument("--export",   action="store_true",
                   help="Export processed shards to training format")
    p.add_argument("--format",   default="jsonl_text_only",
                   choices=["plain_text","jsonl_text_only","jsonl_full","jsonl_sft"],
                   help="Export format (default: jsonl_text_only)")
    p.add_argument("--export-dir", default=None,
                   help="Output directory for exported shards (default: datasets/export/)")
    p.add_argument("--min-quality", type=float, default=0.0,
                   help="Minimum overall_quality score to include in export")
    p.add_argument("--languages", nargs="+", default=[],
                   help="Filter export to these language codes")
    p.add_argument("--run-id",   default=None,
                   help="Resume a specific run ID")
    p.add_argument("--concurrent", type=int, default=None,
                   help="Override max_concurrent_sources from config")
    p.add_argument("--log-level", default=None,
                   choices=["DEBUG","INFO","WARNING","ERROR"],
                   help="Override log level from config")
    return p.parse_args()


def _print_status(cfg, run_id: str) -> None:
    """Print checkpoint status for all sources and exit."""
    from modules.utils.path_resolver import init_path_resolver
    from data_pipeline.core.checkpoint import CheckpointManager

    res     = init_path_resolver(cfg.project.model_id, cfg)
    ckpt_dir = res.logs_dir() / "pipeline"
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir, run_id=run_id)

    sources_cfg = getattr(cfg, "sources", {}) or {}
    if hasattr(sources_cfg, "__dict__"):
        sources_cfg = sources_cfg.__dict__

    print(f"\nPipeline status  run_id={run_id}")
    print("=" * 70)
    print(f"  {'Source':<20} {'Status':<12} {'Raw':>10} {'Accepted':>10} {'Shards':>6}")
    print("-" * 70)

    for name in sorted(sources_cfg.keys()):
        cp = ckpt_mgr.get(name)
        status = "done" if cp.completed else ("failed" if cp.failed else
                 "running" if cp.total_raw > 0 else "pending")
        print(
            f"  {name:<20} {status:<12} {cp.total_raw:>10,} "
            f"{cp.total_accepted:>10,} {len(cp.completed_shards):>6}"
        )
    print("=" * 70)


def _run_export(cfg, args: argparse.Namespace, run_id: str) -> None:
    """Export processed shards to training-ready format."""
    from data_pipeline.storage.export import ShardExporter, ExportFormat

    fmt_map = {
        "plain_text":       ExportFormat.PLAIN_TEXT,
        "jsonl_text_only":  ExportFormat.JSONL_TEXT_ONLY,
        "jsonl_full":       ExportFormat.JSONL_FULL,
        "jsonl_sft":        ExportFormat.JSONL_SFT,
    }
    fmt = fmt_map[args.format]

    input_dirs = []
    sources_cfg = getattr(cfg, "sources", {}) or {}
    if hasattr(sources_cfg, "__dict__"):
        sources_cfg = sources_cfg.__dict__
    for name in sources_cfg:
        d = Path("datasets") / name
        if d.exists():
            input_dirs.append(d)
    # Also include legacy DGB cleaned dir
    legacy = Path("datasets") / cfg.project.model_id / "cleaned"
    if legacy.exists():
        input_dirs.append(legacy)

    if not input_dirs:
        print("No processed data found to export. Run the pipeline first.")
        return

    export_dir = Path(args.export_dir) if args.export_dir else Path("datasets") / "export" / args.format
    print(f"\nExporting to: {export_dir}")
    print(f"Format:       {args.format}")
    print(f"Sources:      {len(input_dirs)} directories")

    exporter = ShardExporter(
        input_dirs=input_dirs,
        output_dir=export_dir,
        format=fmt,
        run_id=run_id,
        shuffle=True,
        min_quality=args.min_quality,
        languages=args.languages or None,
    )
    manifest = exporter.export()
    print(f"\nExport complete:")
    print(f"  Documents:      {manifest.total_docs:,}")
    print(f"  Output shards:  {len(manifest.output_shards)}")
    print(f"  Tokens (est):   {manifest.total_tokens_est/1e9:.2f}B")
    print(f"  By source:")
    for src, n in sorted(manifest.source_breakdown.items(), key=lambda x: -x[1]):
        print(f"    {src:<20} {n:>10,}")


async def _run_pipeline(cfg, args: argparse.Namespace, run_id: str) -> None:
    from data_pipeline.workers.pipeline_orchestrator import PipelineOrchestrator

    pipeline_cfg = getattr(cfg, "pipeline", None) or {}
    if hasattr(pipeline_cfg, "__dict__"):
        pipeline_cfg = pipeline_cfg.__dict__
    max_conc = args.concurrent or pipeline_cfg.get("max_concurrent_sources", 2)

    orchestrator = PipelineOrchestrator.from_cfg(
        cfg=cfg,
        run_id=run_id,
        force_sources=args.force or [],
        only_sources=args.sources or [],
        skip_sources=args.skip or [],
    )
    # Override concurrency from CLI
    orchestrator._max_concurrent = max_conc

    await orchestrator.run()
    print(orchestrator.summary())


def main() -> int:
    args = _build_args()
    cfg  = get_config()

    log_level = args.log_level or cfg.logging.level
    configure_logging(level=log_level)

    ctx = RunContext(model_id=cfg.project.model_id, run_id=args.run_id)

    logger = logging.getLogger(__name__)
    logger.info("DGB Data Pipeline v%s  run_id=%s", cfg.project.version, ctx.run_id)

    # ── Status ─────────────────────────────────────────────────────────
    if args.status:
        _print_status(cfg, ctx.run_id)
        return 0

    # ── Dry run ────────────────────────────────────────────────────────
    if getattr(args, "dry_run", False):
        print(f"\nDry run — DGB Pipeline v{cfg.project.version}  run_id={ctx.run_id}")
        sources_cfg = getattr(cfg, "sources", {}) or {}
        if hasattr(sources_cfg, "__dict__"):
            sources_cfg = sources_cfg.__dict__
        enabled = [n for n, c in sources_cfg.items()
                   if (c if isinstance(c, dict) else c.__dict__).get("enabled", True)]
        print(f"\nEnabled sources ({len(enabled)}):")
        for s in enabled:
            print(f"  - {s}")
        pipeline_cfg = getattr(cfg, "pipeline", None) or {}
        if hasattr(pipeline_cfg, "__dict__"):
            pipeline_cfg = pipeline_cfg.__dict__
        print(f"\nPipeline settings:")
        print(f"  max_concurrent_sources: {pipeline_cfg.get('max_concurrent_sources',2)}")
        print(f"  max_shard_bytes:        {pipeline_cfg.get('max_shard_bytes',536870912)/1024**2:.0f} MB")
        print(f"  batch_size:             {pipeline_cfg.get('batch_size',1000)}")
        stages = getattr(cfg, "pipeline_stages", None) or {}
        if hasattr(stages, "__dict__"):
            stages = stages.__dict__
        print(f"\nPipeline stages:")
        for st, en in stages.items():
            print(f"  {st:<22} {'enabled' if en else 'disabled'}")
        return 0

    # ── Export ─────────────────────────────────────────────────────────
    if args.export:
        with LogStage("export"):
            _run_export(cfg, args, ctx.run_id)
        return 0

    # ── Full pipeline run ──────────────────────────────────────────────
    with LogStage("data_pipeline"):
        t0 = time.time()
        try:
            asyncio.run(_run_pipeline(cfg, args, ctx.run_id))
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        elapsed = time.time() - t0
        logger.info("Total pipeline time: %.1fs (%.1f min)", elapsed, elapsed / 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())