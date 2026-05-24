"""
data_pipeline/workers/source_worker.py
==========================================
Async worker that runs one complete source pipeline:
    extract → batch → preprocess → dedup → write shards

Each source runs in its own asyncio task so sources can run concurrently.
Workers respect the global stop event for graceful shutdown.

Lifecycle
---------
1. Instantiate extractor from registry
2. Build preprocessing pipeline from config
3. Open ShardWriter for this source
4. Async-for over extractor.extract_batches()
5. For each batch: preprocess → write accepted → write rejected (optional)
6. Checkpoint saved after every batch
7. On completion / error: mark checkpoint, close writers, log summary

Memory profile
--------------
  - Only one batch in memory at a time
  - Shard writer buffers at OS level (64 KB)
  - Dedup index grows linearly but is bounded by config
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from data_pipeline.core.checkpoint import CheckpointManager
from data_pipeline.core.pipeline_stages import PreprocessingPipeline
from data_pipeline.core.quota_manager import QuotaManager
from data_pipeline.core.shard_writer import ShardWriter

logger = logging.getLogger(__name__)


class SourceWorker:
    """
    Async worker for one pipeline source.

    Parameters
    ----------
    source_name:    e.g. "wikipedia", "arxiv"
    extractor:      Configured BaseExtractor instance
    cfg:            DGBConfig (full runtime config)
    checkpoint_mgr: Shared CheckpointManager
    quota_mgr:      Shared QuotaManager
    output_base:    Base directory (datasets/)
    run_id:         Run datetime prefix
    stop_event:     asyncio.Event — set to request graceful shutdown
    """

    def __init__(
        self,
        source_name:    str,
        extractor,
        cfg,
        checkpoint_mgr: CheckpointManager,
        quota_mgr:      QuotaManager,
        output_base:    Path,
        run_id:         str,
        stop_event:     Optional[asyncio.Event] = None,
    ) -> None:
        self._name    = source_name
        self._ext     = extractor
        self._cfg     = cfg
        self._ckpt    = checkpoint_mgr
        self._quota   = quota_mgr
        self._base    = Path(output_base)
        self._run_id  = run_id
        self._stop    = stop_event or asyncio.Event()
        self._log     = logging.getLogger(f"dgb.worker.{source_name}")

        # Per-source directories
        self._out_dir       = self._base / source_name
        self._rejected_dir  = self._base / source_name / "rejected"
        self._dedup_dir     = self._base / source_name / "dedup_state"

        for d in [self._out_dir, self._rejected_dir, self._dedup_dir]:
            d.mkdir(parents=True, exist_ok=True)

    async def run(self) -> dict:
        """
        Execute the full source pipeline.
        Returns a stats dict on completion.
        """
        cp   = self._ckpt.get(self._name)
        t0   = time.time()

        if cp.completed:
            self._log.info("%s: already completed — skipping", self._name)
            return {"source": self._name, "status": "skipped"}

        self._log.info("Worker [%s]: starting", self._name)

        # Load pipeline config slice
        pipeline_cfg = _SourcePipelineCfg(self._cfg, self._name)

        # Pipeline
        rejected_writer = ShardWriter(
            output_dir=self._rejected_dir,
            source_name=f"{self._name}_rejected",
            run_id=self._run_id,
            max_shard_bytes=self._get_max_shard_bytes() // 4,
            quota_manager=self._quota,
            checkpoint_mgr=self._ckpt,
        )
        pipeline = PreprocessingPipeline.from_cfg(
            cfg=pipeline_cfg,
            state_dir=self._dedup_dir,
            run_id=self._run_id,
            rejected_writer=rejected_writer,
        )

        writer = ShardWriter(
            output_dir=self._out_dir,
            source_name=self._name,
            run_id=self._run_id,
            max_shard_bytes=self._get_max_shard_bytes(),
            max_shard_records=self._get_max_shard_records(),
            quota_manager=self._quota,
            checkpoint_mgr=self._ckpt,
        )

        total_raw = total_accepted = total_rejected = batch_n = 0

        try:
            async for batch in self._ext.extract_batches():
                if self._stop.is_set():
                    self._log.info("[%s]: stop requested — flushing", self._name)
                    break

                batch_n += 1
                total_raw += len(batch)

                accepted, rejected = pipeline.process_batch(batch)
                total_accepted += len(accepted)
                total_rejected += len(rejected)

                for doc in accepted:
                    writer.write(doc)

                # Update checkpoint stats
                cp.total_raw      = total_raw
                cp.total_accepted = total_accepted
                cp.total_rejected = total_rejected
                cp.last_batch_idx = batch_n

                # Periodic save and log
                if batch_n % 10 == 0:
                    self._ckpt.save(self._name)
                    writer.flush()
                    self._log.info(
                        "[%s] batch=%d  raw=%d  accepted=%d  rejected=%d  "
                        "shards=%d  quota=%s",
                        self._name, batch_n, total_raw, total_accepted, total_rejected,
                        len(writer.completed_shards),
                        f"{self._quota.source_stats(self._name)['used_gb']:.2f}GB"
                        if self._quota else "n/a",
                    )

            self._ckpt.mark_completed(self._name)
            status = "completed"

        except Exception as exc:
            self._log.error("[%s] failed: %s", self._name, exc, exc_info=True)
            self._ckpt.mark_failed(self._name, str(exc))
            status = "failed"

        finally:
            writer.close()
            rejected_writer.close()
            # Save dedup state
            if hasattr(pipeline, "_stages"):
                for stage_name, stage in pipeline._stages:
                    if stage_name == "deduplicator":
                        stage.save()

        elapsed = time.time() - t0
        stats = {
            "source":          self._name,
            "status":          status,
            "total_raw":       total_raw,
            "total_accepted":  total_accepted,
            "total_rejected":  total_rejected,
            "batches":         batch_n,
            "shards":          len(writer.completed_shards),
            "elapsed_sec":     round(elapsed, 1),
            "throughput_dps":  round(total_raw / max(elapsed, 1), 1),
            "pipeline_stats":  pipeline.overall_stats(),
        }
        self._log.info(
            "[%s] done: status=%s  accepted=%d  shards=%d  %.1fs",
            self._name, status, total_accepted, len(writer.completed_shards), elapsed,
        )
        return stats

    def _get_max_shard_bytes(self) -> int:
        sc = getattr(self._cfg, "pipeline", None)
        if sc is None: return 512 * 1024 * 1024
        return sc.max_shard_bytes if hasattr(sc, "max_shard_bytes") else sc.get("max_shard_bytes", 512*1024*1024)

    def _get_max_shard_records(self) -> int:
        sc = getattr(self._cfg, "pipeline", None)
        if sc is None: return 1_000_000
        return sc.max_shard_records if hasattr(sc, "max_shard_records") else sc.get("max_shard_records", 1_000_000)


class _SourcePipelineCfg:
    """Thin wrapper to forward cfg attributes to pipeline stage factories."""
    def __init__(self, cfg, source_name: str) -> None:
        self._cfg  = cfg
        self._src  = source_name
        # Copy top-level attrs
        for attr in ["normalizer", "language_filter", "toxicity_filter",
                     "quality_scorer", "deduplicator", "pipeline_stages"]:
            val = getattr(cfg, attr, None)
            if val is None:
                # Try nested in sources
                sources = getattr(cfg, "sources", {}) or {}
                if hasattr(sources, "__dict__"):
                    sources = sources.__dict__
                src_cfg = sources.get(source_name, {}) or {}
                if hasattr(src_cfg, "__dict__"):
                    src_cfg = src_cfg.__dict__
                val = src_cfg.get(attr)
            setattr(self, attr, val)
