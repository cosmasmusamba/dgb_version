"""
data_pipeline/workers/pipeline_orchestrator.py
================================================
Top-level pipeline orchestrator.

Discovers enabled sources from runtime_config.json, instantiates the
correct extractor, builds per-source workers, and runs them concurrently
as asyncio tasks up to max_concurrent_sources.

Concurrency model
-----------------
  - One asyncio task per source
  - max_concurrent_sources limits simultaneous active tasks (semaphore)
  - Graceful shutdown on SIGINT / SIGTERM — drains current batches
  - Failed sources are marked in checkpoint and skipped on resume
  - --force SOURCE resets a source checkpoint for re-run

Usage
-----
    orchestrator = PipelineOrchestrator.from_cfg(cfg, run_id=ctx.run_id)
    await orchestrator.run()
    print(orchestrator.summary())
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional

from data_pipeline.core.checkpoint import CheckpointManager
from data_pipeline.core.quota_manager import QuotaManager
from data_pipeline.workers.source_worker import SourceWorker

logger = logging.getLogger(__name__)

# Registry: source_name → extractor builder class
_EXTRACTOR_REGISTRY = {
    "wikipedia":    "data_pipeline.extractors.wikipedia_extractor.WikipediaExtractor",
    "stackexchange":"data_pipeline.extractors.stackexchange_extractor.StackExchangeExtractor",
    "arxiv":        "data_pipeline.extractors.arxiv_extractor.ArxivExtractor",
    "gutenberg":    "data_pipeline.extractors.gutenberg_extractor.GutenbergExtractor",
    "commoncrawl":  "data_pipeline.extractors.commoncrawl_extractor.CommonCrawlExtractor",
    "github":       "data_pipeline.extractors.github_extractor.GitHubExtractor",
}


def _load_extractor_class(dotted: str):
    """Dynamically import an extractor class by dotted path."""
    module_path, class_name = dotted.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


class PipelineOrchestrator:
    """
    Orchestrates all enabled pipeline sources.

    Parameters
    ----------
    cfg:                  DGBConfig runtime config.
    run_id:               DateTime-prefixed run identifier.
    output_base:          datasets/ directory.
    checkpoint_dir:       Checkpoint persistence directory.
    max_concurrent:       Max simultaneously active source tasks.
    force_sources:        Sources to reset (re-run) even if completed.
    only_sources:         If set, only run these sources.
    skip_sources:         Sources to skip regardless of config.
    """

    def __init__(
        self,
        cfg,
        run_id:           str,
        output_base:      Path,
        checkpoint_dir:   Path,
        max_concurrent:   int  = 3,
        force_sources:    List[str] = None,
        only_sources:     List[str] = None,
        skip_sources:     List[str] = None,
    ) -> None:
        self._cfg           = cfg
        self._run_id        = run_id
        self._base          = Path(output_base)
        self._max_concurrent = max_concurrent
        self._force         = set(force_sources or [])
        self._only          = set(only_sources  or [])
        self._skip          = set(skip_sources  or [])
        self._results: Dict[str, dict] = {}
        self._stop_event    = asyncio.Event()

        # Shared infrastructure
        self._ckpt = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            run_id=run_id,
        )
        self._quota = self._build_quota_mgr()

        logger.info(
            "PipelineOrchestrator: run_id=%s  max_concurrent=%d",
            run_id, max_concurrent,
        )

    @classmethod
    def from_cfg(
        cls,
        cfg,
        run_id:         str,
        force_sources:  List[str] = None,
        only_sources:   List[str] = None,
        skip_sources:   List[str] = None,
    ) -> "PipelineOrchestrator":
        from modules.utils.path_resolver import init_path_resolver
        res        = init_path_resolver(cfg.project.model_id, cfg)
        output_base = Path("datasets")
        ckpt_dir   = res.logs_dir() / "pipeline"

        pipeline_cfg = getattr(cfg, "pipeline", None) or {}
        if hasattr(pipeline_cfg, "__dict__"):
            pipeline_cfg = pipeline_cfg.__dict__
        max_conc = pipeline_cfg.get("max_concurrent_sources", 3)

        return cls(
            cfg=cfg,
            run_id=run_id,
            output_base=output_base,
            checkpoint_dir=ckpt_dir,
            max_concurrent=max_conc,
            force_sources=force_sources,
            only_sources=only_sources,
            skip_sources=skip_sources,
        )

    async def run(self) -> None:
        """Run all enabled sources with concurrency control."""
        self._install_signal_handlers()
        sources = self._discover_sources()

        if not sources:
            logger.warning("No enabled sources found in config")
            return

        logger.info(
            "Pipeline starting: %d sources  [%s]",
            len(sources), ", ".join(sources),
        )

        semaphore = asyncio.Semaphore(self._max_concurrent)
        t0 = time.time()

        async def _run_with_sem(name: str, source_cfg: dict):
            async with semaphore:
                if self._stop_event.is_set():
                    return
                worker = self._build_worker(name, source_cfg)
                if worker is None:
                    return
                result = await worker.run()
                self._results[name] = result

        tasks = [
            asyncio.create_task(_run_with_sem(name, scfg))
            for name, scfg in sources.items()
        ]

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            logger.info("Pipeline cancelled — saving checkpoints")
        finally:
            self._ckpt.save_all()

        elapsed = time.time() - t0
        logger.info("Pipeline complete in %.1fs", elapsed)
        logger.info(self.summary())

    def summary(self) -> str:
        lines = [f"\n{'='*60}", f"Pipeline summary  run_id={self._run_id}", "="*60]
        total_docs = total_shards = 0
        for name, r in self._results.items():
            lines.append(
                f"  {name:<20} [{r.get('status','?'):<10}] "
                f"accepted={r.get('total_accepted',0):>8,}  "
                f"shards={r.get('shards',0):>4}  "
                f"time={r.get('elapsed_sec',0):.0f}s"
            )
            total_docs   += r.get("total_accepted", 0)
            total_shards += r.get("shards", 0)
        lines.append("-"*60)
        lines.append(f"  {'TOTAL':<20}  accepted={total_docs:>8,}  shards={total_shards:>4}")
        lines.append("="*60)
        return "\n".join(lines)

    # ── Internals ─────────────────────────────────────────────────────

    def _discover_sources(self) -> Dict[str, dict]:
        """Return ordered dict of {source_name: source_cfg_dict} from config."""
        sources_cfg = getattr(self._cfg, "sources", None) or {}
        if hasattr(sources_cfg, "__dict__"):
            sources_cfg = sources_cfg.__dict__

        result = {}
        for name, scfg in sources_cfg.items():
            if hasattr(scfg, "__dict__"):
                scfg = scfg.__dict__
            if not scfg.get("enabled", True):
                continue
            if self._only and name not in self._only:
                continue
            if name in self._skip:
                continue

            # Reset checkpoint if --force
            if name in self._force:
                cp = self._ckpt.get(name)
                cp.completed = False
                cp.failed    = False
                self._ckpt.save(name)
                logger.info("Source %s checkpoint reset (--force)", name)

            if self._ckpt.is_completed(name) and name not in self._force:
                logger.info("Source %s already completed — skipping", name)
                continue

            result[name] = scfg
        return result

    def _build_worker(self, name: str, scfg: dict) -> Optional[SourceWorker]:
        extractor_type = scfg.get("extractor_type", name)
        cls_path = _EXTRACTOR_REGISTRY.get(extractor_type)
        if cls_path is None:
            logger.error("No extractor registered for source: %s", name)
            return None
        try:
            ExtCls   = _load_extractor_class(cls_path)
            extractor = ExtCls.build(
                {**scfg, "source_name": name},
                checkpoint_mgr=self._ckpt,
                quota_mgr=self._quota,
            )
            return SourceWorker(
                source_name=name,
                extractor=extractor,
                cfg=self._cfg,
                checkpoint_mgr=self._ckpt,
                quota_mgr=self._quota,
                output_base=self._base,
                run_id=self._run_id,
                stop_event=self._stop_event,
            )
        except Exception as exc:
            logger.error("Cannot build worker for %s: %s", name, exc)
            return None

    def _build_quota_mgr(self) -> QuotaManager:
        qcfg = getattr(self._cfg, "storage_quotas", None) or {}
        if hasattr(qcfg, "__dict__"):
            qcfg = qcfg.__dict__

        source_quotas = {}
        for k, v in qcfg.items():
            if k.endswith("_gb") and isinstance(v, (int, float)):
                src = k[:-3]
                source_quotas[src] = int(v * 1024**3)

        global_max   = int(qcfg.get("global_max_gb", 0) * 1024**3)
        safety       = int(qcfg.get("safety_margin_gb", 10) * 1024**3)

        return QuotaManager(
            datasets_root=self._base,
            global_max_bytes=global_max,
            safety_margin_bytes=safety,
            source_quotas=source_quotas,
        )

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda: (
                        logger.info("Signal received — graceful shutdown"),
                        self._stop_event.set(),
                        [ext.stop() for ext in []],
                    ),
                )
            except (NotImplementedError, AttributeError):
                pass   # Windows doesn't support add_signal_handler
