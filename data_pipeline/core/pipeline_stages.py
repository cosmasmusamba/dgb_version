"""
data_pipeline/core/pipeline_stages.py
========================================
Preprocessing chain that connects all stages in configurable order.

The PreprocessingPipeline assembles every stage (normalizer, lang filter,
toxicity filter, quality scorer, deduplicator) from runtime config and
exposes a single process() / process_batch() entry point.

Each stage can be individually enabled/disabled via runtime_config.json:

    "pipeline_stages": {
        "normalize":        true,
        "language_filter":  true,
        "toxicity_filter":  true,
        "quality_scorer":   true,
        "deduplicator":     true
    }

The chain processes documents in order, short-circuiting immediately on
rejection so downstream stages never receive already-rejected documents.
Batch-level statistics are logged after every BATCH_LOG_EVERY batches.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data_pipeline.core.document import Document, ProcessingStage

logger = logging.getLogger(__name__)

_BATCH_LOG_EVERY = 100


@dataclass
class StageStats:
    name:          str
    total_in:      int = 0
    total_out:     int = 0
    total_rejected: int = 0
    elapsed_sec:   float = 0.0

    @property
    def rejection_rate(self) -> float:
        return self.total_rejected / max(self.total_in, 1)

    @property
    def throughput(self) -> float:
        return self.total_out / max(self.elapsed_sec, 1e-6)


class PreprocessingPipeline:
    """
    Configurable multi-stage document preprocessing chain.

    Parameters
    ----------
    normalizer:       TextNormalizer instance (or None to skip).
    lang_filter:      LanguageDetector instance (or None to skip).
    toxicity_filter:  ToxicityFilter instance (or None to skip).
    quality_scorer:   QualityScorer instance (or None to skip).
    deduplicator:     Deduplicator instance (or None to skip).
    rejected_writer:  Optional ShardWriter for rejected documents.
    """

    def __init__(
        self,
        normalizer=None,
        lang_filter=None,
        toxicity_filter=None,
        quality_scorer=None,
        deduplicator=None,
        rejected_writer=None,
    ) -> None:
        self._stages: List[Tuple[str, object]] = []
        if normalizer:
            self._stages.append(("normalize", normalizer))
        if lang_filter:
            self._stages.append(("language_filter", lang_filter))
        if toxicity_filter:
            self._stages.append(("toxicity_filter", toxicity_filter))
        if quality_scorer:
            self._stages.append(("quality_scorer", quality_scorer))
        if deduplicator:
            self._stages.append(("deduplicator", deduplicator))

        self._rejected_writer = rejected_writer
        self._stats: Dict[str, StageStats] = {
            name: StageStats(name=name) for name, _ in self._stages
        }
        self._batches_processed = 0
        self._total_in  = 0
        self._total_out = 0

        logger.info(
            "PreprocessingPipeline: stages=[%s]",
            ", ".join(n for n, _ in self._stages),
        )

    @classmethod
    def from_cfg(cls, cfg, state_dir: Path, run_id: str = "",
                 rejected_writer=None) -> "PreprocessingPipeline":
        """
        Build the full pipeline from runtime config.
        Only stages that are enabled in pipeline_stages config are included.
        """
        from data_pipeline.processors.normalizer      import TextNormalizer
        from data_pipeline.processors.language_filter  import LanguageDetector
        from data_pipeline.processors.toxicity_filter  import ToxicityFilter
        from data_pipeline.processors.quality_scorer   import QualityScorer
        from data_pipeline.processors.deduplicator     import Deduplicator

        enabled = {}
        ps = getattr(cfg, "pipeline_stages", None) or {}
        if hasattr(ps, "__dict__"):
            ps = ps.__dict__
        enabled = {
            "normalize":       ps.get("normalize",       True),
            "language_filter": ps.get("language_filter", True),
            "toxicity_filter": ps.get("toxicity_filter", True),
            "quality_scorer":  ps.get("quality_scorer",  True),
            "deduplicator":    ps.get("deduplicator",    True),
        }

        return cls(
            normalizer      = TextNormalizer.from_cfg(cfg) if enabled["normalize"] else None,
            lang_filter     = LanguageDetector.from_cfg(cfg) if enabled["language_filter"] else None,
            toxicity_filter = ToxicityFilter.from_cfg(cfg) if enabled["toxicity_filter"] else None,
            quality_scorer  = QualityScorer.from_cfg(cfg) if enabled["quality_scorer"] else None,
            deduplicator    = Deduplicator.from_cfg(cfg, state_dir=state_dir, run_id=run_id)
                              if enabled["deduplicator"] else None,
            rejected_writer = rejected_writer,
        )

    def process(self, doc: Document) -> Optional[Document]:
        """
        Run document through the full pipeline.
        Returns None if the document was rejected at any stage.
        """
        for name, stage in self._stages:
            st = self._stats[name]
            t0 = time.perf_counter()
            st.total_in += 1

            result = stage.process(doc)

            st.elapsed_sec += time.perf_counter() - t0

            if result is None or result.rejected:
                st.total_rejected += 1
                if self._rejected_writer and result:
                    self._rejected_writer.write(result)
                return None

            st.total_out += 1

        self._total_in  += 1
        self._total_out += 1
        doc.stage = ProcessingStage.FINAL
        return doc

    def process_batch(
        self, docs: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """
        Process a batch.
        Returns (accepted, rejected).
        Logs batch stats every BATCH_LOG_EVERY batches.
        """
        accepted: List[Document] = []
        rejected: List[Document] = []

        for doc in docs:
            result = self.process(doc)
            if result is not None:
                accepted.append(result)
            else:
                rejected.append(doc)

        self._batches_processed += 1
        if self._batches_processed % _BATCH_LOG_EVERY == 0:
            self._log_stats(len(docs), len(accepted))

        return accepted, rejected

    # ── Stats ─────────────────────────────────────────────────────────

    def stage_stats(self) -> Dict[str, dict]:
        return {
            name: {
                "in":          s.total_in,
                "out":         s.total_out,
                "rejected":    s.total_rejected,
                "reject_rate": round(s.rejection_rate * 100, 1),
                "throughput":  round(s.throughput, 1),
            }
            for name, s in self._stats.items()
        }

    def overall_stats(self) -> dict:
        return {
            "total_in":       self._total_in,
            "total_accepted": self._total_out,
            "batches":        self._batches_processed,
            "stages":         self.stage_stats(),
        }

    def _log_stats(self, batch_size: int, accepted: int) -> None:
        rates = "  ".join(
            f"{n}:{s.total_rejected}/{s.total_in}"
            for n, s in self._stats.items()
        )
        logger.info(
            "Pipeline batch=%d  accepted=%d/%d  stage_rejects: [%s]",
            self._batches_processed, accepted, batch_size, rates,
        )