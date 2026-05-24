"""
data_pipeline/processors/language_filter.py
=============================================
Stage 4: Language identification and filtering.

Uses fasttext's lid.176.bin model (loaded lazily) with fallbacks:
  1. fasttext lid.176 for fast, high-accuracy detection
  2. Character n-gram heuristic when model is unavailable

Features
--------
- Per-source language overrides (e.g. arXiv may allow any language)
- Configurable target languages (default: ["en"])
- Confidence threshold (default: 0.80)
- Script-based pre-filter (Latin, CJK, Cyrillic, etc.)
- Passes documents through when detection is uncertain (configurable)
- Language code normalisation (ISO 639-1)
"""
from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from data_pipeline.core.document import Document, ProcessingStage

logger = logging.getLogger(__name__)

# Default target languages
_DEFAULT_TARGETS: Set[str] = {"en"}
_DEFAULT_CONFIDENCE: float = 0.80

# Lang code normalisations (fasttext → ISO 639-1)
_NORMALISE = {
    "zh": "zh", "zh-hans": "zh", "zh-hant": "zh",
    "pt-br": "pt", "pt-pt": "pt",
    "sr-cyrl": "sr", "sr-latn": "sr",
    "bs": "bs", "hr": "hr",
}


class LanguageDetector:
    """
    Language detector with lazy-loaded fasttext model.

    Parameters
    ----------
    target_languages:     Set of ISO 639-1 codes to accept.
    min_confidence:       Minimum fasttext confidence to trust detection.
    pass_on_uncertainty:  If True, pass through documents when confidence < threshold.
    model_path:           Path to lid.176.bin (downloaded separately).
    source_overrides:     Dict mapping source_name → set of allowed languages.
                          Use {"*"} to allow all languages for a source.
    """

    def __init__(
        self,
        target_languages:    Set[str] = None,
        min_confidence:      float    = _DEFAULT_CONFIDENCE,
        pass_on_uncertainty: bool     = False,
        model_path:          Optional[Path] = None,
        source_overrides:    Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        self._targets   = set(target_languages or _DEFAULT_TARGETS)
        self._min_conf  = min_confidence
        self._pass_unc  = pass_on_uncertainty
        self._model_path = Path(model_path) if model_path else None
        self._overrides  = source_overrides or {}
        self._model      = None
        self._model_tried = False

    @classmethod
    def from_cfg(cls, cfg) -> "LanguageDetector":
        lc_raw = getattr(cfg, "language_filter", None) or {}
        if hasattr(lc_raw, "model_dump"):
            lc = lc_raw.model_dump()
        elif isinstance(lc_raw, dict):
            lc = lc_raw
        elif hasattr(lc_raw, "__dict__"):
            lc = dict(lc_raw.__dict__)
        else:
            lc = {}
        overrides = {}
        for src, langs in lc.get("source_overrides", {}).items():
            overrides[src] = set(langs) if langs != ["*"] else {"*"}
        model_path = lc.get("fasttext_model_path")
        return cls(
            target_languages=set(lc.get("target_languages", ["en"])),
            min_confidence=lc.get("min_confidence", 0.80),
            pass_on_uncertainty=lc.get("pass_on_uncertainty", False),
            model_path=Path(model_path) if model_path else None,
            source_overrides=overrides,
        )

    def process(self, doc: Document) -> Optional[Document]:
        """
        Detect language and filter.
        Returns None (rejected) if language not in targets.
        """
        # Check source override
        allowed = self._source_allowed_langs(doc.source_name)
        if "*" in allowed:
            doc.stage = ProcessingStage.FILTERED
            return doc

        lang, conf = self._detect(doc.text)
        lang = _NORMALISE.get(lang, lang)

        doc.language        = lang
        doc.lang_confidence = conf

        if conf < self._min_conf:
            if self._pass_unc:
                doc.stage = ProcessingStage.FILTERED
                return doc
            return doc.mark_rejected(
                f"lang_confidence_too_low:{lang}:{conf:.2f}<{self._min_conf}"
            )

        target = allowed or self._targets
        if lang not in target:
            return doc.mark_rejected(f"language_not_target:{lang}")

        doc.stage = ProcessingStage.FILTERED
        return doc

    def process_batch(self, docs: list) -> tuple:
        accepted, rejected = [], []
        for doc in docs:
            result = self.process(doc)
            (rejected if result is None or result.rejected else accepted).append(doc)
        return accepted, rejected

    # ── Detection ─────────────────────────────────────────────────────

    def _detect(self, text: str) -> Tuple[str, float]:
        """Return (lang_code, confidence)."""
        model = self._load_model()
        sample = text[:512].replace("\n", " ")

        if model is not None:
            try:
                preds = model.predict(sample, k=1)
                raw_lang = preds[0][0].replace("__label__", "")
                conf     = float(preds[1][0])
                return raw_lang, conf
            except Exception as exc:
                logger.debug("fasttext predict error: %s", exc)

        # Fallback: character n-gram heuristic
        return self._heuristic_detect(text)

    def _load_model(self):
        if self._model_tried:
            return self._model
        self._model_tried = True
        if self._model_path and self._model_path.exists():
            try:
                import fasttext
                self._model = fasttext.load_model(str(self._model_path))
                logger.info("fasttext model loaded: %s", self._model_path.name)
            except Exception as exc:
                logger.warning("fasttext unavailable: %s — using heuristic", exc)
        else:
            logger.info(
                "fasttext model not found at %s — using heuristic lang detection",
                self._model_path,
            )
        return self._model

    def _heuristic_detect(self, text: str) -> Tuple[str, float]:
        """
        Character-script heuristic — sufficient for English/non-English split.
        Returns ("en", confidence) or ("xx", confidence).
        """
        sample   = text[:300]
        total    = max(len(sample), 1)
        latin    = sum(1 for c in sample if "LATIN" in unicodedata.name(c, ""))
        cjk      = sum(1 for c in sample if "\u4e00" <= c <= "\u9fff")
        cyrillic = sum(1 for c in sample if "\u0400" <= c <= "\u04ff")
        arabic   = sum(1 for c in sample if "\u0600" <= c <= "\u06ff")

        if latin / total > 0.60:
            return "en", 0.70    # conservative — can't distinguish Romance/Germanic
        if cjk / total > 0.20:
            return "zh", 0.80
        if cyrillic / total > 0.20:
            return "ru", 0.75
        if arabic / total > 0.20:
            return "ar", 0.75
        return "en", 0.50    # uncertain

    def _source_allowed_langs(self, source_name: str) -> Set[str]:
        return self._overrides.get(source_name, set())
