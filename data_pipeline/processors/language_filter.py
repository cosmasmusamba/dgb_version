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
- Configurable target languages (default: African + Heavy Ugandan Priority)
- Confidence threshold (default: 0.65 for low-resource languages)
- Script-based pre-filter (Latin, CJK, Cyrillic, etc.)
- Passes documents through when detection is uncertain (configurable)
- Language code normalisation (ISO 639-1 & ISO 639-3 regional overrides)
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

# Default target languages focusing heavily on Uganda and major African lingua francas.
# Supports mixed ISO 639-1 (2-letter) and ISO 639-3 (3-letter) for regional Ugandan languages.
_DEFAULT_TARGETS: Set[str] = {
    # ─── CORE UGANDAN STANDARD TARGETS (ISO 639-1) ───
    "en",  # English (Official language of Uganda)
    "lg",  # Luganda / Ganda (Widespread Central Ugandan language)
    "sw",  # Swahili / Kiswahili (Official language of Uganda)

    # ─── ADDED REGIONAL UGANDAN TARGETS (ISO 639-3) ───
    "xog",  # Soga (Lusoga)
    "nyn",  # Nyankole (Runyankore)
    "nyu",  # Runyankole alternate mapping
    "ttj",  # Tooro (Rutooro)
    "njo",  # Tooro / Rutooro alternate mapping
    "cgg",  # Chiga (Rukiga)
    "ach",  # Acoli (Acholi)
    "lgg",  # Lugbara
    "gwr",  # Gwere (Lugwere)
    "myx",  # Masaaba / Gishu (Bantu language from Eastern Uganda/Mount Elgon)
    "teo",  # Ateso / Teso (Eastern Nilotic language spoken in Eastern Uganda & Western Kenya)
    "mas",  # Maasai / Masai (Eastern Nilotic language across Kenya & Tanzania)

    # ─── MAJOR PAN-AFRICAN LINGUA FRANCAS ───
    "rw",  # Kinyarwanda (Bordering southwest Uganda)
    "om",  # Oromo (East Africa)
    "so",  # Somali (East Africa)
    "am",  # Amharic (East Africa)
    "ar",  # Arabic (North & East Africa)
    "fr",  # French (West & Central Africa)
    "pt",  # Portuguese (Southern / Central Africa)
    "ha",  # Hausa (West Africa)
    "yo",  # Yoruba (West Africa)
    "ig",  # Igbo (West Africa)
    "sn",  # Shona (Southern Africa)
    "zu",  # Zulu (Southern Africa)
    "xh",  # Xhosa (Southern Africa)
    "wo",  # Wolof (West Africa)
}

_DEFAULT_CONFIDENCE: float = 0.65  # Retained lower threshold for low-resource language accuracy

# Lang code normalisations (fasttext labels / variants → pipeline targets)
_NORMALISE = {
    "zh": "zh", "zh-hans": "zh", "zh-hant": "zh",
    "pt-br": "pt", "pt-pt": "pt",
    "sr-cyrl": "sr", "sr-latn": "sr",
    "bs": "bs", "hr": "hr",
    
    # Normalise common regional macro-language code splits back to targets
    "nyu": "nyn",
    "njo": "ttj",
    "tojo": "ttj"
}


class LanguageDetector:
    """
    Language detector with lazy-loaded fasttext model.

    Parameters
    ----------
    target_languages:     Set of ISO 639-1 or ISO 639-3 codes to accept.
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
        lc = getattr(cfg, "language_filter", None) or {}
        if hasattr(lc, "__dict__"):
            lc = lc.__dict__
        overrides = {}
        for src, langs in lc.get("source_overrides", {}).items():
            overrides[src] = set(langs) if langs != ["*"] else {"*"}
        model_path = lc.get("fasttext_model_path")
        
        target_langs = lc.get("target_languages")
        target_set = set(target_langs) if target_langs else _DEFAULT_TARGETS
        
        return cls(
            target_languages=target_set,
            min_confidence=lc.get("min_confidence", _DEFAULT_CONFIDENCE),
            pass_on_uncertainty=lc.get("pass_on_uncertainty", False),
            model_path=Path(model_path) if model_path else None,
            source_overrides=overrides,
        )

    def process(self, doc: Document) -> Optional[Document]:
        """
        Detect language and filter.
        Returns None (rejected) if language not in targets.
        """
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
        Character-script heuristic adapted for African & Ugandan pipelines.
        """
        sample   = text[:300]
        total    = max(len(sample), 1)
        latin    = sum(1 for c in sample if "LATIN" in unicodedata.name(c, ""))
        arabic   = sum(1 for c in sample if "\u0600" <= c <= "\u06ff")
        ethiopic = sum(1 for c in sample if "\u1200" <= c <= "\u137f")

        # Except for Lugbara (which occasionally defaults to distinct Latin notation variations), 
        # Lusoga, Runyankore, Rutooro, Rukiga, Acholi, and Lugwere all utilize standard Latin script.
        # We target a low confidence pass score to allow pipeline filters to intercept text.
        if latin / total > 0.60:
            return "en", 0.51    
        if arabic / total > 0.20:
            return "ar", 0.75
        if ethiopic / total > 0.20:
            return "am", 0.80
        return "en", 0.50
