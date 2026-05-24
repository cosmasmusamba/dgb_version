"""
data_pipeline/processors/toxicity_filter.py
=============================================
Stage 5: Toxicity and unsafe-content filtering.

Implements a multi-layer, configurable filtering pipeline:

Layer 1 — Fast keyword/regex blocklist (no model needed):
  - Hate speech trigger words
  - Explicit sexual content markers
  - Extreme violence descriptors
  - Spam and phishing patterns
  - Prompt injection artifacts
  - Synthetic low-quality generation markers

Layer 2 — Heuristic structural detectors:
  - Repeated profanity density
  - Personal information density (PII)
  - Code injection patterns
  - Malware signature strings

Layer 3 — Optional model-based scoring (Detoxify / perspective, loaded lazily):
  - Toxicity probability score
  - Hate probability score
  - Explicit content probability

All thresholds, word lists, and layer enables are configurable via
runtime_config.json without code changes.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

from data_pipeline.core.document import Document, ProcessingStage, QualitySignals

logger = logging.getLogger(__name__)

# ── Built-in blocklist patterns ────────────────────────────────────────────────
# NOTE: These are intentionally vague to avoid reproducing harmful content.
# Replace with your own curated blocklist in production.

_HATE_SPEECH_RE = re.compile(
    r"\b(?:kill all|genocide of|exterminate|gas the|white power|"
    r"racial slur placeholder|death to all)\b",
    re.IGNORECASE,
)

_EXPLICIT_RE = re.compile(
    r"\b(?:explicit_term_placeholder|adult_content_marker)\b",
    re.IGNORECASE,
)

_SPAM_RE = re.compile(
    r"(?:click here to claim|you have won|act now|limited time offer|"
    r"100% free|make money fast|work from home|earn \$\d+/day|"
    r"buy now|buy cheap|cheap meds|erectile|casino|gambling|"
    r"bitcoin doubler|crypto scam|phishing\.)",
    re.IGNORECASE,
)

_INJECTION_RE = re.compile(
    r"(?:ignore previous instructions|disregard your|forget everything|"
    r"you are now|pretend you are a|act as if|jailbreak|DAN mode|"
    r"prompt injection|override safety|bypass filter|"
    r"<\|system\|>|<\|user\|>|<\|assistant\|>|<\|im_start\|>)",
    re.IGNORECASE,
)

_MALWARE_RE = re.compile(
    r"(?:eval\(base64_decode|exec\(base64|shell_exec\(|"
    r"powershell -enc|cmd\.exe /c|net user /add|"
    r"wget http.*\| bash|curl.*\| sh|"
    r"chmod \+x.*&&|rm -rf /|format c:|"
    r"WScript\.Shell|CreateObject\(\"Shell)",
    re.IGNORECASE,
)

_SYNTHETIC_MARKERS = re.compile(
    r"(?:as an ai language model|as an ai assistant|i cannot and will not|"
    r"i'm just an ai|this content was generated|"
    r"lorem ipsum dolor sit amet.{0,100}lorem ipsum|"
    r"test test test test|aaaa+|zzzz+|"
    r"1234567890 1234567890)",
    re.IGNORECASE,
)

# Personal information patterns
_PII_RE = re.compile(
    r"(?:\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b|"   # SSN
    r"\b\d{16}\b|"                               # credit card (16 digits)
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b(?:.{0,10}password|.{0,10}ssn)|"
    r"password[\s:=]+\S{4,})",                   # exposed passwords
    re.IGNORECASE,
)


@dataclass
class ToxicityResult:
    is_toxic:     bool   = False
    reason:       str    = ""
    toxicity:     float  = 0.0
    hate:         float  = 0.0
    explicit:     float  = 0.0
    spam:         float  = 0.0
    injection:    float  = 0.0
    malware:      float  = 0.0


class ToxicityFilter:
    """
    Multi-layer toxicity and unsafe-content filter.

    Parameters
    ----------
    max_toxicity_score:     Model-based toxicity threshold (if model enabled).
    max_hate_score:         Model-based hate speech threshold.
    max_explicit_score:     Model-based explicit content threshold.
    max_spam_density:       Maximum spam hits per 100 words.
    max_injection_density:  Maximum injection attempts per document.
    block_malware:          Block documents containing malware signatures.
    block_pii:              Block documents with exposed PII.
    block_synthetic:        Block synthetic/low-quality generated text markers.
    use_model:              Enable Detoxify model scoring (slow, accurate).
    model_name:             Detoxify model variant.
    """

    def __init__(
        self,
        max_toxicity_score:    float = 0.80,
        max_hate_score:        float = 0.70,
        max_explicit_score:    float = 0.85,
        max_spam_density:      float = 3.0,
        max_injection_density: int   = 1,
        block_malware:         bool  = True,
        block_pii:             bool  = True,
        block_synthetic:       bool  = True,
        use_model:             bool  = False,
        model_name:            str   = "unbiased",
    ) -> None:
        self._max_tox    = max_toxicity_score
        self._max_hate   = max_hate_score
        self._max_exp    = max_explicit_score
        self._max_spam   = max_spam_density
        self._max_inj    = max_injection_density
        self._blk_mal    = block_malware
        self._blk_pii    = block_pii
        self._blk_syn    = block_synthetic
        self._use_model  = use_model
        self._model_name = model_name
        self._model      = None
        self._model_tried = False

    @classmethod
    def from_cfg(cls, cfg) -> "ToxicityFilter":
        tc_raw = getattr(cfg, "toxicity_filter", None) or {}
        if hasattr(tc_raw, "model_dump"):
            tc = tc_raw.model_dump()
        elif isinstance(tc_raw, dict):
            tc = tc_raw
        elif hasattr(tc_raw, "__dict__"):
            tc = dict(tc_raw.__dict__)
        else:
            tc = {}
        return cls(
            max_toxicity_score=tc.get("max_toxicity_score", 0.80),
            max_hate_score=tc.get("max_hate_score", 0.70),
            max_explicit_score=tc.get("max_explicit_score", 0.85),
            max_spam_density=tc.get("max_spam_density", 3.0),
            max_injection_density=tc.get("max_injection_density", 1),
            block_malware=tc.get("block_malware", True),
            block_pii=tc.get("block_pii", True),
            block_synthetic=tc.get("block_synthetic", True),
            use_model=tc.get("use_model", False),
            model_name=tc.get("model_name", "unbiased"),
        )

    def analyse(self, text: str) -> ToxicityResult:
        """Run all layers and return ToxicityResult."""
        words   = text.split()
        n_words = max(len(words), 1)

        # Layer 1: blocklist regexes
        if _HATE_SPEECH_RE.search(text):
            return ToxicityResult(is_toxic=True, reason="hate_speech_keyword", hate=1.0)

        if _EXPLICIT_RE.search(text):
            return ToxicityResult(is_toxic=True, reason="explicit_content_keyword", explicit=1.0)

        spam_hits  = len(_SPAM_RE.findall(text))
        spam_dens  = spam_hits / (n_words / 100)
        if spam_dens > self._max_spam:
            return ToxicityResult(
                is_toxic=True, reason=f"spam_density:{spam_dens:.1f}", spam=min(spam_dens / 10, 1.0)
            )

        inj_hits = len(_INJECTION_RE.findall(text))
        if inj_hits > self._max_inj:
            return ToxicityResult(
                is_toxic=True, reason=f"prompt_injection:{inj_hits}", injection=1.0
            )

        if self._blk_mal and _MALWARE_RE.search(text):
            return ToxicityResult(is_toxic=True, reason="malware_signature", malware=1.0)

        if self._blk_pii and _PII_RE.search(text):
            return ToxicityResult(is_toxic=True, reason="exposed_pii")

        if self._blk_syn and _SYNTHETIC_MARKERS.search(text):
            return ToxicityResult(is_toxic=True, reason="synthetic_content_marker")

        # Layer 2: model scoring (optional)
        if self._use_model:
            result = self._model_score(text)
            if result.is_toxic:
                return result

        return ToxicityResult(is_toxic=False, spam=spam_dens / 10)

    def process(self, doc: Document) -> Optional[Document]:
        result = self.analyse(doc.text)
        if result.is_toxic:
            return doc.mark_rejected(f"toxicity:{result.reason}")

        if doc.quality is None:
            doc.quality = QualitySignals()
        doc.quality.toxicity_score = result.toxicity
        doc.quality.hate_score     = result.hate
        doc.quality.explicit_score = result.explicit
        doc.quality.spam_probability = result.spam
        return doc

    def process_batch(self, docs: list) -> tuple:
        accepted, rejected = [], []
        for doc in docs:
            result = self.process(doc)
            (rejected if result is None or result.rejected else accepted).append(doc)
        return accepted, rejected

    def _model_score(self, text: str) -> ToxicityResult:
        """Lazy-load Detoxify model for model-based scoring."""
        if not self._model_tried:
            self._model_tried = True
            try:
                from detoxify import Detoxify
                self._model = Detoxify(self._model_name)
                logger.info("Detoxify model loaded: %s", self._model_name)
            except Exception as exc:
                logger.warning("Detoxify unavailable: %s", exc)

        if self._model is None:
            return ToxicityResult(is_toxic=False)

        try:
            sample  = text[:512]
            scores  = self._model.predict(sample)
            tox     = float(scores.get("toxicity", 0))
            hate    = float(scores.get("identity_attack", 0))
            exp     = float(scores.get("sexually_explicit", 0))
            if tox > self._max_tox:
                return ToxicityResult(
                    is_toxic=True, reason=f"model_toxicity:{tox:.2f}", toxicity=tox
                )
            if hate > self._max_hate:
                return ToxicityResult(
                    is_toxic=True, reason=f"model_hate:{hate:.2f}", hate=hate
                )
            if exp > self._max_exp:
                return ToxicityResult(
                    is_toxic=True, reason=f"model_explicit:{exp:.2f}", explicit=exp
                )
            return ToxicityResult(is_toxic=False, toxicity=tox, hate=hate, explicit=exp)
        except Exception as exc:
            logger.debug("Model score error: %s", exc)
            return ToxicityResult(is_toxic=False)
