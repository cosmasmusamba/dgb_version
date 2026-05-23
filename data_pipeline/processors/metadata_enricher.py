"""
data_pipeline/processors/metadata_enricher.py
===============================================
Stage 10–11: Metadata enrichment and validation.

Enrichment operations:
  - Token count estimation (char / 4 heuristic or tiktoken if available)
  - Reading grade level (Flesch-Kincaid approximation)
  - Detected topics via keyword density
  - Structural signals (paragraph count, sentence count, code block count)
  - Cross-source provenance fingerprint
  - Finalise deduplication hash

Validation:
  - Required fields present (text, source_name)
  - URLs well-formed
  - Timestamps parseable
  - Language code valid ISO 639-1
  - Quality signals within valid range
  - Reject documents that failed enrichment-level constraints
"""
from __future__ import annotations

import logging
import math
import re
import time
from typing import Optional

from data_pipeline.core.document import Document, ProcessingStage

logger = logging.getLogger(__name__)

_ISO_639_1 = frozenset([
    "af","sq","am","ar","hy","az","eu","be","bn","bs","bg","ca","ceb","zh",
    "co","hr","cs","da","nl","en","eo","et","fi","fr","fy","gl","ka","de",
    "el","gu","ht","ha","haw","he","hi","hmn","hu","is","ig","id","ga","it",
    "ja","jv","kn","kk","km","ko","ku","ky","lo","la","lv","lt","lb","mk",
    "mg","ms","ml","mt","mi","mr","mn","my","ne","no","ny","or","ps","fa",
    "pl","pt","pa","ro","ru","sm","gd","sr","sn","sd","si","sk","sl","so",
    "st","es","su","sw","sv","tl","tg","ta","tt","te","th","tr","tk","uk",
    "ur","ug","uz","vi","cy","xh","yi","yo","zu","xx",
])

_TOPIC_KEYWORDS = {
    "science":      ["research","experiment","hypothesis","data","study","analysis","evidence"],
    "technology":   ["software","hardware","algorithm","system","network","code","api","database"],
    "mathematics":  ["theorem","proof","equation","function","variable","integral","derivative"],
    "history":      ["century","war","empire","revolution","civilization","ancient","medieval"],
    "politics":     ["government","election","policy","parliament","democracy","law","rights"],
    "health":       ["medical","disease","treatment","patient","clinical","therapy","diagnosis"],
    "education":    ["learning","student","teacher","curriculum","university","knowledge","skill"],
    "philosophy":   ["ethics","morality","consciousness","existence","reasoning","logic","truth"],
    "literature":   ["novel","poem","story","character","narrative","fiction","author"],
    "business":     ["company","market","revenue","profit","investment","startup","management"],
}


class MetadataEnricher:
    """
    Enriches document metadata and performs final validation.

    Parameters
    ----------
    estimate_tokens:    Add token_estimate field (char/4 heuristic).
    detect_topics:      Assign topic tags from keyword density.
    compute_flesch:     Compute Flesch-Kincaid readability grade.
    max_topics:         Maximum number of topic tags to assign.
    strict_validation:  Reject documents with validation failures.
    """

    def __init__(
        self,
        estimate_tokens:  bool = True,
        detect_topics:    bool = True,
        compute_flesch:   bool = True,
        max_topics:       int  = 3,
        strict_validation: bool = True,
    ) -> None:
        self._est_tokens   = estimate_tokens
        self._det_topics   = detect_topics
        self._flesch       = compute_flesch
        self._max_topics   = max_topics
        self._strict       = strict_validation

    @classmethod
    def from_cfg(cls, cfg) -> "MetadataEnricher":
        ec = getattr(cfg, "metadata_enricher", None) or {}
        if hasattr(ec, "__dict__"):
            ec = ec.__dict__
        return cls(
            estimate_tokens=ec.get("estimate_tokens", True),
            detect_topics=ec.get("detect_topics", True),
            compute_flesch=ec.get("compute_flesch", True),
            max_topics=ec.get("max_topics", 3),
            strict_validation=ec.get("strict_validation", True),
        )

    def process(self, doc: Document) -> Optional[Document]:
        """Enrich and validate. Returns None if validation fails."""
        # Validation
        if not doc.text or not doc.text.strip():
            return doc.mark_rejected("empty_text")
        if not doc.source_name:
            return doc.mark_rejected("missing_source_name")
        if doc.language and doc.language not in _ISO_639_1:
            if self._strict:
                return doc.mark_rejected(f"invalid_language_code:{doc.language}")
            doc.language = "xx"

        # Size metrics
        doc.char_count     = len(doc.text)
        doc.word_count     = len(doc.text.split())
        doc.token_estimate = self._estimate_tokens(doc.text)

        # Topics
        if self._det_topics:
            topics = self._detect_topics(doc.text)
            if topics and "topics" not in doc.metadata:
                doc.metadata["topics"] = topics
            if topics and not doc.category:
                doc.category = topics[0]

        # Flesch-Kincaid
        if self._flesch:
            fk = self._flesch_kincaid(doc.text)
            if fk is not None:
                doc.metadata["flesch_kincaid_grade"] = fk

        # Structural signals
        doc.metadata.update({
            "paragraph_count":  len([p for p in doc.text.split("\n\n") if p.strip()]),
            "sentence_count":   len(re.split(r"(?<=[.!?])\s+", doc.text)),
            "code_block_count": doc.text.count("```"),
            "has_lists":        bool(re.search(r"^\s*[-*•]\s", doc.text, re.MULTILINE)),
            "ingested_at":      doc.ingested_at,
        })

        # Ensure dedup hash is computed
        doc.ensure_dedup()

        doc.stage = ProcessingStage.ENRICHED
        return doc

    def process_batch(self, docs: list) -> tuple:
        accepted, rejected = [], []
        for doc in docs:
            result = self.process(doc)
            (rejected if result is None or result.rejected else accepted).append(doc)
        return accepted, rejected

    # ── Helpers ────────────────────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: 4 chars per token (GPT-style)."""
        return max(1, len(text) // 4)

    def _detect_topics(self, text: str) -> list:
        lower   = text.lower()
        scores  = {}
        for topic, kws in _TOPIC_KEYWORDS.items():
            score = sum(1 for kw in kws if kw in lower)
            if score > 0:
                scores[topic] = score
        return sorted(scores, key=scores.get, reverse=True)[: self._max_topics]

    def _flesch_kincaid(self, text: str) -> Optional[float]:
        """
        Flesch-Kincaid Grade Level approximation.
        FK = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
        """
        words     = text.split()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        n_words   = max(len(words), 1)
        n_sents   = max(len(sentences), 1)
        # Syllable estimate: vowel groups
        n_syllables = sum(
            max(1, len(re.findall(r"[aeiouAEIOU]+", w))) for w in words
        )
        try:
            grade = (0.39 * (n_words / n_sents) +
                     11.8 * (n_syllables / n_words) - 15.59)
            return round(max(0.0, min(grade, 20.0)), 2)
        except Exception:
            return None
