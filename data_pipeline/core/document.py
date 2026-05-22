"""
data_pipeline/core/document.py
================================
Unified internal document schema.

Every record produced by any extractor is normalised into a Document before
entering the preprocessing chain.  The schema is deliberately minimal on
required fields and rich on optional metadata so no source needs to fabricate
data it does not have.

Document objects are serialised to newline-delimited JSON (JSONL) for
streaming-safe, append-only shard writes.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class ContentDomain(str, Enum):
    ENCYCLOPEDIC   = "encyclopedic"
    BOOKS          = "books"
    FORUM          = "forum"
    CODE           = "code"
    QA             = "qa"
    DOCUMENTATION  = "documentation"
    ACADEMIC       = "academic"
    CONVERSATIONAL = "conversational"
    WEB            = "web"
    NEWS           = "news"
    MULTILINGUAL   = "multilingual"
    UNKNOWN        = "unknown"


class ProcessingStage(str, Enum):
    RAW        = "raw"
    EXTRACTED  = "extracted"
    NORMALIZED = "normalized"
    FILTERED   = "filtered"
    SCORED     = "scored"
    DEDUPED    = "deduped"
    ENRICHED   = "enriched"
    VALIDATED  = "validated"
    FINAL      = "final"
    REJECTED   = "rejected"


class LicenseType(str, Enum):
    CC_BY          = "CC-BY"
    CC_BY_SA       = "CC-BY-SA"
    CC0            = "CC0"
    MIT            = "MIT"
    APACHE_2       = "Apache-2.0"
    GPL            = "GPL"
    LGPL           = "LGPL"
    PROPRIETARY    = "proprietary"
    PUBLIC_DOMAIN  = "public_domain"
    UNKNOWN        = "unknown"


@dataclass
class QualitySignals:
    """Fine-grained quality signals computed by quality_scorer.py."""
    # Heuristic scores (0.0–1.0, higher = better quality)
    coherence_score:     float = 0.0
    readability_score:   float = 0.0
    factual_density:     float = 0.0
    educational_value:   float = 0.0
    reasoning_quality:   float = 0.0
    # Noise signals (0.0–1.0, higher = more noise)
    spam_probability:    float = 0.0
    boilerplate_ratio:   float = 0.0
    markup_noise:        float = 0.0
    repetition_ratio:    float = 0.0
    # Toxicity signals (0.0–1.0, higher = more toxic)
    toxicity_score:      float = 0.0
    explicit_score:      float = 0.0
    hate_score:          float = 0.0
    # Structural
    avg_word_length:     float = 0.0
    avg_sentence_length: float = 0.0
    alpha_ratio:         float = 0.0
    digit_ratio:         float = 0.0
    uppercase_ratio:     float = 0.0
    # Composite
    overall_quality:     float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QualitySignals":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DeduplicationState:
    """Deduplication hashes and state for this document."""
    exact_hash:      str = ""    # SHA-256 of normalised text
    simhash:         int = 0     # 64-bit SimHash for near-dup detection
    minhash_sig:     List[int] = field(default_factory=list)   # MinHash signature
    is_duplicate:    bool = False
    duplicate_of:    Optional[str] = None   # doc_id of canonical document
    near_dup_score:  float = 0.0            # Jaccard similarity to nearest dup

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DeduplicationState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Document:
    """
    Unified document schema across all DGB data pipeline sources.

    Required fields (must be set by every extractor):
        doc_id, text, source_name, stage

    All other fields are optional and filled progressively through
    the preprocessing chain.
    """
    # ── Identity ──────────────────────────────────────────────────────
    doc_id:          str = field(default_factory=lambda: str(uuid.uuid4()))

    # ── Content ───────────────────────────────────────────────────────
    text:            str = ""
    title:           str = ""
    url:             str = ""

    # ── Source provenance ─────────────────────────────────────────────
    source_name:     str = ""           # e.g. "wikipedia", "arxiv"
    source_url:      str = ""           # dump index URL
    dump_version:    str = ""           # e.g. "20260501"
    domain:          str = ContentDomain.UNKNOWN
    language:        str = "en"
    lang_confidence: float = 0.0
    category:        str = ""
    license:         str = LicenseType.UNKNOWN

    # ── Temporal ──────────────────────────────────────────────────────
    source_timestamp: Optional[str] = None   # original publication date
    ingested_at:      float = field(default_factory=time.time)

    # ── Size metrics ──────────────────────────────────────────────────
    char_count:      int = 0
    word_count:      int = 0
    token_estimate:  int = 0    # rough estimate: char_count // 4

    # ── Processing state ──────────────────────────────────────────────
    stage:           str = ProcessingStage.RAW
    rejected:        bool = False
    rejection_reason: str = ""

    # ── Quality and deduplication ─────────────────────────────────────
    quality:         Optional[QualitySignals] = None
    dedup:           Optional[DeduplicationState] = None

    # ── Extra source-specific metadata ────────────────────────────────
    metadata:        Dict[str, Any] = field(default_factory=dict)

    # ── Shard tracking ────────────────────────────────────────────────
    shard_id:        str = ""
    shard_offset:    int = 0

    def __post_init__(self) -> None:
        if not self.char_count and self.text:
            self.char_count  = len(self.text)
            self.word_count  = len(self.text.split())
            self.token_estimate = self.char_count // 4

    # ── Hashing ───────────────────────────────────────────────────────

    def compute_exact_hash(self) -> str:
        """SHA-256 of whitespace-normalised lowercased text."""
        import re
        normalised = re.sub(r"\s+", " ", self.text.lower().strip())
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    def ensure_dedup(self) -> "DeduplicationState":
        if self.dedup is None:
            self.dedup = DeduplicationState()
        if not self.dedup.exact_hash:
            self.dedup.exact_hash = self.compute_exact_hash()
        return self.dedup

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        # Flatten nested dataclasses
        if self.quality:
            d["quality"] = self.quality.to_dict()
        if self.dedup:
            d["dedup"] = self.dedup.to_dict()
        return d

    def to_jsonl(self) -> str:
        """Single JSONL line for shard writing."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str) + "\n"

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        q = d.pop("quality", None)
        dd = d.pop("dedup", None)
        doc = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if q:
            doc.quality = QualitySignals.from_dict(q)
        if dd:
            doc.dedup = DeduplicationState.from_dict(dd)
        return doc

    @classmethod
    def from_jsonl(cls, line: str) -> "Document":
        return cls.from_dict(json.loads(line))

    # ── Convenience ───────────────────────────────────────────────────

    def mark_rejected(self, reason: str) -> "Document":
        self.rejected         = True
        self.rejection_reason = reason
        self.stage            = ProcessingStage.REJECTED
        return self

    def is_valid(self) -> bool:
        return bool(self.text and self.source_name and not self.rejected)

    def text_preview(self, n: int = 120) -> str:
        return self.text[:n].replace("\n", " ") + ("…" if len(self.text) > n else "")

    def __repr__(self) -> str:
        return (
            f"Document(id={self.doc_id[:8]}… src={self.source_name} "
            f"lang={self.language} chars={self.char_count} stage={self.stage})"
        )