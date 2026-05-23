"""
data_pipeline/processors/quality_scorer.py
============================================
Stage 6: Quality scoring and heuristic filtering.

Computes a rich set of quality signals for each document and applies
configurable thresholds to filter out low-quality text.

Scoring dimensions
------------------
  Positive signals (higher = better):
    - alpha_ratio:         fraction of alphabetic characters
    - mean_word_len:       average word length (penalises gibberish)
    - mean_sent_len:       average sentence length (penalises one-liners)
    - paragraph_ratio:     fraction of text in multi-sentence paragraphs
    - unique_word_ratio:   vocabulary richness
    - educational_density: presence of explanatory connectives
    - punctuation_ok:      appropriate punctuation density

  Negative signals (higher = more noise):
    - digit_ratio:         excessive numbers (tables, coordinate data)
    - uppercase_ratio:     all-caps noise
    - repetition_ratio:    repeated n-grams or lines
    - boilerplate_ratio:   known boilerplate phrases
    - markup_noise:        residual HTML/wiki markup
    - short_line_ratio:    fraction of lines under 20 chars

  Composite:
    - overall_quality:     weighted combination of all signals

All thresholds and weights are externally configurable.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional

from data_pipeline.core.document import Document, ProcessingStage, QualitySignals

logger = logging.getLogger(__name__)

# ── Compiled patterns ─────────────────────────────────────────────────────────
_RE_SENT_SPLIT  = re.compile(r"(?<=[.!?])\s+")
_RE_MARKUP      = re.compile(r"<[^>]{1,80}>|\{\{[^}]{1,80}\}\}|\[\[[^\]]{1,80}\]\]")
_RE_BOILERPLATE = re.compile(
    r"(click here|privacy policy|terms of service|all rights reserved|"
    r"subscribe to our newsletter|copyright \d{4}|follow us on|"
    r"share this article|loading\.\.\.|advertisement|sponsored content|"
    r"read more|see also|external links|references\s*\[)",
    re.IGNORECASE,
)
_RE_URL_DENSE   = re.compile(r"https?://", re.IGNORECASE)

# Educational connectives — proxy for factual/reasoning content
_EDUCATIONAL_WORDS = frozenset([
    "because", "therefore", "however", "although", "furthermore",
    "moreover", "consequently", "nevertheless", "whereas", "thus",
    "hence", "since", "unless", "meanwhile", "despite", "analysis",
    "research", "study", "evidence", "according", "result", "conclusion",
    "theory", "experiment", "hypothesis", "data", "example", "definition",
    "explains", "demonstrates", "indicates", "suggests", "represents",
])


class QualityScorer:
    """
    Multi-dimensional quality scorer for text documents.

    Parameters (all configurable via runtime_config.json)
    ----------
    min_overall_quality:    Reject documents below this combined score.
    min_alpha_ratio:        Minimum fraction of alphabetic characters.
    max_digit_ratio:        Maximum fraction of digit characters.
    max_uppercase_ratio:    Maximum fraction of uppercase characters.
    max_repetition_ratio:   Maximum fraction of repeated n-grams.
    max_boilerplate_ratio:  Maximum fraction of boilerplate content.
    max_markup_ratio:       Maximum markup character density.
    min_unique_word_ratio:  Minimum vocabulary richness.
    min_mean_word_len:      Minimum average word length (removes gibberish).
    max_mean_word_len:      Maximum average word length (removes hash strings).
    min_words:              Minimum word count after normalisation.
    max_url_density:        Maximum URLs per 100 words.
    weights:                Dict of signal → float weights for overall score.
    """

    DEFAULT_WEIGHTS = {
        "alpha_ratio":        0.15,
        "unique_word_ratio":  0.10,
        "educational_density": 0.15,
        "paragraph_ratio":    0.10,
        "mean_word_len_ok":   0.10,
        "punct_ok":           0.05,
        # Negative (subtracted)
        "repetition_ratio":  -0.20,
        "markup_noise":      -0.15,
        "boilerplate_ratio": -0.10,
        "uppercase_ratio":   -0.05,
        "url_density":       -0.05,
    }

    def __init__(
        self,
        min_overall_quality:   float = 0.35,
        min_alpha_ratio:       float = 0.50,
        max_digit_ratio:       float = 0.30,
        max_uppercase_ratio:   float = 0.20,
        max_repetition_ratio:  float = 0.25,
        max_boilerplate_ratio: float = 0.15,
        max_markup_ratio:      float = 0.05,
        min_unique_word_ratio: float = 0.25,
        min_mean_word_len:     float = 3.0,
        max_mean_word_len:     float = 15.0,
        min_words:             int   = 20,
        max_url_density:       float = 5.0,
        weights:               Optional[Dict[str, float]] = None,
    ) -> None:
        self._min_qual     = min_overall_quality
        self._min_alpha    = min_alpha_ratio
        self._max_digit    = max_digit_ratio
        self._max_upper    = max_uppercase_ratio
        self._max_rep      = max_repetition_ratio
        self._max_bplate   = max_boilerplate_ratio
        self._max_markup   = max_markup_ratio
        self._min_unique   = min_unique_word_ratio
        self._min_wlen     = min_mean_word_len
        self._max_wlen     = max_mean_word_len
        self._min_words    = min_words
        self._max_url_dens = max_url_density
        self._weights      = weights or self.DEFAULT_WEIGHTS

    @classmethod
    def from_cfg(cls, cfg) -> "QualityScorer":
        qc = getattr(cfg, "quality_scorer", None) or {}
        if hasattr(qc, "__dict__"):
            qc = qc.__dict__
        return cls(
            min_overall_quality=qc.get("min_overall_quality", 0.35),
            min_alpha_ratio=qc.get("min_alpha_ratio", 0.50),
            max_digit_ratio=qc.get("max_digit_ratio", 0.30),
            max_uppercase_ratio=qc.get("max_uppercase_ratio", 0.20),
            max_repetition_ratio=qc.get("max_repetition_ratio", 0.25),
            max_boilerplate_ratio=qc.get("max_boilerplate_ratio", 0.15),
            max_markup_ratio=qc.get("max_markup_ratio", 0.05),
            min_unique_word_ratio=qc.get("min_unique_word_ratio", 0.25),
            min_mean_word_len=qc.get("min_mean_word_len", 3.0),
            max_mean_word_len=qc.get("max_mean_word_len", 15.0),
            min_words=qc.get("min_words", 20),
            max_url_density=qc.get("max_url_density", 5.0),
            weights=qc.get("weights"),
        )

    def score(self, doc: Document) -> QualitySignals:
        """Compute all quality signals and return a QualitySignals object."""
        text   = doc.text
        lines  = text.split("\n")
        words  = text.split()
        n      = max(len(text), 1)
        n_words = max(len(words), 1)

        # Character-level ratios
        alpha_count  = sum(1 for c in text if c.isalpha())
        digit_count  = sum(1 for c in text if c.isdigit())
        upper_count  = sum(1 for c in text if c.isupper())

        alpha_ratio   = alpha_count / n
        digit_ratio   = digit_count / n
        upper_ratio   = upper_count / max(alpha_count, 1)

        # Word statistics
        word_lengths  = [len(w) for w in words]
        mean_word_len = sum(word_lengths) / n_words
        unique_ratio  = len(set(w.lower() for w in words)) / n_words

        # Sentence statistics
        sentences     = _RE_SENT_SPLIT.split(text)
        n_sents       = max(len(sentences), 1)
        mean_sent_len = n_words / n_sents

        # Short-line ratio
        short_lines   = sum(1 for l in lines if 0 < len(l.strip()) < 20)
        short_ratio   = short_lines / max(len([l for l in lines if l.strip()]), 1)

        # Paragraph structure
        paras         = [p for p in text.split("\n\n") if p.strip()]
        long_paras    = sum(1 for p in paras if len(p.split()) >= 10)
        para_ratio    = long_paras / max(len(paras), 1)

        # Repetition (duplicate trigrams)
        tokens   = words[:500]   # cap for performance
        trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
        rep_ratio = (
            (len(trigrams) - len(set(trigrams))) / max(len(trigrams), 1)
        ) if trigrams else 0.0

        # Boilerplate
        bplate_hits = len(_RE_BOILERPLATE.findall(text))
        bplate_ratio = min(bplate_hits / max(n_sents, 1), 1.0)

        # Markup noise
        markup_chars = sum(len(m) for m in _RE_MARKUP.findall(text))
        markup_ratio = markup_chars / n

        # URL density
        url_count    = len(_RE_URL_DENSE.findall(text))
        url_density  = url_count / (n_words / 100)

        # Educational density
        lower_words  = {w.lower() for w in words}
        edu_hits     = len(lower_words & _EDUCATIONAL_WORDS)
        edu_density  = min(edu_hits / n_words * 10, 1.0)

        # Punctuation adequacy
        punct_count  = sum(1 for c in text if c in ".,;:!?")
        punct_ok     = min(punct_count / n_sents, 1.0)

        # Word-length sanity score
        mean_word_ok = 1.0 if self._min_wlen <= mean_word_len <= self._max_wlen else 0.3

        # Compute overall quality as weighted combination
        signals = {
            "alpha_ratio":        min(alpha_ratio / max(self._min_alpha, 0.01), 1.0),
            "unique_word_ratio":  min(unique_ratio / max(self._min_unique, 0.01), 1.0),
            "educational_density": edu_density,
            "paragraph_ratio":    para_ratio,
            "mean_word_len_ok":   mean_word_ok,
            "punct_ok":           punct_ok,
            "repetition_ratio":   rep_ratio,
            "markup_noise":       markup_ratio,
            "boilerplate_ratio":  bplate_ratio,
            "uppercase_ratio":    upper_ratio,
            "url_density":        min(url_density / max(self._max_url_dens, 1), 1.0),
        }
        overall = sum(
            self._weights.get(k, 0) * v for k, v in signals.items()
        )
        overall = max(0.0, min(1.0, overall + 0.5))   # centre around 0.5

        return QualitySignals(
            coherence_score=para_ratio,
            readability_score=min(mean_sent_len / 20, 1.0),
            factual_density=edu_density,
            educational_value=edu_density,
            reasoning_quality=min(edu_density * para_ratio * 1.5, 1.0),
            spam_probability=min(bplate_ratio + url_density / 10, 1.0),
            boilerplate_ratio=bplate_ratio,
            markup_noise=markup_ratio,
            repetition_ratio=rep_ratio,
            avg_word_length=round(mean_word_len, 2),
            avg_sentence_length=round(mean_sent_len, 2),
            alpha_ratio=round(alpha_ratio, 4),
            digit_ratio=round(digit_ratio, 4),
            uppercase_ratio=round(upper_ratio, 4),
            overall_quality=round(overall, 4),
        )

    def process(self, doc: Document) -> Optional[Document]:
        """Score and filter one document. Returns None if rejected."""
        if len(doc.text.split()) < self._min_words:
            return doc.mark_rejected(f"too_few_words:{len(doc.text.split())}")

        qs  = self.score(doc)
        doc.quality = qs

        # Hard filters (fast-fail before overall score)
        if qs.alpha_ratio < self._min_alpha:
            return doc.mark_rejected(f"low_alpha:{qs.alpha_ratio:.3f}")
        if qs.digit_ratio > self._max_digit:
            return doc.mark_rejected(f"high_digit:{qs.digit_ratio:.3f}")
        if qs.uppercase_ratio > self._max_upper:
            return doc.mark_rejected(f"high_uppercase:{qs.uppercase_ratio:.3f}")
        if qs.repetition_ratio > self._max_rep:
            return doc.mark_rejected(f"high_repetition:{qs.repetition_ratio:.3f}")
        if qs.boilerplate_ratio > self._max_bplate:
            return doc.mark_rejected(f"high_boilerplate:{qs.boilerplate_ratio:.3f}")
        if qs.markup_noise > self._max_markup:
            return doc.mark_rejected(f"high_markup:{qs.markup_noise:.4f}")

        # Composite quality gate
        if qs.overall_quality < self._min_qual:
            return doc.mark_rejected(f"low_quality:{qs.overall_quality:.3f}")

        doc.stage = ProcessingStage.SCORED
        return doc

    def process_batch(self, docs: list) -> tuple:
        accepted, rejected = [], []
        for doc in docs:
            result = self.process(doc)
            (rejected if result is None or result.rejected else accepted).append(doc)
        return accepted, rejected
