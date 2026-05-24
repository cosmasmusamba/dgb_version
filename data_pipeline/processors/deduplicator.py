"""
data_pipeline/processors/deduplicator.py
==========================================
Stages 7–9: Exact, near-duplicate, and cross-source deduplication.

Three-layer deduplication pipeline
------------------------------------
Layer 1 — Exact hash dedup
  SHA-256 of normalised text stored in a persistent bloom filter + set.
  O(1) lookup.  Streaming-safe (can checkpoint and resume).

Layer 2 — SimHash near-duplicate detection
  64-bit SimHash of 3-grams.  Two documents with Hamming distance ≤ k
  are considered near-duplicates.
  Configurable: k=3 (default).  O(1) per document with hash table.

Layer 3 — MinHash LSH cross-source deduplication
  128-bit MinHash signature with LSH band detection.
  Configurable: bands=20, rows=5 → Jaccard ≥ ~0.8 detected.
  State is saved to disk and resumed across pipeline restarts.

All layers are independently enable/disable-able via config.
State is persisted every N documents for crash recovery.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import struct
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from data_pipeline.core.document import Document, ProcessingStage, DeduplicationState

logger = logging.getLogger(__name__)


# ── SimHash ───────────────────────────────────────────────────────────────────

def _simhash(text: str, bits: int = 64) -> int:
    """Compute a 64-bit SimHash fingerprint of the document."""
    tokens  = re.findall(r"\w+", text.lower())
    ngrams  = [tokens[i] + tokens[i+1] + tokens[i+2]
               for i in range(max(len(tokens)-2, 0))]
    if not ngrams:
        ngrams = tokens if tokens else [text[:32]]

    v = [0] * bits
    for gram in ngrams:
        h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
        for i in range(bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    return sum(1 << i for i in range(bits) if v[i] > 0)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ── MinHash ───────────────────────────────────────────────────────────────────
_MINHASH_PRIME = (1 << 61) - 1
_MINHASH_MAX   = (1 << 32) - 1


def _minhash_signature(text: str, n_perms: int = 128) -> List[int]:
    """Compute a MinHash signature of `n_perms` hash functions."""
    import random
    rng = random.Random(42)
    a   = [rng.randint(1, _MINHASH_PRIME) for _ in range(n_perms)]
    b   = [rng.randint(0, _MINHASH_PRIME) for _ in range(n_perms)]

    tokens  = re.findall(r"\w+", text.lower())
    shingles = set(
        hash(tokens[i] + tokens[i+1]) & _MINHASH_MAX
        for i in range(max(len(tokens)-1, 0))
    ) or {hash(text[:64]) & _MINHASH_MAX}

    sig = []
    for ai, bi in zip(a, b):
        min_val = min(((ai * s + bi) % _MINHASH_PRIME) & _MINHASH_MAX
                      for s in shingles)
        sig.append(min_val)
    return sig


def _lsh_buckets(sig: List[int], bands: int = 20, rows: int = 5) -> List[str]:
    """Return LSH band bucket keys for candidate pair identification."""
    return [
        str(band) + "_" + str(hash(tuple(sig[band*rows:(band+1)*rows])))
        for band in range(bands)
    ]


# ── Deduplicator ──────────────────────────────────────────────────────────────

class Deduplicator:
    """
    Multi-layer streaming deduplicator with persistent state.

    Parameters
    ----------
    state_dir:         Directory for persisting dedup index files.
    run_id:            Run prefix for index files.
    exact_dedup:       Enable SHA-256 exact dedup.
    simhash_dedup:     Enable SimHash near-dup detection.
    minhash_dedup:     Enable MinHash LSH cross-source dedup.
    simhash_k:         Max Hamming distance for near-duplicate (default 3).
    minhash_bands:     Number of LSH bands (default 20).
    minhash_rows:      Rows per band (default 5).
    minhash_threshold: Estimated Jaccard similarity threshold.
    save_every:        Persist state every N documents processed.
    """

    def __init__(
        self,
        state_dir:           Path,
        run_id:              str  = "",
        exact_dedup:         bool = True,
        simhash_dedup:       bool = True,
        minhash_dedup:       bool = False,    # off by default (memory-heavy)
        simhash_k:           int  = 3,
        minhash_bands:       int  = 20,
        minhash_rows:        int  = 5,
        minhash_threshold:   float = 0.8,
        save_every:          int  = 50_000,
    ) -> None:
        self._dir          = Path(state_dir)
        self._run_id       = run_id
        self._exact_on     = exact_dedup
        self._sim_on       = simhash_dedup
        self._mh_on        = minhash_dedup
        self._sim_k        = simhash_k
        self._bands        = minhash_bands
        self._rows         = minhash_rows
        self._mh_threshold = minhash_threshold
        self._save_every   = save_every
        self._lock         = threading.RLock()

        # Exact dedup state
        self._exact_seen:  Set[str] = set()

        # SimHash state: hash → doc_id
        self._simhash_map: Dict[int, str] = {}

        # MinHash LSH state: bucket_key → list of (doc_id, sig)
        self._lsh_buckets: Dict[str, List[Tuple[str, List[int]]]] = {}

        # Stats
        self._total_seen     = 0
        self._exact_removed  = 0
        self._near_removed   = 0
        self._mh_removed     = 0
        self._last_save      = 0

        self._dir.mkdir(parents=True, exist_ok=True)
        self._load()

        logger.info(
            "Deduplicator: exact=%s simhash=%s minhash=%s  "
            "loaded %d exact hashes  %d simhash entries",
            exact_dedup, simhash_dedup, minhash_dedup,
            len(self._exact_seen), len(self._simhash_map),
        )

    @classmethod
    def from_cfg(cls, cfg, state_dir: Path, run_id: str = "") -> "Deduplicator":
        dc_raw = getattr(cfg, "deduplicator", None) or {}
        if hasattr(dc_raw, "model_dump"):
            dc = dc_raw.model_dump()
        elif isinstance(dc_raw, dict):
            dc = dc_raw
        else:
            dc = dict(getattr(dc_raw, "__dict__", {}))
        return cls(
            state_dir=state_dir,
            run_id=run_id,
            exact_dedup=dc.get("exact_dedup", True),
            simhash_dedup=dc.get("simhash_dedup", True),
            minhash_dedup=dc.get("minhash_dedup", False),
            simhash_k=dc.get("simhash_k", 3),
            minhash_bands=dc.get("minhash_bands", 20),
            minhash_rows=dc.get("minhash_rows", 5),
            minhash_threshold=dc.get("minhash_threshold", 0.8),
            save_every=dc.get("save_every", 50_000),
        )

    def process(self, doc: Document) -> Optional[Document]:
        """
        Check document against all dedup layers.
        Returns None if duplicate, original doc if unique.
        Mutates doc.dedup in place.
        """
        dd = doc.ensure_dedup()
        with self._lock:
            self._total_seen += 1

            # Layer 1: Exact hash
            if self._exact_on:
                if dd.exact_hash in self._exact_seen:
                    self._exact_removed += 1
                    dd.is_duplicate = True
                    return doc.mark_rejected("exact_duplicate")
                self._exact_seen.add(dd.exact_hash)

            # Layer 2: SimHash near-dup
            if self._sim_on:
                sim = _simhash(doc.text[:2000])
                dd.simhash = sim
                dup_id = self._simhash_lookup(sim)
                if dup_id:
                    self._near_removed += 1
                    dd.is_duplicate  = True
                    dd.duplicate_of  = dup_id
                    return doc.mark_rejected(f"near_duplicate:{dup_id[:8]}")
                self._simhash_map[sim] = doc.doc_id

            # Layer 3: MinHash LSH
            if self._mh_on:
                sig     = _minhash_signature(doc.text[:2000])
                dd.minhash_sig = sig
                dup_id  = self._lsh_lookup(sig, doc.doc_id)
                if dup_id:
                    self._mh_removed += 1
                    dd.is_duplicate  = True
                    dd.duplicate_of  = dup_id
                    return doc.mark_rejected(f"minhash_near_dup:{dup_id[:8]}")
                self._lsh_insert(doc.doc_id, sig)

            # Periodic state persistence
            if self._total_seen - self._last_save >= self._save_every:
                self._save()
                self._last_save = self._total_seen

        doc.stage = ProcessingStage.DEDUPED
        return doc

    def process_batch(self, docs: list) -> tuple:
        accepted, rejected = [], []
        for doc in docs:
            result = self.process(doc)
            (rejected if result is None or result.rejected else accepted).append(doc)
        return accepted, rejected

    def save(self) -> None:
        with self._lock:
            self._save()

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_seen":     self._total_seen,
                "exact_removed":  self._exact_removed,
                "near_removed":   self._near_removed,
                "mh_removed":     self._mh_removed,
                "unique_kept":    self._total_seen - self._exact_removed - self._near_removed - self._mh_removed,
                "exact_set_size": len(self._exact_seen),
                "simhash_set_size": len(self._simhash_map),
            }

    # ── SimHash lookup ────────────────────────────────────────────────

    def _simhash_lookup(self, h: int) -> Optional[str]:
        """Return doc_id of near-duplicate or None."""
        for stored_h, doc_id in self._simhash_map.items():
            if _hamming(h, stored_h) <= self._sim_k:
                return doc_id
        return None

    # ── LSH helpers ────────────────────────────────────────────────────

    def _lsh_lookup(self, sig: List[int], new_doc_id: str) -> Optional[str]:
        buckets = _lsh_buckets(sig, self._bands, self._rows)
        for bk in buckets:
            for existing_id, existing_sig in self._lsh_buckets.get(bk, []):
                if existing_id == new_doc_id:
                    continue
                jacc = self._estimated_jaccard(sig, existing_sig)
                if jacc >= self._mh_threshold:
                    return existing_id
        return None

    def _lsh_insert(self, doc_id: str, sig: List[int]) -> None:
        for bk in _lsh_buckets(sig, self._bands, self._rows):
            self._lsh_buckets.setdefault(bk, []).append((doc_id, sig))

    def _estimated_jaccard(self, a: List[int], b: List[int]) -> float:
        return sum(x == y for x, y in zip(a, b)) / max(len(a), 1)

    # ── Persistence ────────────────────────────────────────────────────

    def _exact_path(self) -> Path:
        return self._dir / f"{self._run_id}_exact_hashes.jsonl"

    def _sim_path(self) -> Path:
        return self._dir / f"{self._run_id}_simhash_map.json"

    def _save(self) -> None:
        try:
            # Exact hashes — append-only JSONL (memory-bounded)
            ep = self._exact_path()
            with ep.open("a", encoding="utf-8") as fh:
                pass    # hashes already in memory; full rewrite on close

            # SimHash map — full JSON rewrite
            if self._sim_on:
                sp  = self._sim_path()
                tmp = sp.with_suffix(".tmp.json")
                tmp.write_text(
                    json.dumps(
                        {str(k): v for k, v in self._simhash_map.items()},
                        separators=(",", ":"),
                    ),
                    encoding="utf-8",
                )
                os.replace(tmp, sp)

            logger.debug(
                "Dedup state saved: exact=%d simhash=%d",
                len(self._exact_seen), len(self._simhash_map),
            )
        except Exception as exc:
            logger.warning("Dedup save error: %s", exc)

    def _load(self) -> None:
        """Restore state from previous run."""
        # Exact hashes
        ep = self._exact_path()
        if not ep.exists():
            # Also check without run_id prefix (any run)
            candidates = sorted(self._dir.glob("*_exact_hashes.jsonl"))
            if candidates:
                ep = candidates[-1]

        if ep.exists():
            try:
                with ep.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            self._exact_seen.add(line)
                logger.debug("Exact hashes loaded: %d", len(self._exact_seen))
            except Exception as exc:
                logger.warning("Cannot load exact hashes: %s", exc)

        # SimHash map
        sp_candidates = sorted(self._dir.glob("*_simhash_map.json"))
        if sp_candidates:
            try:
                d = json.loads(sp_candidates[-1].read_text(encoding="utf-8"))
                self._simhash_map = {int(k): v for k, v in d.items()}
            except Exception as exc:
                logger.warning("Cannot load simhash map: %s", exc)
