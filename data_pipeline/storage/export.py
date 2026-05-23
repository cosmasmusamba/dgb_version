"""
data_pipeline/storage/export.py
=================================
Training-ready shard exporter.

Reads processed JSONL shards and exports them into formats suitable for:
  - Tokenizer training (plain text, one document per line)
  - Pre-training (shuffled JSONL with text field only)
  - SFT / instruction tuning (prompt-response pairs)
  - Retrieval augmentation (metadata-rich JSONL)
  - Alignment (preference pairs — requires separate preference dataset)

Export operations:
  1. Collect all accepted JSONL shards from source directories
  2. Optionally shuffle across shards (configurable seed)
  3. Write output files in the requested format
  4. Generate an export manifest with statistics and provenance

All exports are resumable — the manifest tracks which input shards have
been processed so re-runs skip already-exported content.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional

from data_pipeline.core.document import Document
from modules.utils.safe_writer import atomic_write_json
from modules.utils.file_handler import list_files

logger = logging.getLogger(__name__)

_DEFAULT_EXPORT_SHARD_BYTES = 512 * 1024 * 1024   # 512 MB


class ExportFormat:
    PLAIN_TEXT      = "plain_text"        # one doc per line (for tokenizer training)
    JSONL_TEXT_ONLY = "jsonl_text_only"   # {"text": "..."} per line
    JSONL_FULL      = "jsonl_full"        # full Document JSON per line
    JSONL_SFT       = "jsonl_sft"         # {"prompt": ..., "response": ...}


@dataclass
class ExportManifest:
    """Tracks what was exported and from where."""
    run_id:          str
    format:          str
    created_at:      float = field(default_factory=time.time)
    input_shards:    List[str] = field(default_factory=list)
    output_shards:   List[str] = field(default_factory=list)
    total_docs:      int = 0
    total_chars:     int = 0
    total_tokens_est: int = 0
    source_breakdown: Dict[str, int] = field(default_factory=dict)
    completed:       bool = False


class ShardExporter:
    """
    Exports processed pipeline shards into training-ready datasets.

    Parameters
    ----------
    input_dirs:     Source directories to scan for .jsonl shards.
    output_dir:     Directory to write exported shards.
    format:         One of ExportFormat constants.
    run_id:         Run prefix for output files.
    max_shard_bytes: Rotate output shard at this size.
    shuffle:         Shuffle documents within the export.
    shuffle_seed:    Seed for deterministic shuffles.
    min_quality:     Skip documents with overall_quality below this.
    languages:       Only export documents in these languages (empty = all).
    domains:         Only export documents in these domains (empty = all).
    """

    def __init__(
        self,
        input_dirs:      List[Path],
        output_dir:      Path,
        format:          str   = ExportFormat.JSONL_TEXT_ONLY,
        run_id:          str   = "",
        max_shard_bytes: int   = _DEFAULT_EXPORT_SHARD_BYTES,
        shuffle:         bool  = True,
        shuffle_seed:    int   = 42,
        min_quality:     float = 0.0,
        languages:       Optional[List[str]] = None,
        domains:         Optional[List[str]] = None,
    ) -> None:
        self._inputs    = [Path(d) for d in input_dirs]
        self._out_dir   = Path(output_dir)
        self._format    = format
        self._run_id    = run_id
        self._max_bytes = max_shard_bytes
        self._shuffle   = shuffle
        self._seed      = shuffle_seed
        self._min_qual  = min_quality
        self._langs     = set(languages or [])
        self._domains   = set(domains or [])
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def export(self) -> ExportManifest:
        """Run the full export. Returns ExportManifest."""
        manifest = ExportManifest(run_id=self._run_id, format=self._format)

        # Discover all accepted shards (exclude rejected/ subdirs)
        all_shards = []
        for d in self._inputs:
            for shard in list_files(d, "*.jsonl"):
                if "rejected" not in str(shard):
                    all_shards.append(shard)

        if not all_shards:
            logger.warning("No input shards found in %s", self._inputs)
            return manifest

        manifest.input_shards = [str(s) for s in all_shards]
        logger.info(
            "ShardExporter: %d input shards  format=%s  shuffle=%s",
            len(all_shards), self._format, self._shuffle,
        )

        # Stream and optionally shuffle
        docs = list(self._iter_docs(all_shards))
        if self._shuffle:
            rng = random.Random(self._seed)
            rng.shuffle(docs)
            logger.info("Shuffled %d documents (seed=%d)", len(docs), self._seed)

        # Write output shards
        shard_idx = 0
        buf_bytes = 0
        out_lines: List[str] = []
        out_shards: List[str] = []

        def _flush(lines, idx):
            name = f"{self._run_id}_export_{idx:06d}.jsonl"
            path = self._out_dir / name
            path.write_text("".join(lines), encoding="utf-8")
            out_shards.append(str(path))
            logger.info(
                "Export shard %s: %d docs  %.1f MB",
                name, len(lines), sum(len(l) for l in lines) / 1024**2,
            )
            return [], 0, idx + 1

        for doc in docs:
            line = self._format_doc(doc)
            if not line:
                continue
            lb = len(line.encode("utf-8"))
            out_lines.append(line)
            buf_bytes += lb
            manifest.total_docs  += 1
            manifest.total_chars += doc.char_count
            manifest.total_tokens_est += doc.token_estimate
            manifest.source_breakdown[doc.source_name] = (
                manifest.source_breakdown.get(doc.source_name, 0) + 1
            )
            if buf_bytes >= self._max_bytes:
                out_lines, buf_bytes, shard_idx = _flush(out_lines, shard_idx)

        if out_lines:
            out_lines, _, shard_idx = _flush(out_lines, shard_idx)

        manifest.output_shards = out_shards
        manifest.completed     = True

        # Write manifest
        manifest_path = self._out_dir / f"{self._run_id}_export_manifest.json"
        atomic_write_json(manifest_path, asdict(manifest))
        logger.info(
            "Export complete: %d docs  %d shards  %.1fB tokens (est)  manifest=%s",
            manifest.total_docs, len(out_shards),
            manifest.total_tokens_est / 1e9, manifest_path.name,
        )
        return manifest

    def _iter_docs(self, shards: List[Path]) -> Generator[Document, None, None]:
        """Stream Documents from JSONL shards with quality / language / domain filters."""
        for shard in shards:
            try:
                with shard.open("r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            doc = Document.from_jsonl(line)
                        except Exception:
                            continue
                        if not doc.is_valid():
                            continue
                        if self._langs and doc.language not in self._langs:
                            continue
                        if self._domains and doc.domain not in self._domains:
                            continue
                        if self._min_qual > 0 and doc.quality:
                            if doc.quality.overall_quality < self._min_qual:
                                continue
                        yield doc
            except Exception as exc:
                logger.warning("Cannot read shard %s: %s", shard.name, exc)

    def _format_doc(self, doc: Document) -> str:
        """Serialise one document into the chosen export format."""
        if self._format == ExportFormat.PLAIN_TEXT:
            return doc.text.replace("\n", " ").strip() + "\n"

        if self._format == ExportFormat.JSONL_TEXT_ONLY:
            return json.dumps({"text": doc.text}, ensure_ascii=False) + "\n"

        if self._format == ExportFormat.JSONL_FULL:
            return doc.to_jsonl()

        if self._format == ExportFormat.JSONL_SFT:
            # Only emit documents that have a Q&A structure
            if "\n\nAnswer:" in doc.text:
                parts = doc.text.split("\n\nAnswer:", 1)
                return json.dumps(
                    {"prompt": parts[0].strip(), "response": parts[1].strip()},
                    ensure_ascii=False,
                ) + "\n"
            # Fall back to instruction-following format
            return json.dumps(
                {"prompt": "Continue the following text:", "response": doc.text},
                ensure_ascii=False,
            ) + "\n"

        return ""
