# DGB Data Pipeline — Architecture & Operations Guide
**Version:** 3.0.0 | **Module:** `data_pipeline/`

---

## Overview

The DGB data pipeline is a modular, streaming-first, memory-bounded ingestion system capable of continuously acquiring, cleaning, deduplicating, scoring, and sharding text corpora from heterogeneous sources without requiring full upfront downloads.

Everything is controlled through `configs/runtime_config.json` — no code changes are needed to add sources, adjust thresholds, change quotas, or toggle preprocessing stages.

---

## Quickstart

```bash
# Dry-run — validates config and lists enabled sources
python main_data_pipeline.py --dry-run

# Run all enabled sources
python main_data_pipeline.py

# Run specific sources
python main_data_pipeline.py --sources wikipedia arxiv

# Check status of a pipeline run
python main_data_pipeline.py --status

# Force re-run a completed source
python main_data_pipeline.py --force wikipedia

# Export to training format
python main_data_pipeline.py --export --format plain_text
python main_data_pipeline.py --export --format jsonl_text_only
python main_data_pipeline.py --export --format jsonl_sft
```

---

## Directory structure

```
data_pipeline/
├── config/
│   └── pipeline_config.py       Typed config manager with runtime overrides
├── core/
│   ├── document.py              Unified Document schema + JSONL serialisation
│   ├── checkpoint.py            Durable multi-level checkpoint/resume system
│   ├── shard_writer.py          Rolling shard writer with size limits + atomic rotation
│   ├── quota_manager.py         Per-source storage quota + disk exhaustion prevention
│   └── pipeline_stages.py       Preprocessing chain assembler
├── extractors/
│   ├── base_extractor.py        Abstract base with HTTP streaming, retry, resume
│   ├── wikipedia_extractor.py   Wikipedia BZ2/XML dump streaming
│   ├── stackexchange_extractor.py  SE Q&A pair extraction
│   ├── arxiv_extractor.py       arXiv abstract metadata streaming
│   ├── gutenberg_extractor.py   Project Gutenberg book streaming
│   ├── commoncrawl_extractor.py WARC streaming with trafilatura text extraction
│   └── github_extractor.py      Code repository document extraction
├── processors/
│   ├── normalizer.py            Unicode/whitespace/HTML normalisation
│   ├── language_filter.py       fasttext language detection + filtering
│   ├── toxicity_filter.py       Keyword blocklist + optional Detoxify model
│   ├── quality_scorer.py        12-signal quality scoring + threshold filtering
│   ├── deduplicator.py          Exact SHA-256 + SimHash + MinHash LSH dedup
│   └── metadata_enricher.py     Topic detection, Flesch-Kincaid, structural signals
├── storage/
│   └── export.py                Training-ready shard exporter
└── workers/
    ├── source_worker.py         Per-source async worker
    └── pipeline_orchestrator.py Top-level concurrent orchestrator

datasets/
├── wikipedia/                   Accepted shards + rejected/ + dedup_state/
├── stackexchange/
├── arxiv/
├── gutenberg/
├── commoncrawl/
├── github/
└── export/                      Training-ready exported shards
    ├── plain_text/
    ├── jsonl_text_only/
    └── jsonl_sft/
```

---

## Configuring sources

Enable or disable any source in `configs/runtime_config.json`:

```json
"sources": {
  "wikipedia": {
    "enabled": true,
    "extra": { "local_dir": "datasets/dgb1/wk_raw" }
  },
  "arxiv": {
    "enabled": true,
    "extra": { "categories": ["cs", "math", "physics"] }
  }
}
```

---

## Preprocessing stages

All stages are individually togglable:

```json
"pipeline_stages": {
  "normalize":        true,
  "language_filter":  true,
  "toxicity_filter":  true,
  "quality_scorer":   true,
  "deduplicator":     true
}
```

Stage order: normalize → language_filter → toxicity_filter → quality_scorer → deduplicator

---

## Storage quotas

```json
"storage_quotas": {
  "global_max_gb":    500,
  "safety_margin_gb": 10,
  "wikipedia_gb":     100,
  "arxiv_gb":         30,
  "commoncrawl_gb":   200
}
```

When a source reaches its quota, extraction pauses automatically. When disk free space falls below `safety_margin_gb`, ALL sources pause until space is freed.

---

## Resume semantics

Every stage checkpoints at multiple granularities:
- **Source level** — which sources are complete
- **Stream offset** — byte position in the remote dump
- **Shard level** — which output shards are finalised
- **Batch level** — last committed batch index
- **Dedup index** — SimHash map persisted every 50,000 documents

On restart, `main_data_pipeline.py` automatically resumes from the last saved checkpoint. No data is re-processed.

---

## Adding a new source

1. Create `data_pipeline/extractors/mysource_extractor.py`
2. Inherit from `BaseExtractor`, implement `stream()` and `build()`
3. Register in `pipeline_orchestrator.py` `_EXTRACTOR_REGISTRY`
4. Add source config block to `runtime_config.json`
5. Create output directory: `datasets/mysource/`

No other changes required.

---

## Export formats

| Format | Use case |
|---|---|
| `plain_text` | Tokenizer training (one doc per line) |
| `jsonl_text_only` | Pre-training `{"text": "..."}` |
| `jsonl_full` | Full metadata for RAG / retrieval |
| `jsonl_sft` | Instruction tuning `{"prompt":…,"response":…}` |
