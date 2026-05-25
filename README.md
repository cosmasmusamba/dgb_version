# DGB AI Platform — v3.0.0

Fully custom AI framework built from scratch. No HuggingFace, no SentencePiece,
no LangChain. Every component — tokenizer, transformer, trainer, inference engine,
API, streaming — hand-built in PyTorch.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set secret key
export DGB_SECRET_KEY=your-strong-secret

# 3. Add Wikipedia dump files
# Place .txt files in: datasets/dgb1/wk_raw/

# 4. Run full pipeline (clean → tokenize → train)
python main_pipeline.py

# 5. Start API server
python main_api_server.py
# → http://localhost:8000
# → http://localhost:8000/docs

# 6. Run tests
pytest tests/
```

## Individual stages
```bash
python main_dataset_clean.py          # clean raw text
python main_train_tokenizer.py        # train BPE tokenizer
python model_trainer.py               # pre-train transformer

python main_pipeline.py --status      # show pipeline state
python main_pipeline.py --force model_training  # re-run a stage
```

## Environment variables
```
DGB_SECRET_KEY      JWT signing secret (required in production)
BRAVE_API_KEY       Web search grounding (optional)
DGB_RAW_DIR         Override dataset raw path
DGB_MODELS_DIR      Override model checkpoint path
```

## Architecture
See docs/architecture.md for full module index and data flow.

## Data pipeline

The DGB data pipeline ingests and preprocesses large-scale text corpora from multiple sources simultaneously.

```bash
# See all available options
python main_data_pipeline.py --help

# Dry run (validate config, list sources)
python main_data_pipeline.py --dry-run

# Run the pipeline (processes enabled sources from runtime_config.json)
python main_data_pipeline.py

# Export processed data to training format
python main_data_pipeline.py --export --format plain_text
```

See `docs/data_pipeline.md` for full documentation.

### Supported sources
| Source | Status | Domain |
|---|---|---|
| Wikipedia (local wk_*.txt) | Enabled by default | Encyclopedic |
| Wikipedia (online dumps) | Config: dump_urls | Encyclopedic |
| StackExchange | Config: enabled:true | Q&A |
| arXiv | Config: enabled:true | Academic |
| Project Gutenberg | Config: enabled:true | Books |
| Common Crawl | Config: enabled:true | Web |
| GitHub | Config: enabled:true | Code |

### Pipeline stages
1. Text normalisation (Unicode, whitespace, HTML)
2. Language detection and filtering (fasttext)
3. Toxicity and unsafe-content filtering
4. Quality scoring (12 heuristic signals)
5. Exact + near-duplicate deduplication (SHA-256 + SimHash)
6. Metadata enrichment (topics, readability, structure)
7. Shard writing (512 MB max, atomic rotation, SHA-256 verified)
