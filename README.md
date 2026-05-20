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

## Bug fixes in v3.0.0
- B1: max_seq_len correctly set to 512 (was vocab_size=8000)
- B2: removed duplicate checkpoint load in trainer.py
- B3: inference.py uses PathResolver.tokenizer_dir() correctly
- B4: learning_rate read from config (was hardcoded 1e-3)
- T4: seed 42 applied to torch/random/numpy
- T6: num_workers=0 until multi-file corpus
