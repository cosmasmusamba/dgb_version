# DGB AI Platform — System Architecture
**Version:** 3.0.0 | **Status:** Ground-truth aligned

## Philosophy
- From scratch — no HuggingFace, no SentencePiece, no LangChain
- PyTorch for tensors only, everything else custom
- Config-driven via runtime_config.json
- Datetime-prefixed artifacts (YYYYMMDDHHmmss_)
- Modular: tokenizer ↔ transformer ↔ trainer ↔ inference ↔ API never mixed

## Bug Fixes in v3.0.0
- B1: max_seq_len=512 (was vocab_size=8000) in dgb_tokenizer._finalize()
- B2: removed double checkpoint load from trainer.py
- B3: api inference.py uses PathResolver.tokenizer_dir() not cfg.tokenizer_dir()
- B4: learning_rate read from config (was hardcoded 1e-3)
- T4: seed applied to torch/random/numpy in training_loop.py
- T6: num_workers=0 default until multi-file corpus

## Status Tags
[DONE] = implemented | [PARTIAL] = partially built | [TODO] = roadmap

## Module Index
configs/          [DONE]  — Pydantic config, constants, JSON loader
modules/          [DONE]  — logging, utils, streaming, memory, metrics
tokenizer/        [DONE]  — byte-level BPE from scratch
transformer/      [DONE]  — encoder-decoder, pre-LN, sinusoidal PE
trainer/          [DONE]  — training loop, checkpoint manager, dataset loader
inference/        [DONE]  — beam search, top-p sampling, streaming generator
api_manager/      [DONE]  — FastAPI, JWT auth, SSE/WS streaming, admin routes
integrations/     [DONE]  — web search grounding (Brave/Tavily/SerpAPI)
security/         [DONE]  — sliding-window rate limiter with audit trail
monitoring/       [DONE]  — Prometheus metrics, request timing middleware
data_pipeline/    [TODO]  — multi-source extraction, dedup, PII removal
ui/               [TODO]  — React admin dashboard + research portal

## Data Flow
raw txt → dataset_clean → tokenizer_train → model_train → inference API → UI
