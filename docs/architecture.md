# DGB AI Platform — System Architecture
**Version:** 3.0.0 | **Status:** Ground-truth aligned

## Philosophy
- From scratch — no HuggingFace, no SentencePiece, no LangChain
- PyTorch for tensors only, everything else custom
- Config-driven via runtime_config.json
- Datetime-prefixed artifacts (YYYYMMDDHHmmss_)
- Modular: tokenizer ↔ transformer ↔ trainer ↔ inference ↔ API never mixed

## Core UI Objectives
- Unified Control Surface: Trigger all LLM operations (data pipeline, cleaning, tokenizer training, model training, finetuning, inference).
- Live Monitoring: Stream logs, metrics, and resource usage in real time.
- Historical Analysis: Query past runs, checkpoints, and metrics.
- Audit & Compliance: Evidence retention, reproducibility, and traceability baked into the UI.

##  Recommended UI Features
1. Pipeline Orchestration
Data pipeline triggers: Buttons/forms to start collection, cleaning, deduplication, toxicity filtering.

Tokenizer training: Configurable vocab size, merges, checkpoint save path.

Model training: Epochs, batch size, learning rate scheduling, device selection.

Finetuning: Dataset selection (expert.jsonl, manual finetune sets), adapter configs (LoRA).

Inference: API-driven text generation, sampling strategies (beam search, top-p).

2. Monitoring & Metrics
Live log streaming: Integrate api_manager/log_streamer.py with WebSocket/Server-Sent Events for real-time logs.

Prometheus metrics: Expose monitoring/prometheus_metrics.py to UI dashboards (CPU, RAM, throughput).

Historical metrics: Visualize JSON logs (metrics_steps.json, metrics_epochs.json) with charts (loss curves, gradient norms).

Resource monitoring: Surface modules/utils/device_monitor.py outputs (CPU %, memory, GPU availability).

3. Audit & Compliance
Checkpoint lineage: Display finetune/utils/checkpoint_lineage.py outputs in a timeline view.

Evidence retention: UI access to datasets/dgb1/cleaned/20260522133713_cleaning_summary.json.

Pipeline state: Show pipeline_state.json and granular checkpoints for reproducibility.

File traceability: Integrate datetime-prefixed filenames into sortable tables.

4. User Experience
Config-driven UI: Pull defaults from configs/runtime_config.json and configs/constants.py.

Progress tracking: Visualize modules/utils/progress_tracking.py outputs as progress bars.

Error handling: Surface structured logs from modules/utils/error_handler.py in alert banners.

Streaming inference: Use inference/engine/generator.py with WebSocket streaming for responsive text generation.

📚 Implementation Notes
Frontend stack: React (with hooks for live metrics), TailwindCSS for clean modular design.

Backend integration: FastAPI endpoints already exist (api_manager/app.py, routes/*). Extend them for UI triggers.

Historical data: Use checkpoints/logs/dgb1/training/* JSON files as sources for charts.

Security: Integrate api_manager/auth/api_security.py and security/rate_limiter.py for authenticated UI sessions.

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
