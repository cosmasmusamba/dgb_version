# DGB AI — ROADMAP (planner: Claude · implementer: Copilot · owner/runner: Xosmas)

Last updated 2026-11-11: T0.1 done; batch2 ships T0.2 + T0.3. Reviewed against `0d20019` / `6ff5720`.

**Status board**
- T0.1 leak proof ........ DONE (Copilot run: loss 1.647 -> 0.024 on random tokens in 300 steps;
  independent run: 1.74 -> 0.003, and editing later src tokens moved earlier logits by 0.64)
- T0.2 decoder-only model  SHIPPED in batch2 (`DGBGPT`, `build_model`, `_forward_loss`, checkpoint architecture guard)
- T0.3 sanity suite ....... SHIPPED in batch2 (`tests/unit/test_gpt_model.py`)
- Verified end to end (Claude's sandbox, real PyTorch, repo tokenizer + `StreamingTextDataset` + `_eval_pass`):
  2M-param DGBGPT, 700 steps on 3,000 lines of real English: loss 9.02 (chance) -> train 3.29,
  HELD-OUT 3.71 (perplexity 41). The old leaked setup reported 0.0002, which no real model can reach.
  Rerun it yourself: `python batch2\e2e_smoke.py . %TEMP%\dgb_e2e 300`
- T0.4 eval harness ....... next (note: `_eval_pass` returns 0.0 when it produces no batches,
  e.g. a validation file with only short lines; it must return None or raise instead)
- T1.1 tokenizer .......... PRIORITY: `decode()` drops all whitespace ("Thefunctionreturns...") because the
  tokenizer has no space marker, and non-ASCII/emoji become UNK (3 of 13 tokens in a test string).
  Users would see unspaced text from the API, and code indentation cannot round-trip.
- Known gaps after batch2: `finetune/` (LoRA) still feeds tgt==src, the same leak, so it is
  not usable until ported in Phase 3; beam search is encoder-decoder only (decoder-only falls
  back to greedy); no KV cache yet (T2.3); old leak-trained checkpoints are refused by the
  architecture guard on purpose.
- Note on a Copilot remark: `.view()` failing in the T0.1 test came from non-contiguous
  test tensors (`ids[:, 1:]`), not from the training loop; the real loader builds contiguous tensors.

## 0. Where the project really stands (from reading the repo)

**Strong, keep it**
- Data pipeline: `datasets` run reported 1,106,534,538 raw lines → 403,641,191 cleaned lines
  from 71,533 files in ~1.9 h (`checkpoints/logs/dgb1/training/20260526140632_pipeline_state.json`),
  with dedup, toxicity, quality scoring, sharding, resumable state.
- Config system, structured logging, checkpoint lineage, LoRA adapter, FastAPI service,
  and (commit `0d20019`) real auth hardening: no default accounts, no fallback JWT secret,
  self-registration closed, auth on inference routes, hashed-token rate-limit keys.

**Broken, fix first — the model has never actually learned language**
1. *Label leakage in the training objective.* `trainer/core/dataset_loader.py` yields
   `src=ids[:-1]`, `tgt=ids[1:]` (correct for a decoder-only LM). But
   `trainer/core/training_loop.py` feeds BOTH into the encoder-decoder
   `DGBTransformer` and scores `model(src, tgt)` against the same `tgt`. So the decoder's
   input at position t is exactly the label for position t (the causal mask hides only
   *future* positions), and the encoder sees the whole sequence bidirectionally. The model
   can copy instead of predict. Evidence: the `model_training` stage finished in 8.4 s with
   `best_loss 0.0002`, which a real language model cannot reach.
   (Diagnosed by reading the code; T0.1 proves it by running it.)
2. *Fine-tune never really ran:* 3 epochs / 9 steps on a tiny file, loss rose
   0.177 → 0.197 → 0.228, lineage `checkpoint_path: null`.
3. *Wrong shape for an LLM:* encoder-decoder, ~9M parameters (estimate from config; the
   model logs its exact count), sinusoidal positions, vocab 8,000, non-byte-level BPE
   (anything unseen becomes `<UNK>`), and every training example is ONE cleaned line
   (`dataset_loader.py` encodes line by line), so the model never sees long context.
4. *Tokenizer training took ~39 h* (`train_tokenizer.duration_sec` = 140,970).
5. `trainer/eval/` is empty: there is no perplexity or sample-based evaluation.

**Realistic target (so we measure the right thing).** A from-scratch DGB model will be a
small LM (tens to low hundreds of millions of parameters). Goal: coherent text, measurable
perplexity, then narrow fine-tuned skills. It will not match frontier models; do not
claim it does. Every phase ends with numbers in `docs/WORKLOG.md`.

## 1. Working agreement
- Claude plans and ships each task as a patch zip (`dgb_batchN.zip`): extract in the project
  root, run `patch_batchN.py --dry-run` (checks the environment, writes nothing), then apply,
  then run the verify commands printed by the script. Overwritten files are backed up.
- Copilot is optional: use it for follow-up edits inside the repo. If used, ONE task per
  session on a branch `task/<id>-<name>`, and it logs to `docs/WORKLOG.md`.
- Xosmas commits, runs cloud jobs and pastes outputs. Claude reads the pushed code on GitHub
  (repo is public, read-only) and reviews real diffs, then ships the next batch.
- Local runs: tiny configs only (CPU, no GPU, ~3 GB RAM free). Real training: cloud GPU.

---

## PHASE 0 — Make training honest (local, CPU, tiny configs)

### T0.1 Prove the leak (tests only; do NOT fix anything yet) — shipped in batch1
Adds `tests/unit/test_training_objective.py` (two xfail tests, details below):
- Build a tiny `DGBTransformer` (vocab 50, d_model 32, 2 heads, 1 encoder + 1 decoder layer, dropout 0).
- Data: random token ids in `[6, 50)` (above the special tokens), length 32, paired exactly
  like `StreamingTextDataset` (`src=ids[:-1]`, `tgt=ids[1:]`). Random tokens are unpredictable,
  so a correct language model's loss must stay near `ln(44) ≈ 3.78`.
- Train 300 steps with the SAME forward/loss calls as `run_training_loop`
  (`model(src, tgt, src_mask, tgt_mask)`, `CrossEntropyLoss(ignore_index=pad)` vs `tgt`).
- Assert final loss > 0.8 × ln(44). Mark `@pytest.mark.xfail(strict=True, reason="label leakage — fixed in T0.2")`.
**Accept:** test runs <30 s on CPU; WORKLOG shows the printed loss at steps 0/100/200/300.
If loss collapses toward 0 the leak is confirmed; if it does NOT, report that honestly and stop.

### T0.2 Decoder-only model + correct objective
- New `transformer/core/gpt_model.py`: `DGBGPT` — token embedding tied to output head,
  pre-norm blocks with RMSNorm, rotary position embeddings (RoPE), SwiGLU MLP, causal
  attention via `torch.nn.functional.scaled_dot_product_attention(is_causal=True)`.
  `forward(ids) -> logits`. Config: `n_layers, n_heads, d_model, d_ff, max_seq_len, rope_theta, dropout`.
- `configs/loader.py`: add `architecture: "decoder_only" | "encoder_decoder"` (default `decoder_only`).
  Keep the old encoder-decoder code, mark legacy. `trainer/core/trainer.py` builds by config.
- `training_loop.py` / `_eval_pass`: for decoder-only use `logits = model(src)`; loss vs `tgt`.
- Checkpoints store `architecture` + model config; loading validates them.
**Accept:** T0.1 test now passes (remove xfail); a single-batch overfit test reaches loss < 0.1
within 300 CPU steps; `pytest tests/unit -q` green; param count logged.

### T0.3 Model sanity suite (`tests/unit/test_gpt_sanity.py`)
- Causality: changing token at position t+k leaves logits at positions ≤ t unchanged (atol 1e-5).
- Padding invariance; seeded determinism; save→load gives identical logits; no NaN/inf grads.

### T0.4 Evaluation harness (`trainer/eval/`)
- Validation perplexity over fixed held-out windows; `eval/prompts.jsonl` (20 fixed prompts)
  generated with seeded greedy + top-p; results to `checkpoints/logs/<model>/eval/<ts>_eval.json`;
  hook into checkpoint saving. **Accept:** runs on a tiny model in <60 s; JSON schema documented.

## PHASE 1 — Tokenizer and data for real training

### T1.1 Tokenizer: profile, then rebuild
Profile `main_train_tokenizer.py` / `tokenizer/core` on a 50 MB sample and report where time
goes. Then: byte-level BPE (no `<UNK>` possible), configurable vocab (default 32,000),
pre-tokenize once into a unique-word frequency table and run merges with incremental pair-count
updates. **Accept:** exact encode→decode round-trip on a varied UTF-8 test set (emoji, code,
Luganda/other languages, newlines); zero UNK; deterministic; report speed vs the old tokenizer
on the same sample (target ≥10×).

### T1.2 Token shards + packed loader
`main_build_token_shards.py`: stream cleaned text → tokenize → documents joined with EOS →
`uint16` (or `uint32` if vocab > 65535) `.bin` shards + `index.json` (token counts, sha256,
tokenizer hash); sequential writes, resumable, train/val split by shard.
`MemmapTokenDataset` yields random fixed-length windows (e.g. 1024 tokens).
**Accept:** decode of a shard slice matches source text; throughput and total token count reported.

### T1.3 Corpus audit (decision gate)
Token counts by source, dedup/PII stats, 200 random documents saved for human review.
Rule of thumb for compute-optimal training is ~20 tokens per parameter (small models are
often trained on far more); the token count picks the Phase 2 model size.

## PHASE 2 — Real pretraining on a cloud GPU (Kaggle/Colab first)
- T2.1 Config ladder (`dgb_10m` debug, `dgb_50m`, `dgb_125m`), bf16/fp16 autocast, grad accumulation,
  cosine LR + warmup, AdamW (no decay on norms/bias), grad clip, optional `torch.compile` and activation
  checkpointing; atomic, fully resumable checkpoints (model, optimizer, scheduler, RNG, data position);
  `scripts/cloud_train.py` that survives session limits; tokens/sec logging.
- T2.2 First run: 10M model; loss must fall smoothly, val perplexity improves, samples are fluent.
  Do not scale until this curve is healthy. Then 50M, then 125M.
- T2.3 KV-cache generation + weight-only int8 for CPU serving; cached == uncached greedy output.

## PHASE 3 — Instruction tuning
Port `finetune/core/lora_adapter.py` to the decoder-only model; chat/SFT data format with
license audit; before/after eval on the T0.4 prompts; adapter merge + load in the API.

## PHASE 4 — Serving and product
DB-backed users and revocable refresh tokens (current store is in-memory with one bootstrap admin);
rate limiter: trusted-proxy handling for `X-Forwarded-For`/`CF-Connecting-IP`, atomic Redis check,
explicit fail-open/closed policy; reject placeholder secrets from `.env.example` at startup;
Dockerfile, cloud deployment, streaming OpenAI-compatible API, UI portal.
