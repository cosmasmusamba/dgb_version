# DGB AI — instructions for GitHub Copilot

You are the implementer on DGB AI, a from-scratch LLM framework (PyTorch for tensors only:
no HuggingFace, no SentencePiece, no LangChain; `torch.nn.functional` incl.
`scaled_dot_product_attention` is allowed). The owner is Xosmas. Planning and review are done
by Claude in a separate chat; Claude writes the task specs in `docs/ROADMAP.md`.

## How we work
1. Do ONE task from `docs/ROADMAP.md` per session (the one the owner names). Do not start the next.
2. Work on a branch named `task/<id>-<short-name>` (e.g. `task/T0.1-leakage-test`). Never commit to `main` directly.
3. Stay in scope. If you see another bug or improvement, add it under "Open questions" in
   `docs/WORKLOG.md` instead of fixing it.
4. A task is done only when every acceptance criterion in its spec is met AND you have
   actually run the commands and pasted their real output into `docs/WORKLOG.md`.
   Never write "should work" or "tests pass" without having run them. If you cannot run
   something (no GPU, too slow), say so explicitly.
5. After finishing, append an entry to `docs/WORKLOG.md` (template inside that file),
   commit, and tell the owner the commit hash so Claude can review the real code on GitHub.

## Environment (hard limits for local runs)
- Windows 10, Python 3.14 (create/use a venv; if a dependency lacks 3.14 wheels, say so and
  suggest 3.12 rather than hacking around it), Intel i3-4160 (2 cores / 4 threads), 8 GB RAM
  with ~3 GB usually free, **no usable GPU**, HDD storage.
- Local runs must therefore use tiny configs (`d_model<=64`, `n_layers<=2`, short sequences,
  a few hundred steps). Real training happens on cloud GPUs (Kaggle/Colab) in Phase 2.
- `datasets/` and model weights are git-ignored. Never commit data, weights or secrets.

## Code rules
- Keep the existing layout (`transformer/`, `tokenizer/`, `trainer/`, `inference/`, `finetune/`,
  `api_manager/`, `data_pipeline/`, `configs/`). Read config via `configs/loader.py`; do not
  hard-code paths or hyper-parameters.
- Type hints, docstrings, no bare `except`. Use the existing logging setup in `modules/`.
- Every behavioural change needs a test in `tests/unit/` (fast, CPU-only, <30 s each).
  Run `pytest tests/unit -q` before every commit and paste the summary line in the WORKLOG.
- Keep checkpoints resumable and backward compatible, or bump a version field and write a
  migration note. Datetime-prefixed artifact naming (`YYYYMMDDHHmmss_`) stays.
- Prefer small, reviewable diffs. Do not reformat files you are not changing.
- Do not add new third-party dependencies without asking; list the reason in the WORKLOG.
