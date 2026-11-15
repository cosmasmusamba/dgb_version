# DGB WORKLOG

Copilot appends one entry per finished task (newest at the bottom). Claude reads this file on
GitHub to review. Paste REAL command output, not summaries.

## Entry template

```
### <task id> — <title>  (<date>)
- Branch / commit: <branch> @ <hash>
- Status: DONE | PARTIAL | BLOCKED
- What changed: <files and one line each>
- Commands run and their real output:
  <paste>
- Acceptance criteria: <each criterion → met / not met, with evidence>
- Open questions / things I noticed but did not change:
  - <...>
```

## Entries

### T0.1 — Prove training label leakage (2026-11-10)
- Branch / commit: `task/T0.1-leakage-test` @ `a57a6a5c5c777fa03372e4a37fb6690f0658169d`
- Status: DONE
- What changed:
  - `tests/unit/test_training_objective.py`: added a tiny CPU test that trains on fresh random token batches using the existing encoder-decoder objective and strictly xfails when loss falls below the unpredictability threshold.
- Commands run and their real output:
  - `C:\Users\Cosmas\AppData\Local\Programs\Python\Python313\python.exe -m pytest -rxX -s tests/unit/test_training_objective.py`
    ```
    collected 1 item
    step 0: loss=1.647272
    step 100: loss=0.127895
    step 200: loss=0.045565
    step 300: loss=0.024246
    XFAIL (label leakage — fixed in T0.2)
    ============================= 1 xfailed in 9.66s ==============================
    ```
  - `C:\Users\Cosmas\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests/unit -q`
    ```
    collected 35 items
    ======================= 34 passed, 1 xfailed in 11.14s ========================
    ```
- Acceptance criteria:
  - Tiny `DGBTransformer` configured as specified; data uses token IDs in `[6, 50)` and the `src=ids[:-1]`, `tgt=ids[1:]` shift: met.
  - Uses the current model forward path, padding masks, and cross-entropy objective: met. `reshape` is used for flattening because logits are non-contiguous for this input and `.view()` raises; the actual training loop's `.view()` was not changed.
  - Strict xfail with reason `label leakage — fixed in T0.2`; final-loss assertion is greater than `0.8 * ln(44)`: met. The measured final loss `0.024246` is below the threshold (approximately `3.024`), so the expected failure confirms leakage.
  - Test under 30 seconds on CPU: met; targeted run took 9.66 seconds.
  - Unit suite run: met; 34 passed and the expected T0.1 xfail.
- Open questions / things I noticed but did not change:
  - `training_loop.py` flattens logits with `.view()`, which raised on the non-contiguous logits returned in this test setup. Left unchanged because T0.1 is test-only; investigate in a later task.
  - The workspace venv has no pytest and the available Python interpreters are 3.13.7, not the documented 3.14. Used the installed system Python 3.13.7 with pytest and PyTorch; no dependencies were installed.

### T0.2 / T0.3 — Decoder-only model, leak-free objective, tests (2026-11-11) [Claude]
- Branch / commit: `task/T0.2-batch2` @ `7b43954`, merged to `main` @ `9c253ac`
- Status: DONE
- What changed: `transformer/core/gpt_model.py` (DGBGPT), `build_model` factory and architecture-aware checkpoints,
  `_forward_loss` in the training loop, decoder-only paths in the inference engine, `tests/unit/test_gpt_model.py` (10 tests).
- Verification (owner's PC): `44 passed, 1 xfailed`; end-to-end 2M-param run, 300 steps: loss 9.04 -> 4.04, held-out 4.31.
- Open: LoRA fine-tuning still leaks; beam search is encoder-decoder only; no KV cache.

### T1.1 — Byte-level BPE tokenizer and eval tools (2026-11-12) [Claude]
- Branch / commit: `task/T1.1-byte-tokenizer` @ `5bde4a1`, merged to `main` @ `ab1b67a`
- Status: DONE (vocab 8000; 32k decision deferred to cloud-pretraining planning)
- What changed: `tokenizer/byte_bpe.py`, `tools/train_byte_bpe.py` (samples across all files), `tools/eval_model.py`,
  `tests/unit/test_byte_bpe.py`; `DGBTokenizer.from_pretrained` returns the byte tokenizer when `byte_bpe.json` exists.
- Verification (owner's PC): trained on 2,002,690 lines from all 71,535 files (read 853 s, train 47 s):
  3.89 bytes/token, 0 exact-round-trip failures on 2,010 lines. Unit tests: `70 passed, 1 xfailed` at that point.
- Open: speed comparison with the old tokenizer; incremental decoder for streaming multi-byte characters;
  `main_train_tokenizer.py` is broken and unused.

### Admin role claim fix (2026-11-13) [Claude]
- Branch / commit: `backend/fix-admin-role-claim` @ `fe8447a`, merged to `main` @ `208f4bf`
- Status: DONE
- Cause: `str(UserRole.ADMIN)` is `'UserRole.ADMIN'`; the token stored that text, so `require_role` always answered 403.
- Fix: `normalize_role` (plain values, fails closed, reads legacy tokens without escalation); tests set their own
  `DGB_SECRET_KEY`; login log prints the plain role.
- Verification: owner's PC `59 passed, 1 xfailed`, then `85 passed, 1 xfailed` after the tokenizer merge. Copilot's browser check:
  `/auth/me` role `admin`, `/v1/admin/health` 200 for admin, 403 for a user, 401 for a guest; frontend compares `"admin"` exactly.

### Database foundation (2026-11-15) [Claude patch; Copilot local verification]
- Branch / commit: `backend/db-foundation` @ `a89fb217de96018f3d29624138bbc29dc63251f9`
- Status: DONE — implementation and local verification complete; owner review/merge pending
- What changed: `database/` package (SQLAlchemy 2 + Alembic migrations, SQLite fallback, PostgreSQL (Docker, `docker-compose.db.yml`, host port 5434) via `DGB_DATABASE_URL`);
  `users` and `audit_log` tables; accounts replace the in-memory user store; login, refresh and every protected request read
  the account (role, active) from the database, so a demoted or disabled user loses access immediately; bootstrap admin is
  seeded once at start-up and never overwritten; unknown-user logins cost the same time as real ones; login/refresh events are
  audited (hashed client key, no passwords); `tools/manage_users.py` (list/create/set-role/disable/enable/set-password/audit);
  last active admin cannot be demoted or disabled; tests get a fresh database each.
- Commands run and their real output (Python 3.13.7; local Docker PostgreSQL 17.11; credentials and tokens omitted):
  - `Get-ChildItem database,tools,tests,api_manager -Recurse -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }`
    - Exit code 0; completed with no output.
  - `python -m pytest tests\unit -q -p no:cacheprovider` (SQLite default):
    ```
    ======================= 102 passed, 1 xfailed in 29.79s =======================
    ```
  - `python -m pytest tests\unit -q -p no:cacheprovider` with `DGB_TEST_DATABASE_URL` pointing to the isolated `dgb_test` database:
    ```
    ======================= 102 passed, 1 xfailed in 24.10s =======================
    ```
  - `docker exec dgb-postgres psql -U dgb -d dgb -Atc 'SHOW server_version'`:
    ```
    17.11 (Debian 17.11-1.pgdg13+2)
    ```
    Container health was `healthy`, bound only to `127.0.0.1:5434`.
  - `python tools\manage_users.py list` against the fresh `dgb` database:
    ```
    database: 127.0.0.1:5434/dgb
    username                role      active  email                       last login
    0 account(s)
    ```
  - Local API auth smoke check against PostgreSQL:
    ```
    POST /auth/login: HTTP 200; token returned (not printed)
    GET /auth/me: role=admin
    GET /v1/admin/health: HTTP 200; status=ok
    ```
  - `python tools\manage_users.py audit` (actor/target values omitted):
    ```
    login_success
    bootstrap_admin_created
    ```
- Acceptance criteria:
  - Docker `postgres:17` is healthy on loopback port 5434; verified server version `17.11`: met.
  - Python files compile and the full unit suite passes with the SQLite default: met; 102 passed, 1 expected xfail.
  - Full unit suite runs against isolated PostgreSQL `dgb_test`: met; 102 passed, 1 expected xfail.
  - Fresh PostgreSQL `dgb` DB starts empty, seeds the bootstrap admin on API startup, accepts login, returns canonical admin role and allows admin health: met.
  - Bootstrap and successful-login audit events are present: met.
- Open: refresh-token rotation/revocation; admin HTTP endpoints for users and audit (next branch). Production still needs provider-managed credentials, private/TLS database connectivity, controlled serialized migrations, and tested backup/restore; see `docs/DATABASE.md`.
