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
