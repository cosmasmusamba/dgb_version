"""
modules/utils/run_context.py
==============================
RunContext — datetime-prefixed run identifier and artifact naming.

Every pipeline run generates a unique ID of the form:
    YYYYMMDDHHmmss   (e.g. 20260518130915)

All output files — checkpoints, metrics, logs, tokenizer artifacts —
are prefixed with this ID for chronological sorting and audit trail.

RunContext also provides helper methods for constructing the specific
artifact names used across the codebase.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TIMESTAMP_FMT = "%Y%m%d%H%M%S"


class RunContext:
    """
    Immutable run identity.

    Parameters
    ----------
    model_id:   The project model identifier, e.g. "dgb1".
    run_id:     Pre-existing run ID string.  If None, a new timestamp is generated.
    """

    def __init__(self, model_id: str = "dgb1", run_id: Optional[str] = None) -> None:
        self._model_id = model_id
        self._run_id   = run_id or datetime.now().strftime(_TIMESTAMP_FMT)
        logger.info("RunContext: model_id=%s  run_id=%s", model_id, self._run_id)

    # ── Identity ───────────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def model_id(self) -> str:
        return self._model_id

    # ── Artifact naming helpers ────────────────────────────────────────

    def prefix(self, name: str) -> str:
        """Return '{run_id}_{name}'."""
        return f"{self._run_id}_{name}"

    def checkpoint_name(self, epoch: int, loss: float) -> str:
        return f"{self._run_id}_epoch_{epoch:03d}_loss_{loss:.4f}.pt"

    def best_model_name(self) -> str:
        return f"{self._run_id}_best_model.pt"

    def vocab_name(self) -> str:
        return f"{self._run_id}_vocabulary.json"

    def merges_name(self) -> str:
        return f"{self._run_id}_bpe_merges.json"

    def vocab_meta_name(self) -> str:
        return f"{self._run_id}_vocab_meta.json"

    def cleaned_file_name(self, source: str) -> str:
        return f"{self._run_id}_{self._model_id}_cleaned_{source}"

    def cleaning_summary_name(self) -> str:
        return f"{self._run_id}_cleaning_summary.json"

    def metrics_steps_name(self) -> str:
        return f"{self._run_id}_metrics_steps.json"

    def metrics_epochs_name(self) -> str:
        return f"{self._run_id}_metrics_epochs.json"

    def granular_checkpoint_name(self) -> str:
        return f"{self._run_id}_granular_checkpoint.json"

    def pipeline_state_name(self) -> str:
        return f"{self._run_id}_pipeline_state.json"

    def pipeline_log_name(self) -> str:
        return f"{self._run_id}_pipeline.jsonl"

    def training_log_name(self) -> str:
        return f"{self._run_id}_training.jsonl"

    def loss_curve_name(self) -> str:
        return f"{self._run_id}_loss_curve.png"

    def training_txt_log_name(self) -> str:
        return f"{self._run_id}_training.log"

    # ── Path helpers ───────────────────────────────────────────────────

    def tokenizer_checkpoint_dir(self, base: Path) -> Path:
        return Path(base)   # files live directly in the tokenizer dir

    def models_dir(self, base: Path) -> Path:
        return Path(base)

    def logs_dir(self, base: Path) -> Path:
        return Path(base)

    # ── Persistence ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"model_id": self._model_id, "run_id": self._run_id}

    @classmethod
    def from_dict(cls, d: dict) -> "RunContext":
        return cls(model_id=d["model_id"], run_id=d["run_id"])

    @classmethod
    def latest_for_model(cls, logs_dir: Path, model_id: str = "dgb1") -> "RunContext":
        """
        Discover the most recent run_id from datetime-prefixed files in logs_dir.
        Falls back to generating a new run_id if none found.
        """
        candidates = sorted(Path(logs_dir).glob("*_pipeline_state.json"))
        if candidates:
            run_id = candidates[-1].name.split("_")[0]
            ctx    = cls(model_id=model_id, run_id=run_id)
            logger.info("RunContext: resumed run_id=%s from %s", run_id, logs_dir)
            return ctx
        return cls(model_id=model_id)

    def __repr__(self) -> str:
        return f"RunContext(model_id={self._model_id!r}, run_id={self._run_id!r})"
