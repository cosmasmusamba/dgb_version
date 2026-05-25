"""
modules/utils/run_context.py
=============================
Centralized run identity and datetime-prefixed filename generation.

Every pipeline run gets a single RunContext created at startup.
All file-creating components (trainer, cleaner, metrics, checkpoints)
call `get_run_context()` to obtain the shared run_id so every artefact
from one run carries the same YYYYMMDDHHMMSS prefix.

Sorting files by name therefore automatically gives chronological order
across runs, and groups all artefacts from the same run together.

Naming contract
---------------
<YYYYMMDDHHMMSS>_<descriptive_name>.<ext>

Examples
--------
20260510202040_training.log
20260510202040_pipeline_state.json
20260510202040_dgb1_cleaned_wk_0.txt
20260510202040_epoch_001_loss_6.3665.pt
20260510202040_metrics_steps.json
20260510202040_bpe_merges.json
20260510202040_cleaning_summary.json
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------

class RunContext:
    """
    Holds the identity of a single pipeline run.

    Parameters
    ----------
    run_id:
        YYYYMMDDHHMMSS string.  Generated from UTC wall-clock time at
        construction so runs started seconds apart never collide.
    model_id:
        e.g. "dgb1"
    """

    def __init__(self, run_id: Optional[str] = None, model_id: str = "dgb1") -> None:
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.run_id   = run_id
        self.model_id = model_id

    # ------------------------------------------------------------------
    # Filename helpers

    def prefix(self, name: str) -> str:
        """
        Return  '<run_id>_<name>'

        Examples
        --------
        ctx.prefix("training.log")            → "20260510202040_training.log"
        ctx.prefix("cleaning_summary.json")   → "20260510202040_cleaning_summary.json"
        """
        return f"{self.run_id}_{name}"

    def prefixed(self, directory: Path, name: str) -> Path:
        """Return Path(directory / prefix(name))."""
        return Path(directory) / self.prefix(name)

    def checkpoint_name(self, epoch: int, loss: float) -> str:
        """
        20260510202040_epoch_001_loss_6.3665.pt
        """
        return self.prefix(f"epoch_{epoch:03d}_loss_{loss:.4f}.pt")

    def cleaned_name(self, source_stem: str, model_id: Optional[str] = None) -> str:
        """
        Build the cleaned-file name preserving the source stem.

        source_stem is the raw filename without extension, e.g. "wk_0"
        (wk = Wikipedia dump folder abbreviation).

        Result: 20260510202040_dgb1_cleaned_wk_0.txt
        """
        mid = model_id or self.model_id
        return self.prefix(f"{mid}_cleaned_{source_stem}.txt")

    def cleaned_file_name(self, raw_filename: str) -> str:
        """
        Convenience method - takes full raw filename, returns cleaned filename.
        
        Example: raw_filename = "wk_0.txt" -> returns "20260525135015_dgb1_cleaned_wk_0.txt"
        """
        from pathlib import Path
        source_stem = Path(raw_filename).stem
        return self.cleaned_name(source_stem)

    def training_log_name(self) -> str:
        return self.prefix("training.log")

    def progress_log_name(self) -> str:
        return self.prefix("progress.txt")

    def pipeline_state_name(self) -> str:
        return self.prefix("pipeline_state.json")

    def pipeline_log_name(self) -> str:
        return self.prefix("pipeline_log.jsonl")

    def metrics_steps_name(self) -> str:
        return self.prefix("metrics_steps.json")

    def metrics_epochs_name(self) -> str:
        return self.prefix("metrics_epochs.json")

    def vocab_name(self) -> str:
        return self.prefix("vocabulary.json")

    def vocab_meta_name(self) -> str:
        return self.prefix("vocab_meta.json")

    def bpe_merges_name(self) -> str:
        return self.prefix("bpe_merges.json")

    def cleaning_summary_name(self) -> str:
        return self.prefix("cleaning_summary.json")

    def loss_curve_name(self) -> str:
        return self.prefix("loss_curve.png")

    def best_model_name(self) -> str:
        return self.prefix("best_model.pt")

    # ------------------------------------------------------------------
    # Class methods for finding existing runs

    @classmethod
    def latest_for_model(cls, log_dir: Path, model_id: str) -> Optional["RunContext"]:
        """
        Find the most recent run for this model by scanning log_dir for
        pipeline_state_<run_id>.json files and return a RunContext with that run_id.
        
        Parameters
        ----------
        log_dir: Directory containing run artefacts (e.g., res.logs_dir())
        model_id: Model identifier (e.g., "dgb1")
        
        Returns
        -------
        RunContext with the latest run_id, or None if no previous runs found.
        """
        if not log_dir.exists():
            return None
        
        # Look for pipeline_state_<run_id>.json files
        import re
        pattern = re.compile(r'^(\d{14})_pipeline_state\.json$')
        
        latest_ts = None
        latest_run_id = None
        
        for file in log_dir.glob("*_pipeline_state.json"):
            match = pattern.match(file.name)
            if match:
                run_id = match.group(1)
                if latest_ts is None or run_id > latest_ts:
                    latest_ts = run_id
                    latest_run_id = run_id
        
        if latest_run_id:
            logger.info(f"Found latest run: {latest_run_id}")
            return cls(run_id=latest_run_id, model_id=model_id)
        
        return None

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"RunContext(run_id={self.run_id!r}, model_id={self.model_id!r})"


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

_lock:    threading.Lock        = threading.Lock()
_context: Optional[RunContext]  = None


def create_run_context(model_id: str = "dgb1") -> RunContext:
    """
    Create a new RunContext for this process invocation.

    Call once at the very start of main_pipeline.py / model_trainer.py /
    main_dataset_clean.py.  All subsequent calls to get_run_context()
    return the same object.
    """
    global _context
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    with _lock:
        _context = RunContext(run_id=run_id, model_id=model_id)
    logger.info(
        "RunContext created: run_id=%s  model_id=%s",
        run_id, model_id,
    )
    return _context


def get_run_context() -> RunContext:
    """
    Return the active RunContext.  Creates one with defaults if none exists
    (e.g. when a module is imported standalone in tests).
    """
    global _context
    if _context is None:
        with _lock:
            if _context is None:
                _context = RunContext(
                    run_id=datetime.now().strftime("%Y%m%d%H%M%S"),
                )
                logger.debug(
                    "RunContext auto-created: run_id=%s", _context.run_id
                )
    return _context


def reset_run_context(model_id: str = "dgb1") -> RunContext:
    """Force a fresh RunContext (used by tests and pipeline re-runs)."""
    global _context
    with _lock:
        _context = None
    return create_run_context(model_id)


# ---------------------------------------------------------------------------
# Utility: find the latest run artefact in a directory
# ---------------------------------------------------------------------------

def latest_run_id(directory: Path, glob: str = "*.pt") -> Optional[str]:
    """
    Scan `directory` for files matching `glob` that start with a
    YYYYMMDDHHMMSS prefix and return the most recent run_id, or None.

    Sorting by name = sorting by datetime because the prefix is always
    the first 14 characters.
    """
    files = sorted(directory.glob(glob))
    for f in reversed(files):
        stem = f.stem
        candidate = stem[:14]
        if len(candidate) == 14 and candidate.isdigit():
            return candidate
    return None
