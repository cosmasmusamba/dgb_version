"""
modules/utils/metrics_logger.py
================================
Records, aggregates, and persists training / finetune metrics.

Designed to be:
  - Thread-safe (threading.Lock internally)
  - Importable by the streaming layer for live metric exposure
  - Capable of producing loss-curve plots via matplotlib (optional)

Constructor accepts either the legacy positional (save_dir, model_id)
signature OR the keyword-only `run_id` kwarg used by DGBTrainer:

    # Training / finetune (explicit run_id prefix)
    ml = MetricsLogger(save_dir=log_dir, run_id=ctx.run_id, model_id="dgb1")

    # Legacy / minimal
    ml = MetricsLogger(save_dir=log_dir)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.utils.safe_writer import atomic_write_json
from modules.utils.file_handler import ensure_dir

logger = logging.getLogger(__name__)

try:
    import matplotlib as _mpl_check
    if not hasattr(_mpl_check, "use"):
        raise ImportError("stub")
    _mpl_check.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except (ImportError, AttributeError):
    _HAS_MPL = False
    logger.debug("matplotlib not installed — plots will not be generated")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepMetric:
    epoch:    int
    step:     int
    loss:     float
    lr:       float   = 0.0
    grad_norm: float  = 0.0
    tokens_per_sec: float = 0.0
    timestamp: float  = field(default_factory=time.time)
    extras:   Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(d.pop("extras", {}))
        return d


@dataclass
class EpochMetric:
    epoch:     int
    avg_loss:  float
    val_loss:  Optional[float] = None
    duration_sec: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------

class MetricsLogger:
    """
    Collects step-level and epoch-level metrics, exposes the latest values
    for the streaming API, and persists them to disk.

    Parameters
    ----------
    save_dir:
        Directory where metric JSON files are written.
    model_id:
        Model identifier (used in log messages and plot titles).
    run_id:
        RunContext run_id — when provided, metric files are prefixed:
        <run_id>_metrics_steps.json  /  <run_id>_metrics_epochs.json.
        When absent, uses "metrics_steps.json" / "metrics_epochs.json".
    flush_every:
        Flush to disk every N log_step calls (default 1 = always).

    Usage
    -----
    ml = MetricsLogger(save_dir=log_dir, run_id=ctx.run_id)
    ml.log_step(epoch=1, step=10, loss=3.14, lr=3e-4)
    ml.log_epoch(epoch=1, avg_loss=3.14)
    ml.save()
    ml.plot()

    Generic dict log (finetune_loop compatibility):
    ml.log({"epoch": 1, "step": 10, "loss": 3.14})
    """

    def __init__(
        self,
        save_dir:    Path,
        *,
        model_id:    str = "dgb1",
        run_id:      str = "",
        flush_every: int = 1,
    ) -> None:
        self._dir        = Path(save_dir)
        self._model_id   = model_id
        self._run_id     = run_id
        self._flush_every = flush_every
        self._lock       = threading.Lock()

        self._steps:  List[StepMetric]  = []
        self._epochs: List[EpochMetric] = []
        self._latest: Dict[str, Any]    = {}

        ensure_dir(self._dir)

        # File names — prefixed with run_id if provided
        pfx = f"{run_id}_" if run_id else ""
        self._steps_path  = self._dir / f"{pfx}metrics_steps.json"
        self._epochs_path = self._dir / f"{pfx}metrics_epochs.json"

    # ------------------------------------------------------------------
    # Structured logging

    def log_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        lr: float = 0.0,
        grad_norm: float = 0.0,
        tokens_per_sec: float = 0.0,
        **extras: float,
    ) -> None:
        m = StepMetric(
            epoch=epoch, step=step, loss=loss,
            lr=lr, grad_norm=grad_norm,
            tokens_per_sec=tokens_per_sec, extras=extras,
        )
        with self._lock:
            self._steps.append(m)
            self._latest.update(m.to_dict())
        self.save()

    def log_epoch(
        self,
        epoch: int,
        avg_loss: float,
        val_loss: Optional[float] = None,
        duration_sec: float = 0.0,
    ) -> None:
        m = EpochMetric(
            epoch=epoch, avg_loss=avg_loss,
            val_loss=val_loss, duration_sec=duration_sec,
        )
        with self._lock:
            self._epochs.append(m)
            self._latest["epoch_avg_loss"] = avg_loss
            if val_loss is not None:
                self._latest["val_loss"] = val_loss
        self.save()
        if _HAS_MPL:
            self.plot()

    def log(self, data: Dict[str, Any]) -> None:
        """
        Generic dict log — used by finetune_loop and any component that
        builds its own metric dict.

        Extracts known fields and falls back to log_step for anything that
        has at least a 'loss' and 'step' key.
        """
        epoch = int(data.get("epoch", 0))
        step  = int(data.get("step", 0))
        loss  = float(data.get("loss", 0.0))
        lr    = float(data.get("lr", 0.0))
        extras = {
            k: float(v) for k, v in data.items()
            if k not in {"epoch", "step", "loss", "lr"} and isinstance(v, (int, float))
        }
        self.log_step(epoch=epoch, step=step, loss=loss, lr=lr, **extras)

    # ------------------------------------------------------------------
    # Live access

    def latest(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def step_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [m.to_dict() for m in self._steps]

    def epoch_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [m.to_dict() for m in self._epochs]

    # ------------------------------------------------------------------
    # Persistence

    def save(self) -> None:
        with self._lock:
            steps  = [m.to_dict() for m in self._steps]
            epochs = [m.to_dict() for m in self._epochs]
        atomic_write_json(self._steps_path,  steps)
        atomic_write_json(self._epochs_path, epochs)
        logger.debug("Metrics saved — %d steps, %d epochs", len(steps), len(epochs))

    def load(self) -> None:
        if self._steps_path.exists():
            raw = json.loads(self._steps_path.read_text())
            with self._lock:
                self._steps = [StepMetric(**r) for r in raw]
        if self._epochs_path.exists():
            raw = json.loads(self._epochs_path.read_text())
            with self._lock:
                self._epochs = [EpochMetric(**r) for r in raw]
        logger.info(
            "Metrics restored — %d steps, %d epochs",
            len(self._steps), len(self._epochs),
        )

    # ------------------------------------------------------------------
    # Plots

    def plot(self, filename: str = "loss_curve.png") -> Optional[Path]:
        if not _HAS_MPL:
            return None
        with self._lock:
            steps  = list(self._steps)
            epochs = list(self._epochs)
        if not steps:
            return None

        pfx      = f"{self._run_id}_" if self._run_id else ""
        out_path = self._dir / f"{pfx}{filename}"

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        xs = [m.step for m in steps]
        ys = [m.loss for m in steps]
        axes[0].plot(xs, ys, linewidth=0.8, color="#378ADD", alpha=0.85)
        axes[0].set_title("Training Loss (per step)")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

        if epochs:
            ex = [m.epoch for m in epochs]
            ey = [m.avg_loss for m in epochs]
            axes[1].plot(ex, ey, marker="o", linewidth=1.5, color="#1D9E75")
            val = [m for m in epochs if m.val_loss is not None]
            if val:
                axes[1].plot(
                    [m.epoch for m in val],
                    [m.val_loss for m in val],
                    marker="s", linewidth=1.5, color="#D85A30", label="val",
                )
                axes[1].legend()
            axes[1].set_title("Avg Loss (per epoch)")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Avg Loss")
            axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"DGB — {self._model_id}", fontsize=13)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Loss curve saved: %s", out_path)
        return out_path
