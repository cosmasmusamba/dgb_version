"""
trainer/core/checkpoint_manager.py
=====================================
Manages all checkpoint I/O: per-epoch .pt files, granular batch-level
state, best-model tracking, and progress restoration on resume.
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path
from typing import Optional
logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, models_dir, log_dir, save_every_epochs=5, run_id=""):
        self._mdir  = Path(models_dir); self._mdir.mkdir(parents=True, exist_ok=True)
        self._ldir  = Path(log_dir);   self._ldir.mkdir(parents=True, exist_ok=True)
        self._freq  = save_every_epochs
        self._run_id = run_id
        self._best_loss = float("inf")

    def _prefix(self, name): return f"{self._run_id}_{name}" if self._run_id else name

    def latest_model_checkpoint(self) -> Optional[Path]:
        pts = sorted(self._mdir.glob("*.pt"))
        best = [p for p in pts if "best_model" in p.name]
        epoch = [p for p in pts if "epoch_" in p.name]
        if best: return best[-1]
        if epoch: return epoch[-1]
        return None

    def on_epoch_done(self, progress, epoch, avg_loss, model, optimizer):
        if epoch % self._freq == 0:
            from modules.utils.safe_writer import write_checksum
            from transformer.utils.model_helpers import save_checkpoint
            name = f"{self._prefix(f'epoch_{epoch:03d}_loss_{avg_loss:.4f}')}.pt"
            path = self._mdir / name
            save_checkpoint(path, model, optimizer, epoch=epoch, loss=avg_loss)
            logger.info("Epoch checkpoint → %s", name)
        if avg_loss < self._best_loss:
            self._best_loss = avg_loss
            from transformer.utils.model_helpers import save_checkpoint
            best = self._mdir / f"{self._prefix('best_model')}.pt"
            save_checkpoint(best, model, optimizer, epoch=epoch, loss=avg_loss)
            logger.info("Best model updated → loss=%.4f", avg_loss)
        self._save_granular(progress, epoch, avg_loss)

    def on_batch_done(self, progress, epoch, batch_idx, global_step, loss):
        progress.epoch       = epoch
        progress.batch_idx   = batch_idx
        progress.global_step = global_step
        if loss < getattr(progress, "best_loss", float("inf")):
            progress.best_loss = loss
        if global_step % 500 == 0:
            self._save_granular(progress, epoch, loss)

    def _save_granular(self, progress, epoch, loss):
        from modules.utils.safe_writer import atomic_write_json
        data = {
            "epoch": epoch, "global_step": getattr(progress,"global_step",0),
            "batch_idx": getattr(progress,"batch_idx",0),
            "best_loss": getattr(progress,"best_loss",float("inf")),
            "timestamp": time.time(),
        }
        path = self._ldir / f"{self._prefix('granular_checkpoint')}.json"
        atomic_write_json(path, data)

    def restore_progress(self, progress) -> bool:
        path = self._ldir / f"{self._prefix('granular_checkpoint')}.json"
        if not path.exists():
            candidates = sorted(self._ldir.glob("*granular_checkpoint.json"))
            if candidates: path = candidates[-1]
        if not path.exists(): return False
        try:
            d = json.loads(path.read_text())
            progress.epoch       = d.get("epoch", 0)
            progress.global_step = d.get("global_step", 0)
            progress.batch_idx   = d.get("batch_idx", 0)
            progress.best_loss   = d.get("best_loss", float("inf"))
            logger.info("Progress restored: epoch=%d step=%d", progress.epoch, progress.global_step)
            return True
        except Exception as exc:
            logger.warning("Cannot restore progress: %s", exc)
            return False
