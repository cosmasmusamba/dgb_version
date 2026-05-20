"""
transformer/utils/model_helpers.py
====================================
Shared utilities for model initialisation, device selection,
weight initialisation, and checkpoint management.

FIX B5 (v3.0.0): removed dead `import os as _os` and duplicate
docstring inside save_checkpoint(). No logic changes.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

from configs.constants import DeviceType
from modules.utils.error_handler import ModelInitError, CheckpointError
from modules.utils.safe_writer import write_checksum, verify_checksum

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def resolve_device(preference: str = "auto") -> "torch.device":
    """Resolve the best available compute device."""
    if not _HAS_TORCH:
        raise ModelInitError("PyTorch is not installed")
    pref = preference.lower()
    if pref == DeviceType.CUDA and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == DeviceType.MPS:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    if pref == DeviceType.AUTO:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def count_parameters(model: "nn.Module") -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    if not _HAS_TORCH:
        return 0, 0
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def log_model_info(model: "nn.Module", name: str = "model") -> None:
    total, trainable = count_parameters(model)
    logger.info(
        "%s — total params: %s  trainable: %s  (%.1fM)",
        name, _fmt_params(total), _fmt_params(trainable), total / 1e6,
    )


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def init_weights(module: "nn.Module") -> None:
    """
    Xavier-uniform for linear layers, normal for embeddings, ones/zeros for LayerNorm.
    Applied recursively via `model.apply(init_weights)`.
    """
    if not _HAS_TORCH:
        return
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def save_checkpoint(
    path: Path,
    model: "nn.Module",
    optimizer: Optional[object] = None,
    epoch: int = 0,
    step: int = 0,
    loss: float = 0.0,
    extra: Optional[dict] = None,
    ctx=None,
) -> Path:
    """
    Save a training checkpoint atomically with a SHA-256 sidecar.

    Parameters
    ----------
    path:      Destination .pt file (or directory if ctx provided).
    model:     The nn.Module to save.
    optimizer: Optional optimizer state dict.
    epoch:     Epoch number.
    step:      Global optimizer step.
    loss:      Current loss value.
    extra:     Additional JSON-serialisable metadata.
    ctx:       RunContext — if given and path is a dir, generates prefixed filename.
    """
    if not _HAS_TORCH:
        raise CheckpointError("PyTorch is not installed")

    # FIX B5: removed dead `import os as _os` that was here
    if ctx is not None and Path(path).suffix == "":
        Path(path).mkdir(parents=True, exist_ok=True)
        path = Path(path) / ctx.checkpoint_name(epoch, loss)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch":           epoch,
        "step":            step,
        "loss":            loss,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        **(extra or {}),
    }

    tmp = path.with_suffix(".tmp.pt")
    torch.save(payload, tmp)
    tmp.replace(path)

    write_checksum(path)
    logger.info("Checkpoint saved → %s  (epoch=%d  loss=%.4f)", path.name, epoch, loss)
    return path


def load_checkpoint(
    path: Path,
    model: "nn.Module",
    optimizer: Optional[object] = None,
    *,
    strict: bool = True,
    verify: bool = True,
    device: Optional["torch.device"] = None,
) -> dict:
    """
    Load a checkpoint, verifying SHA-256 sidecar if present.
    Returns the full payload dict.
    """
    if not _HAS_TORCH:
        raise CheckpointError("PyTorch is not installed")

    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"Checkpoint not found: {path}")

    if verify:
        try:
            verify_checksum(path)
        except Exception as exc:
            raise CheckpointError(f"Checksum failed for {path.name}: {exc}") from exc

    map_loc = device or torch.device("cpu")
    payload = torch.load(path, map_location=map_loc, weights_only=False)

    model.load_state_dict(payload["model_state"], strict=strict)
    if optimizer and payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])

    logger.info(
        "Checkpoint loaded from %s  (epoch=%d  loss=%.4f)",
        path.name, payload.get("epoch", 0), payload.get("loss", 0.0),
    )
    return payload


def latest_checkpoint(directory: Path) -> Optional[Path]:
    """Return the most recent epoch .pt file (best_model excluded)."""
    from modules.utils.file_handler import list_files
    all_pts    = sorted(list_files(directory, glob="*.pt"))
    epoch_pts  = [p for p in all_pts if "best_model" not in p.name]
    return epoch_pts[-1] if epoch_pts else None
