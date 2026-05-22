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

from transformer.core.transformer_model import DGBTransformer

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

def load_model(checkpoint_path: str, device: torch.device, vocab_size: int = None) -> DGBTransformer:
    """
    Load a trained DGBTransformer model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: torch device to load to
        vocab_size: Optional vocab size (if not provided, tries to infer from checkpoint or config)
    
    Returns:
        Loaded model in eval mode
    """
    from configs.loader import get_config
    
    # Load config for transformer parameters
    cfg = get_config()
    transformer_cfg = cfg.transformer
    
    # Load checkpoint to inspect
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict (handle both wrapped and unwrapped formats)
    if 'model_state' in checkpoint:
        model_state = checkpoint['model_state']
        logger.info(f"Loaded training checkpoint: epoch={checkpoint.get('epoch', 'unknown')}, loss={checkpoint.get('loss', 'unknown'):.6f}")
    else:
        model_state = checkpoint
        logger.info("Loaded raw model checkpoint")
    
    # Determine vocab_size
    if vocab_size is None:
        # Try to get vocab_size from checkpoint
        if 'vocab_size' in checkpoint:
            vocab_size = checkpoint['vocab_size']
        elif hasattr(transformer_cfg, 'vocab_size'):
            vocab_size = transformer_cfg.vocab_size
        else:
            vocab_size = 8000  # Default fallback
        logger.info(f"Using vocab_size={vocab_size}")
    
    # Create model with all required parameters
    model = DGBTransformer(
        vocab_size=vocab_size,
        d_model=transformer_cfg.d_model,
        n_heads=transformer_cfg.n_heads,
        n_encoder_layers=transformer_cfg.n_encoder_layers,
        n_decoder_layers=transformer_cfg.n_decoder_layers,
        d_ff=transformer_cfg.d_ff,
        dropout=transformer_cfg.dropout,
        max_seq_len=transformer_cfg.max_seq_len,
        pad_idx=transformer_cfg.pad_idx,
        tie_embeddings=transformer_cfg.tie_embeddings,
        layer_norm_eps=transformer_cfg.layer_norm_eps,
    )
    
    # Load state dict
    model.load_state_dict(model_state)
    
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {Path(checkpoint_path).name}")
    
    return model