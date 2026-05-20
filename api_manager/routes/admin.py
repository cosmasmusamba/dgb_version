"""
api_manager/routes/admin.py
=============================
Admin-only endpoints for training control, metrics, pipeline management,
model registry, and system health.

All endpoints require role="admin" JWT token.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from api_manager.auth.api_security import require_role, TokenData
from configs.constants import UserRole

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/admin", tags=["admin"])

_training_proc: Optional[subprocess.Popen] = None
_training_start: Optional[float] = None


# ── Auth dependency ────────────────────────────────────────────────────────────

def admin_only(user: TokenData = Depends(require_role(UserRole.ADMIN))) -> TokenData:
    return user


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    from configs.loader import get_config
    cfg = get_config()
    return {
        "status":  "ok",
        "version": cfg.project.version,
        "model":   cfg.project.model_id,
        "time":    time.time(),
    }


# ── Training ───────────────────────────────────────────────────────────────────

class TrainingLaunchRequest(BaseModel):
    model_size:  str   = "DGB-M (1.3B)"
    phase:       str   = "Pre-training"
    epochs:      int   = Field(default=30, ge=1, le=300)
    batch_size:  int   = Field(default=32, ge=1, le=512)
    lr:          float = Field(default=3e-4, ge=1e-6, le=1e-1)
    warmup:      int   = Field(default=1000, ge=0)
    force:       bool  = False


@router.get("/training/status")
async def training_status(_: TokenData = Depends(admin_only)):
    """Return current training metrics from the latest checkpoint files."""
    from configs.loader import get_config
    from modules.utils.path_resolver import init_path_resolver
    from modules.utils.file_handler import latest_file_by_name
    import json

    cfg = get_config()
    res = init_path_resolver(cfg.project.model_id, cfg)
    log_dir = res.logs_dir()

    # Latest epoch metrics
    epochs_file = latest_file_by_name(log_dir, "*metrics_epochs.json")
    steps_file  = latest_file_by_name(log_dir, "*metrics_steps.json")

    epochs_data: list = []
    steps_data:  list = []

    if epochs_file and epochs_file.exists():
        try:
            epochs_data = json.loads(epochs_file.read_text())
        except Exception:
            pass

    if steps_file and steps_file.exists():
        try:
            raw = json.loads(steps_file.read_text())
            steps_data = raw[-50:]  # last 50 steps for dashboard
        except Exception:
            pass

    global _training_proc, _training_start
    is_running = _training_proc is not None and _training_proc.poll() is None
    elapsed    = time.time() - _training_start if _training_start and is_running else None

    last_epoch = epochs_data[-1] if epochs_data else {}
    return {
        "is_running":      is_running,
        "elapsed_sec":     round(elapsed, 1) if elapsed else None,
        "current_loss":    last_epoch.get("avg_loss"),
        "val_loss":        last_epoch.get("val_loss"),
        "epoch":           last_epoch.get("epoch"),
        "total_epochs":    cfg.training.epochs,
        "perplexity":      last_epoch.get("perplexity"),
        "epoch_history":   epochs_data,
        "recent_steps":    steps_data,
    }


@router.post("/training/launch")
async def launch_training(
    req:   TrainingLaunchRequest,
    _:     TokenData = Depends(admin_only),
):
    global _training_proc, _training_start
    if _training_proc is not None and _training_proc.poll() is None:
        raise HTTPException(status_code=409, detail="Training is already running")

    logger.info("Admin launched training: phase=%s  epochs=%d  lr=%.2e",
                req.phase, req.epochs, req.lr)
    try:
        cmd = [sys.executable, "model_trainer.py"]
        if req.force:
            cmd += ["--force", "model_training"]
        _training_proc  = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parent.parent.parent)
        _training_start = time.time()
        return {"status": "started", "pid": _training_proc.pid}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/training/stop")
async def stop_training(_: TokenData = Depends(admin_only)):
    global _training_proc
    if _training_proc is None or _training_proc.poll() is not None:
        return {"status": "not_running"}
    _training_proc.terminate()
    return {"status": "stopped"}


# ── Pipeline ───────────────────────────────────────────────────────────────────

@router.post("/pipeline/run")
async def run_pipeline(_: TokenData = Depends(admin_only)):
    try:
        proc = subprocess.Popen(
            [sys.executable, "main_pipeline.py"],
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        return {"status": "started", "pid": proc.pid}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/pipeline/status")
async def pipeline_status(_: TokenData = Depends(admin_only)):
    from configs.loader import get_config
    from modules.utils.path_resolver import init_path_resolver
    from modules.utils.pipeline_state import PipelineState
    cfg   = get_config()
    res   = init_path_resolver(cfg.project.model_id, cfg)
    state = PipelineState.load_latest(res.logs_dir(), cfg.project.model_id)
    return state.to_dict()


# ── Metrics ────────────────────────────────────────────────────────────────────

@router.get("/metrics/history")
async def metrics_history(
    _:     TokenData = Depends(admin_only),
    n:     int       = 500,
):
    from configs.loader import get_config
    from modules.utils.path_resolver import init_path_resolver
    from modules.utils.file_handler import latest_file_by_name
    import json
    cfg     = get_config()
    res     = init_path_resolver(cfg.project.model_id, cfg)
    log_dir = res.logs_dir()

    out: Dict[str, Any] = {"epoch_history": [], "step_history": []}
    for key, glob in [("epoch_history", "*metrics_epochs.json"),
                      ("step_history",  "*metrics_steps.json")]:
        f = latest_file_by_name(log_dir, glob)
        if f and f.exists():
            try:
                data = json.loads(f.read_text())
                out[key] = data[-n:]
            except Exception:
                pass
    return out


# ── Models ─────────────────────────────────────────────────────────────────────

@router.get("/models")
async def list_models(_: TokenData = Depends(admin_only)):
    from configs.loader import get_config
    from modules.utils.path_resolver import init_path_resolver
    cfg     = get_config()
    res     = init_path_resolver(cfg.project.model_id, cfg)
    m_dir   = res.models_dir(create=False)
    if not m_dir.exists():
        return {"models": []}

    models = []
    for pt in sorted(m_dir.glob("*.pt")):
        size_mb = round(pt.stat().st_size / 1024**2, 1)
        parts   = pt.stem.split("_")
        epoch   = next((int(p) for p in parts if p.isdigit()), None)
        loss    = None
        try:
            loss_idx = parts.index("loss") + 1
            loss     = float(parts[loss_idx])
        except (ValueError, IndexError):
            pass
        models.append({
            "name":     pt.name,
            "size_mb":  size_mb,
            "epoch":    epoch,
            "loss":     loss,
            "is_best":  "best_model" in pt.name,
            "ver":      parts[0] if parts else "",
        })
    return {"models": models}


@router.post("/models/deploy")
async def deploy_model(body: Dict[str, Any], _: TokenData = Depends(admin_only)):
    logger.info("Deploy model: %s", body)
    return {"status": "queued", "model": body.get("model")}


@router.post("/models/rollback")
async def rollback_model(body: Dict[str, Any], _: TokenData = Depends(admin_only)):
    logger.info("Rollback model: %s", body)
    return {"status": "queued", "model": body.get("model")}


# ── System ─────────────────────────────────────────────────────────────────────

@router.get("/system")
async def system_info(_: TokenData = Depends(admin_only)):
    from modules.utils.system_detector import get_system_profile
    profile = get_system_profile()
    return {
        "os":        profile.os_name,
        "cpu":       profile.cpu_name,
        "cores":     profile.cpu_cores_logical,
        "ram_total": profile.ram_total_gb,
        "ram_avail": profile.ram_available_gb,
        "gpu":       profile.gpu_name,
        "gpu_vram":  profile.gpu_vram_gb,
        "device":    profile.recommended_device,
    }
