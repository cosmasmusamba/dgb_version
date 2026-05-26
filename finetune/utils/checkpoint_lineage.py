"""
finetune/utils/checkpoint_lineage.py
Record checkpoint lineage metadata using modules.utils.safe_writer for atomic writes.
This links finetune checkpoints to parent/base checkpoints and records dataset/config hashes.
"""
import json
from typing import Dict, Any, Optional
from datetime import datetime

from modules.utils.safe_writer import atomic_write_json
from modules.utils.run_context import get_run_context
from modules.utils.pipeline_state import PipelineState


def record_lineage(
    config,
    state: PipelineState,
    run_ctx: Optional = None,
    checkpoint_path: Optional = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record checkpoint lineage information.
    
    Args:
        config: DGBConfig instance or dict
        state: PipelineState instance
        run_ctx: RunContext instance
        checkpoint_path: Path to the checkpoint file
        metrics: Additional metrics to record
    """
    from configs.loader import get_config
    from modules.utils.path_resolver import init_path_resolver
    
    run_ctx = run_ctx or get_run_context()
    
    # Handle both DGBConfig object and dict
    if hasattr(config, "project"):
        # It's a DGBConfig object
        model_id = config.project.model_id
        finetune_config = getattr(config, "finetune", {})
        if hasattr(finetune_config, "__dict__"):
            finetune_config = {k: v for k, v in finetune_config.__dict__.items() if not k.startswith("_")}
    else:
        # It's a dict
        model_id = config.get("project", {}).get("model_id", "dgb1")
        finetune_config = config.get("finetune", {})
    
    # Get resolver for paths
    cfg = get_config()
    resolver = init_path_resolver(model_id, cfg)
    
    # Create lineage directory
    lineage_dir = resolver.logs_dir() / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)
    
    # Build lineage data
    lineage_data = {
        "run_id": run_ctx.run_id,
        "model_id": model_id,
        "timestamp": datetime.now().isoformat(),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "parent_checkpoint": finetune_config.get("parent_checkpoint"),
        "dataset_hash": finetune_config.get("dataset_hash"),
        "config_snapshot": finetune_config,
        "stage_state": state.get_kv_all() if hasattr(state, "get_kv_all") else {},
        "metrics": metrics or {},
    }
    
    # Save lineage file
    lineage_file = lineage_dir / run_ctx.prefix("finetune_lineage.json")
    atomic_write_json(lineage_file, lineage_data)