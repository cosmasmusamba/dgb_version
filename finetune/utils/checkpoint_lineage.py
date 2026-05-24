"""
Record checkpoint lineage metadata using modules.utils.safe_writer for atomic writes.
This links finetune checkpoints to parent/base checkpoints and records dataset/config hashes.
"""
import json
from typing import Dict, Any, Optional

from modules.utils.safe_writer import safe_write
from modules.utils.run_context import RunContext
from modules.utils.pipeline_state import PipelineState

def record_lineage(config: Dict[str, Any], state: PipelineState, run_ctx: Optional[RunContext] = None):
    run_ctx = run_ctx or RunContext.default()
    lineage = {
        "run_id": run_ctx.run_id,
        "checkpoint_id": state.current_checkpoint(),
        "parent_checkpoint": config["finetune"].get("parent_checkpoint"),
        "dataset_hash": config["finetune"].get("dataset_hash"),
        "config_snapshot": config["finetune"],
        "timestamp": run_ctx.timestamp_iso()
    }
    path = run_ctx.path("checkpoints/logs/finetune_lineage.json")
    safe_write(path, json.dumps(lineage, indent=2))
