"""
trainer/core/trainer.py
=========================
DGBTrainer — high-level training orchestrator.

FIX B2 (v3.0.0): Removed the early load_checkpoint() call that was here.
training_loop.py handles checkpoint loading WITH optimizer state correctly.
This class only builds objects and delegates to run_training_loop().
"""
from __future__ import annotations
import logging, time
from pathlib import Path
from typing import Optional
logger = logging.getLogger(__name__)

class DGBTrainer:
    def __init__(self, cfg=None):
        from configs.loader import get_config
        self._cfg = cfg or get_config()

    def train(self, run_id: str = "", force: bool = False) -> float:
        from modules.utils.run_context import RunContext
        from modules.utils.path_resolver import init_path_resolver
        from modules.utils.unified_log import init_unified_log
        from modules.utils.system_detector import get_system_profile
        from modules.utils.dynamic_resource_manager import DynamicResourceManager
        from modules.utils.metrics_logger import MetricsLogger
        from modules.utils.progress_tracking import ProgressTracker
        from transformer.core.transformer_model import DGBTransformer
        from transformer.utils.model_helpers import resolve_device, log_model_info
        from trainer.core.checkpoint_manager import CheckpointManager
        from trainer.core.training_loop import TrainingConfig, run_training_loop
        from tokenizer.dgb_tokenizer import DGBTokenizer

        cfg = self._cfg
        ctx = RunContext(model_id=cfg.project.model_id, run_id=run_id or None)
        res = init_path_resolver(cfg.project.model_id, cfg)

        tokenizer = DGBTokenizer.from_pretrained(res.tokenizer_dir())
        device    = resolve_device(cfg.training.device)
        tf        = cfg.transformer

        model = DGBTransformer(
            vocab_size=tokenizer.vocab_size, d_model=tf.d_model, n_heads=tf.n_heads,
            n_encoder_layers=tf.n_encoder_layers, n_decoder_layers=tf.n_decoder_layers,
            d_ff=tf.d_ff, dropout=tf.dropout, max_seq_len=tf.max_seq_len,
            pad_idx=tf.pad_idx, tie_embeddings=tf.tie_embeddings,
        ).to(device)
        log_model_info(model, "DGBTransformer")

        train_cfg = TrainingConfig.from_cfg(cfg)
        profile   = get_system_profile()
        resource  = DynamicResourceManager.from_profile(profile, cfg)
        ckpt_mgr  = CheckpointManager(
            models_dir=res.models_dir(), log_dir=res.logs_dir(),
            save_every_epochs=train_cfg.save_every_epochs, run_id=ctx.run_id,
        )
        # FIX B2: NO load_checkpoint here. training_loop.py does it with optimizer.
        metrics = MetricsLogger(log_dir=res.logs_dir(), run_id=ctx.run_id)
        unified = init_unified_log(
            path=res.logs_dir() / ctx.training_log_name(),
            run_id=ctx.run_id, model_id=cfg.project.model_id,
        )

        class _Prog:
            epoch=0; global_step=0; batch_idx=0; best_loss=float("inf")
        prog = _Prog()
        ckpt_mgr.restore_progress(prog)

        best = run_training_loop(
            model=model, tokenizer=tokenizer,
            cleaned_dir=res.cleaned_dir(), device=device,
            train_cfg=train_cfg, checkpoint_mgr=ckpt_mgr,
            resource_handle=resource, metrics_logger=metrics,
            unified_log=unified, progress=prog,
            run_id=ctx.run_id, model_id=cfg.project.model_id,
        )
        metrics.flush()
        unified.close()
        return best
