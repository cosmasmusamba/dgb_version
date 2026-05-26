"""
finetune/core/lora_adapter.py - ENTERPRISE GRADE
LoRA adapter for DGBTransformer - reuses existing utilities.

Reuses:
- transformer.utils.model_helpers: save_checkpoint, load_checkpoint, resolve_device
- modules.utils.safe_writer: atomic file operations
- modules.utils.path_resolver: PathResolver singleton
- modules.utils.run_context: RunContext singleton
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler

from configs.loader import get_config
from modules.utils.run_context import get_run_context
from modules.utils.path_resolver import get_path_resolver
from modules.utils.metrics_logger import MetricsLogger
from modules.utils.memory_manager import MemoryManager
from modules.utils.error_handler import log_exception, CheckpointError
from transformer.utils.model_helpers import save_checkpoint, load_checkpoint, resolve_device
from transformer.core.transformer_model import DGBTransformer

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear layers.
    Adds low-rank adapters while keeping original weights frozen.
    """
    def __init__(self, linear: nn.Linear, r: int = 4, alpha: int = 1, dropout: float = 0.0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        
        # Keep original linear (frozen)
        self.weight = linear.weight
        self.bias = linear.bias
        
        # Freeze original weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        
        # LoRA parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            # Initialize LoRA parameters
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass (frozen)
        base = nn.functional.linear(x, self.weight, self.bias)
        
        # LoRA forward pass
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            x = self.dropout(x)
            if x.dim() == 3:
                # Handle sequence inputs: (B, T, D)
                B, T, D = x.shape
                x_flat = x.reshape(-1, D)
                lora_out = (x_flat @ self.lora_A.t()) @ self.lora_B.t()
                lora_out = lora_out.reshape(B, T, self.out_features)
            else:
                lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
            return base + self.scaling * lora_out
        
        return base


class LoRAAdapter:
    """
    Enterprise LoRA adapter for DGBTransformer.
    
    Reuses:
    - save_checkpoint / load_checkpoint from transformer.utils.model_helpers
    - PathResolver singleton for consistent paths
    - RunContext singleton for run identification
    - resolve_device for device selection
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        run_ctx: Optional = None,
        model: Optional[nn.Module] = None,
    ):
        # Use singletons for consistency
        self.cfg = get_config()
        self.run_ctx = run_ctx or get_run_context()
        self.resolver = get_path_resolver(self.cfg.project.model_id, self.cfg)
        
        # Get finetune config
        self.fin_cfg = getattr(self.cfg, "finetune", None)
        if self.fin_cfg is None:
            self.fin_cfg = {}
        if hasattr(self.fin_cfg, "__dict__"):
            self.fin_cfg = {k: v for k, v in self.fin_cfg.__dict__.items() if not k.startswith("_")}
        
        # LoRA hyperparameters
        lora_cfg = self.fin_cfg.get("lora", {})
        self.lora_r = int(lora_cfg.get("r", 4))
        self.lora_alpha = int(lora_cfg.get("alpha", 1))
        self.lora_dropout = float(lora_cfg.get("dropout", 0.0))
        self.freeze_base = bool(self.fin_cfg.get("freeze_base", True))
        
        # Training hyperparameters
        optim_cfg = self.fin_cfg.get("optimizer", {})
        self.lr = float(optim_cfg.get("lr", 1e-4))
        self.weight_decay = float(optim_cfg.get("weight_decay", 0.01))
        self.betas = tuple(optim_cfg.get("betas", (0.9, 0.95)))
        self.eps = float(optim_cfg.get("eps", 1e-8))
        self.warmup_steps = int(self.fin_cfg.get("warmup_steps", 100))
        self.total_steps = int(self.fin_cfg.get("total_steps", 10000))
        self.max_grad_norm = float(self.fin_cfg.get("max_grad_norm", 1.0))
        
        # Mixed precision
        self.use_amp = bool(self.fin_cfg.get("use_amp", False)) and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Device
        self.device = resolve_device(self.cfg.training.device)
        
        # Load or use provided model
        self.model = model or self._load_base_model()
        
        # Apply LoRA to model
        self._inject_lora(self.model)
        
        # Move to device
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Metrics
        self.metrics = MetricsLogger(
            save_dir=self.resolver.logs_dir(),
            model_id=self.cfg.project.model_id,
            run_id=self.run_ctx.run_id,
        )
        self.memory = MemoryManager()
        
        logger.info(
            f"LoRAAdapter initialized: r={self.lora_r}, alpha={self.lora_alpha}, "
            f"lr={self.lr}, device={self.device}, use_amp={self.use_amp}"
        )
    
    def _load_base_model(self) -> DGBTransformer:
        """Load base DGBTransformer with pretrained weights."""
        from tokenizer.dgb_tokenizer import DGBTokenizer
        
        # Load tokenizer for correct vocab size
        tokenizer = DGBTokenizer.from_pretrained(self.resolver.tokenizer_dir())
        
        # Create model
        model = DGBTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=self.cfg.transformer.d_model,
            n_heads=self.cfg.transformer.n_heads,
            n_encoder_layers=self.cfg.transformer.n_encoder_layers,
            n_decoder_layers=self.cfg.transformer.n_decoder_layers,
            d_ff=self.cfg.transformer.d_ff,
            dropout=self.cfg.transformer.dropout,
            max_seq_len=self.cfg.transformer.max_seq_len,
            pad_idx=self.cfg.transformer.pad_idx,
            tie_embeddings=self.cfg.transformer.tie_embeddings,
        )
        
        # Load pretrained checkpoint
        base_ckpt = self.fin_cfg.get("base_checkpoint")
        if base_ckpt:
            ckpt_path = self.resolver.models_dir() / base_ckpt
            if ckpt_path.exists():
                load_checkpoint(ckpt_path, model, device=torch.device('cpu'))
                logger.info(f"Loaded base model from {ckpt_path.name}")
            else:
                logger.warning(f"Base checkpoint not found: {ckpt_path}")
        else:
            # Load latest checkpoint
            from transformer.utils.model_helpers import latest_checkpoint
            latest = latest_checkpoint(self.resolver.models_dir())
            if latest:
                load_checkpoint(latest, model, device=torch.device('cpu'))
                logger.info(f"Loaded latest checkpoint: {latest.name}")
        
        return model
    
    def _inject_lora(self, module: nn.Module, target_modules: list = None):
        """
        Recursively replace target Linear layers with LoRALinear wrappers.
        
        Target modules: attention projection layers (W_q, W_k, W_v, W_o)
        """
        if target_modules is None:
            target_modules = ['W_q', 'W_k', 'W_v', 'W_o']
        
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(t in name for t in target_modules):
                # Replace with LoRA wrapper
                lora_linear = LoRALinear(
                    child, 
                    r=self.lora_r, 
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout
                )
                setattr(module, name, lora_linear)
                logger.debug(f"Applied LoRA to {name}")
            else:
                self._inject_lora(child, target_modules)
    
    def _gather_trainable_params(self) -> list:
        """Collect trainable parameters (LoRA params only by default)."""
        trainable = []
        for name, param in self.model.named_parameters():
            # LoRA parameters should be trainable
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
                trainable.append(param)
            # Optionally include bias and LayerNorm
            elif self.fin_cfg.get("include_bias_and_norm", False):
                if "bias" in name or "norm" in name.lower():
                    param.requires_grad = True
                    trainable.append(param)
        
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
        return trainable
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build AdamW optimizer for trainable parameters."""
        params = self._gather_trainable_params()
        if not params:
            raise ValueError("No trainable parameters found. Check LoRA injection.")
        
        # Separate weight decay for biases
        decay_params = [p for p in params if p.dim() >= 2]
        no_decay_params = [p for p in params if p.dim() < 2]
        
        optimizer_group = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return optim.AdamW(optimizer_group, lr=self.lr, betas=self.betas, eps=self.eps)
    
    def _build_scheduler(self) -> LambdaLR:
        """Linear warmup + linear decay scheduler."""
        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            return max(0.0, 1.0 - progress)
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(
        self, 
        batch: list, 
        epoch: int = 0, 
        step: int = 0,
        criterion: Optional[nn.Module] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Perform a single training step.
        
        Args:
            batch: List of dicts with 'input_ids', 'attention_mask', 'labels'
            epoch: Current epoch
            step: Global step counter
            criterion: Loss function (defaults to CrossEntropyLoss)
        
        Returns:
            (loss, extra_metrics)
        """
        self.model.train()
        
        # Prepare batch tensors
        inputs = self._prepare_batch(batch)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Default criterion
        if criterion is None:
            criterion = nn.CrossEntropyLoss(ignore_index=self.cfg.transformer.pad_idx)
        
        self.optimizer.zero_grad()
        
        with autocast(enabled=self.use_amp):
            # DGBTransformer forward pass
            src = inputs.get("input_ids")
            tgt = inputs.get("labels", src)
            src_mask = self.model.make_padding_mask(src)
            tgt_mask = self.model.make_padding_mask(tgt)
            
            logits = self.model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            
            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
        
        # Backward pass with gradient scaling
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self._gather_trainable_params(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._gather_trainable_params(), self.max_grad_norm)
            self.optimizer.step()
        
        # Update scheduler
        self.scheduler.step()
        
        loss_val = loss.detach().cpu().item()
        current_lr = self.scheduler.get_last_lr()[0]
        
        # Metrics
        extra = {
            "lr": current_lr,
            "mem_gb": self.memory.available_gb if hasattr(self.memory, 'available_gb') else -1.0,
            "step": step,
            "epoch": epoch,
        }
        
        self.metrics.log_step(
            epoch=epoch, 
            step=step, 
            loss=loss_val, 
            lr=current_lr,
            grad_norm=self.max_grad_norm,
        )
        
        return loss_val, extra
    
    def _prepare_batch(self, batch: list) -> Dict[str, torch.Tensor]:
        """Convert batch to model input tensors using tokenizer.encode_batch."""
        from tokenizer.dgb_tokenizer import DGBTokenizer
        
        tokenizer = DGBTokenizer.from_pretrained(self.resolver.tokenizer_dir())
        
        input_texts = []
        output_texts = []
        
        for item in batch:
            input_text = item.get("input") or item.get("instruction") or item.get("text")
            output_text = item.get("output") or item.get("response") or input_text
            
            if input_text is None:
                raise ValueError(f"Batch item missing text field: {item}")
            
            input_texts.append(input_text)
            output_texts.append(output_text)
        
        max_len = self.cfg.transformer.max_seq_len
        
        inputs_encoding = tokenizer.encode_batch(
            input_texts,
            add_special_tokens=True,
            max_length=max_len,
            padding=True,
            truncation=True,
        )
        outputs_encoding = tokenizer.encode_batch(
            output_texts,
            add_special_tokens=True,
            max_length=max_len,
            padding=True,
            truncation=True,
        )
        
        # BatchEncoding has .input_ids attribute (list of lists)
        input_ids = inputs_encoding.input_ids
        labels = outputs_encoding.input_ids
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    
    def save_checkpoint(self, state: Dict[str, Any], name: Optional[str] = None) -> Path:
        """
        Save checkpoint using enterprise save_checkpoint from model_helpers.py.
        
        Args:
            state: Additional metadata to save
            name: Checkpoint filename (auto-generated if None)
        
        Returns:
            Path to saved checkpoint
        """
        ckpt_dir = self.resolver.models_dir()
        
        if name is None:
            name = self.run_ctx.prefix(f"finetune_step_{state.get('step', 'final')}.pt")
        
        ckpt_path = ckpt_dir / name
        
        # Reuse enterprise save_checkpoint
        return save_checkpoint(
            path=ckpt_path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=state.get("epoch", 0),
            step=state.get("step", 0),
            loss=state.get("loss", 0.0),
            extra={
                "lora_config": {
                    "r": self.lora_r,
                    "alpha": self.lora_alpha,
                    "dropout": self.lora_dropout,
                },
                "finetune_state": state,
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict() if self.use_amp else None,
            },
            ctx=self.run_ctx,
        )
    
    def load_checkpoint(self, path: Path, load_optimizer: bool = True) -> Dict[str, Any]:
        """
        Load checkpoint using enterprise load_checkpoint from model_helpers.py.
        
        Args:
            path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        
        Returns:
            Checkpoint payload dict
        """
        # Reuse enterprise load_checkpoint
        payload = load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer if load_optimizer else None,
            device=self.device,
        )
        
        # Restore scheduler and scaler from extra
        extra = payload.get("extra", {})
        if "scheduler_state" in extra and self.scheduler:
            self.scheduler.load_state_dict(extra["scheduler_state"])
        if "scaler_state" in extra and self.use_amp:
            self.scaler.load_state_dict(extra["scaler_state"])
        
        logger.info(f"LoRA checkpoint loaded from {path.name}")
        return payload
    
    def get_trainable_params_count(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self._gather_trainable_params())
    
    def state_dict(self) -> Dict[str, Any]:
        """Return full state dict for serialization."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "lora_config": {
                "r": self.lora_r,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
            },
        }

    def merge_and_unload(self) -> None:
        """
        Merge LoRA weights into the original linear layers and replace LoRALinear
        wrappers with plain nn.Linear. After this, `self.model` is a standard
        DGBTransformer without LoRA parameters.
        """
        from torch import nn
        for name, module in list(self.model.named_modules()):
            if isinstance(module, LoRALinear):
                # Compute merged weight: W = W + (alpha/r) * (B @ A)
                weight = module.weight.data
                if module.r > 0 and module.lora_A is not None and module.lora_B is not None:
                    lora_weight = (module.lora_B @ module.lora_A) * module.scaling
                    weight = weight + lora_weight

                # Create plain Linear layer
                new_linear = nn.Linear(module.in_features, module.out_features,
                                       bias=module.bias is not None)
                new_linear.weight.data = weight
                if module.bias is not None:
                    new_linear.bias.data = module.bias.data

                # Replace LoRALinear wrapper in the parent module
                parent = self.model
                for part in name.split('.')[:-1]:
                    parent = getattr(parent, part)
                child_name = name.split('.')[-1]
                setattr(parent, child_name, new_linear)

        logger.info("LoRA weights merged and unloaded – model is now a plain DGBTransformer")