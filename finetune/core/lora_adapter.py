# finetune/core/lora_adapter.py
"""
Concrete PyTorch LoRA adapter tailored to the repo's transformer model.
Features:
- In-place LoRA injection for nn.Linear layers
- Optimizer and scheduler setup from config
- Mixed precision training using torch.cuda.amp
- Checkpoint save/load with safe_writer and path_resolver
- Metrics and unified logging integration
- Minimal external assumptions: expects transformer model class at transformer.core.transformer_model.TransformerModel
"""

from typing import Dict, Optional, Tuple, Any
import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler

from modules.utils.unified_log import UnifiedLogger
from modules.utils.metrics_logger import MetricsLogger
from modules.utils.memory_manager import MemoryManager
from modules.utils.safe_writer import safe_write
from modules.utils.path_resolver import PathResolver
from modules.utils.run_context import RunContext

# Try to import your transformer model class; adapt name if different
try:
    from transformer.core.transformer_model import TransformerModel
except Exception:
    TransformerModel = None  # fallback; user must inject model via config if not present


class LoRALinear(nn.Module):
    """
    Replacement wrapper for nn.Linear that adds LoRA low-rank adapters.
    Forward: y = linear(x) + alpha / r * (A @ (B @ x))
    A: (out_features, r), B: (r, in_features)
    """
    def __init__(self, orig_linear: nn.Linear, r: int = 4, alpha: int = 1, dropout: float = 0.0):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        # original linear kept frozen by default
        self.weight = orig_linear.weight
        self.bias = orig_linear.bias

        # LoRA parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((self.out_features, r)))
            self.lora_B = nn.Parameter(torch.zeros((r, self.in_features)))
            # initialize following LoRA paper: A ~ N(0, 0.01), B ~ N(0, 0.01)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = nn.functional.linear(x, self.weight, self.bias)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = self.dropout(x)  # apply dropout to input if configured
            # B @ x^T -> (r, batch, seq_len?) handle 2D/3D inputs by flattening last dim
            # Use linear for efficiency: (x @ B.T) then @ A.T
            # x: (batch, in_features) or (batch, seq, in_features)
            orig_shape = x.shape
            if x.dim() == 3:
                b, s, _ = x.shape
                x_flat = x.reshape(b * s, self.in_features)
                mid = torch.matmul(x_flat, self.lora_B.t())  # (b*s, r)
                out_flat = torch.matmul(mid, self.lora_A.t())  # (b*s, out_features)
                lora_out = out_flat.reshape(b, s, self.out_features)
            else:
                mid = torch.matmul(x, self.lora_B.t())  # (batch, r)
                lora_out = torch.matmul(mid, self.lora_A.t())  # (batch, out_features)
            return base + self.scaling * lora_out
        return base


class LoRAAdapter:
    """
    LoRAAdapter manages:
    - loading base model (or receiving injected model)
    - injecting LoRA modules into linear layers
    - optimizer, scheduler, mixed precision training
    - checkpointing and metrics logging
    """

    def __init__(self, config: Dict[str, Any], run_ctx: Optional[RunContext] = None, model: Optional[nn.Module] = None):
        self.config = config
        self.run_ctx = run_ctx or RunContext.default()
        self.logger = UnifiedLogger(component="lora_adapter", run_ctx=self.run_ctx)
        self.metrics = MetricsLogger(namespace="finetune", run_ctx=self.run_ctx)
        self.memory = MemoryManager()
        self.path_resolver = PathResolver(self.run_ctx)

        # device selection
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # LoRA hyperparameters
        fin_cfg = config.get("finetune", {})
        self.lora_r = int(fin_cfg.get("lora", {}).get("r", 4))
        self.lora_alpha = int(fin_cfg.get("lora", {}).get("alpha", 1))
        self.lora_dropout = float(fin_cfg.get("lora", {}).get("dropout", 0.0))
        self.freeze_base = bool(fin_cfg.get("freeze_base", True))

        # training hyperparameters
        optim_cfg = fin_cfg.get("optimizer", {})
        self.lr = float(optim_cfg.get("lr", 1e-4))
        self.weight_decay = float(optim_cfg.get("weight_decay", 0.0))
        self.betas = tuple(optim_cfg.get("betas", (0.9, 0.95)))
        self.eps = float(optim_cfg.get("eps", 1e-8))
        self.warmup_steps = int(fin_cfg.get("warmup_steps", 100))
        self.total_steps = int(fin_cfg.get("total_steps", 10000))
        self.scheduler_type = fin_cfg.get("scheduler", "linear")

        # mixed precision
        self.use_amp = bool(fin_cfg.get("use_amp", True)) and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        # model: either provided or loaded from transformer module
        if model is not None:
            self.model = model
        else:
            self.model = self._load_base_model(config)

        # inject LoRA into model
        self._inject_lora(self.model)

        # move to device
        self.model.to(self.device)

        # prepare optimizer to only update LoRA params (and optionally bias/LayerNorm if configured)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(self.optimizer)

        self.logger.log_event("lora_adapter_init", {
            "device": str(self.device),
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "use_amp": self.use_amp
        })

    def _load_base_model(self, config: Dict[str, Any]) -> nn.Module:
        """
        Load base transformer model. If TransformerModel is not importable,
        expect a path to a serialized model or a factory in config.
        """
        if TransformerModel is not None:
            model_cfg = config.get("model", {})
            model = TransformerModel(model_cfg)
            # optionally load pretrained weights if provided
            base_ckpt = config.get("finetune", {}).get("base_checkpoint")
            if base_ckpt:
                resolved = self.path_resolver.resolve(base_ckpt)
                if os.path.exists(resolved):
                    state = torch.load(resolved, map_location="cpu")
                    model.load_state_dict(state, strict=False)
            return model
        else:
            raise RuntimeError("TransformerModel not found. Provide a model instance via LoRAAdapter(..., model=your_model)")

    def _inject_lora(self, module: nn.Module):
        """
        Recursively replace nn.Linear modules with LoRALinear wrappers.
        Only replace layers that are appropriate (e.g., query/key/value/proj).
        """
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                # create LoRA wrapper preserving original parameters
                lora_linear = LoRALinear(child, r=self.lora_r, alpha=self.lora_alpha, dropout=self.lora_dropout)
                # optionally freeze base weights
                if self.freeze_base:
                    lora_linear.weight.requires_grad = False
                    if lora_linear.bias is not None:
                        lora_linear.bias.requires_grad = False
                setattr(module, name, lora_linear)
            else:
                # recurse
                self._inject_lora(child)

    def _gather_trainable_params(self):
        """
        Collect parameters that should be optimized: LoRA A/B params and optionally biases/LayerNorms.
        """
        trainable = []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                trainable.append(p)
            else:
                # LoRA params may be set requires_grad True; ensure they are included
                if "lora_A" in n or "lora_B" in n:
                    p.requires_grad = True
                    trainable.append(p)
        # Optionally include LayerNorm and bias if config requests
        fin_cfg = self.config.get("finetune", {})
        include_bias_and_norm = fin_cfg.get("include_bias_and_norm", False)
        if include_bias_and_norm:
            for n, p in self.model.named_parameters():
                if ("bias" in n or "layernorm" in n.lower() or "ln" in n.lower()) and p not in trainable:
                    p.requires_grad = True
                    trainable.append(p)
        return trainable

    def _build_optimizer(self):
        params = self._gather_trainable_params()
        # AdamW optimizer
        optim_group = [
            {"params": [p for p in params if p.ndim > 1], "weight_decay": self.weight_decay},
            {"params": [p for p in params if p.ndim == 1], "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(optim_group, lr=self.lr, betas=self.betas, eps=self.eps)
        return optimizer

    def _build_scheduler(self, optimizer):
        # simple linear warmup -> decay scheduler
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0,
                float(self.total_steps - current_step) / float(max(1, self.total_steps - self.warmup_steps))
            )
        return LambdaLR(optimizer, lr_lambda)

    def train_step(self, batch: list, epoch: int = 0, step: int = 0) -> Tuple[float, Dict[str, Any]]:
        """
        Perform a single training step.
        Expects batch to be a list of dicts with keys matching your tokenizer/model input.
        Returns (loss, extra_info)
        """
        self.model.train()
        # convert batch to tensors according to your model's expected input
        inputs = self._prepare_batch(batch)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.optimizer.zero_grad()
        with autocast(enabled=self.use_amp):
            outputs = self.model(**inputs)
            # assume model returns dict with 'loss' or tuple (loss, logits)
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            elif isinstance(outputs, tuple):
                loss = outputs[0]
            else:
                # fallback: compute a dummy loss if model doesn't provide one
                raise RuntimeError("Model must return loss in outputs for finetuning")

        # scale and backward
        if self.use_amp:
            self.scaler.scale(loss).backward()
            # gradient clipping if configured
            max_grad_norm = float(self.config.get("finetune", {}).get("max_grad_norm", 1.0))
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self._gather_trainable_params(), max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            max_grad_norm = float(self.config.get("finetune", {}).get("max_grad_norm", 1.0))
            torch.nn.utils.clip_grad_norm_(self._gather_trainable_params(), max_grad_norm)
            self.optimizer.step()

        # scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        # metrics and memory
        mem = self.memory.current_usage()
        loss_val = loss.detach().cpu().item()
        extra = {"mem_usage": mem}
        # log metrics
        self.metrics.log({"epoch": epoch, "step": step, "loss": loss_val, "mem_usage": mem})
        self.logger.log_event("train_step", {"epoch": epoch, "step": step, "loss": loss_val, "mem_usage": mem})

        return loss_val, extra

    def _prepare_batch(self, batch: list) -> Dict[str, torch.Tensor]:
        """
        Convert a batch (list of dicts) into model input tensors.
        This function must be adapted to your tokenizer and model signature.
        Default expects each item to have 'input_ids' and 'attention_mask' and optional 'labels'.
        """
        # collate simple lists into tensors
        # find keys
        keys = set().union(*(item.keys() for item in batch))
        collated = {}
        for k in keys:
            values = [item.get(k) for item in batch]
            # if already tensors, stack; if lists, convert to tensor
            if isinstance(values[0], torch.Tensor):
                collated[k] = torch.stack(values)
            else:
                collated[k] = torch.tensor(values, dtype=torch.long)
        return collated

    def save_checkpoint(self, state: Dict[str, Any], name: Optional[str] = None):
        """
        Save model + optimizer + scheduler + scaler state atomically using safe_write.
        """
        ckpt_dir = self.path_resolver.resolve(self.config.get("finetune", {}).get("checkpoint_dir", "checkpoints/finetune"))
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_name = name or f"finetune_{self.run_ctx.run_id}_step_{state.get('step', 'final')}.pt"
        path = os.path.join(ckpt_dir, ckpt_name)
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state": self.scaler.state_dict() if self.use_amp else None,
            "meta": state
        }
        # write atomically
        safe_write(path, torch.save(payload, path))
        self.logger.log_event("checkpoint_saved", {"path": path})
        return path

    def load_checkpoint(self, path: str):
        """
        Load checkpoint into model/optimizer/scaler/scheduler.
        """
        resolved = self.path_resolver.resolve(path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(resolved)
        ckpt = torch.load(resolved, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self.use_amp and "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.logger.log_event("checkpoint_loaded", {"path": resolved})

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.use_amp else None
        }
