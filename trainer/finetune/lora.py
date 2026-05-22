"""
trainer/finetune/lora.py
=========================
Low-Rank Adaptation (LoRA) implementation for parameter-efficient fine-tuning
"""

import math
import logging
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """LoRA wrapper for linear layers"""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        alpha: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.original = original_layer
        self.r = r
        self.alpha = alpha
        
        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False
        
        # LoRA matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        self.scaling = alpha / r
    
    def forward(self, x):
        original_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))
        return original_out + self.scaling * lora_out


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Optional[list] = None
) -> nn.Module:
    """
    Apply LoRA to all linear layers in attention and FFN modules
    
    Args:
        model: The model to apply LoRA to
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout rate
        target_modules: List of module name patterns to target (e.g., ['attn', 'ff'])
    """
    if target_modules is None:
        target_modules = ['attn', 'ff', 'W_q', 'W_k', 'W_v', 'W_o', 'fc1', 'fc2']
    
    def _apply_lora(module, name=''):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Check if this module should be replaced
            should_replace = any(
                target in full_name or target in child_name
                for target in target_modules
            )
            
            if should_replace and isinstance(child, nn.Linear):
                # Replace with LoRA version
                lora_layer = LoRALinear(child, r, alpha, dropout)
                setattr(module, child_name, lora_layer)
                logger.debug(f"Applied LoRA to {full_name}")
            else:
                # Recurse into child
                _apply_lora(child, full_name)
    
    _apply_lora(model)
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA applied: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
    
    return model