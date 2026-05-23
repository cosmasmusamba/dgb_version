"""
trainer/finetune/finetuner.py
===============================
Fine-tuning orchestrator that integrates with DGB's training infrastructure
"""

import logging
import time
import math
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FineTuningConfig:
    """Configuration for fine-tuning"""
    
    def __init__(self, cfg_dict: Dict[str, Any]):
        self.learning_rate = cfg_dict.get('learning_rate', 1e-5)
        self.batch_size = cfg_dict.get('batch_size', 8)
        self.epochs = cfg_dict.get('epochs', 3)
        self.warmup_steps = cfg_dict.get('warmup_steps', 100)
        self.weight_decay = cfg_dict.get('weight_decay', 0.01)
        self.grad_clip = cfg_dict.get('grad_clip', 1.0)
        self.gradient_accumulation_steps = cfg_dict.get('gradient_accumulation_steps', 4)
        self.freeze_layers = cfg_dict.get('freeze_layers', 0)
        self.freeze_embeddings = cfg_dict.get('freeze_embeddings', False)
        self.use_lora = cfg_dict.get('use_lora', False)
        self.lora_r = cfg_dict.get('lora_r', 8)
        self.lora_alpha = cfg_dict.get('lora_alpha', 32)
        self.lora_dropout = cfg_dict.get('lora_dropout', 0.1)
        self.task_type = cfg_dict.get('task_type', 'chat')
        self.max_seq_len = cfg_dict.get('max_seq_len', 512)
        self.save_every_epochs = cfg_dict.get('save_every_epochs', 1)
        self.eval_every_steps = cfg_dict.get('eval_every_steps', 100)
    
    @classmethod
    def from_config(cls, cfg) -> "FineTuningConfig":
        """Load from runtime_config.json"""
        ft_cfg = getattr(cfg, 'fine_tuning', {})
        return cls(ft_cfg)


class ModelFineTuner:
    """
    Fine-tune DGB models on custom datasets
    
    Usage:
        finetuner = ModelFineTuner(model, tokenizer, config)
        finetuner.fine_tune(train_path="datasets/finetune/train.jsonl")
        finetuner.save("checkpoints/dgb1/finetuned/model.pt")
    """
    
    def __init__(self, model, tokenizer, config: FineTuningConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        
        # Apply LoRA if configured
        if config.use_lora:
            from trainer.finetune.lora import apply_lora_to_model
            apply_lora_to_model(
                model,
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout
            )
        
        # Freeze layers if requested
        self._freeze_layers()
    
    def _freeze_layers(self):
        """Freeze bottom layers of encoder/decoder"""
        if self.config.freeze_layers <= 0 and not self.config.freeze_embeddings:
            return
        
        # Freeze embeddings
        if self.config.freeze_embeddings:
            if hasattr(self.model, 'src_embed'):
                for param in self.model.src_embed.parameters():
                    param.requires_grad = False
            if hasattr(self.model, 'tgt_embed'):
                for param in self.model.tgt_embed.parameters():
                    param.requires_grad = False
            logger.info("Froze embeddings")
        
        # Freeze encoder layers
        if hasattr(self.model, 'encoder_layers'):
            for i, layer in enumerate(self.model.encoder_layers):
                if i < self.config.freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                    logger.debug(f"Froze encoder layer {i}")
        
        # Freeze decoder layers
        if hasattr(self.model, 'decoder_layers'):
            for i, layer in enumerate(self.model.decoder_layers):
                if i < self.config.freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                    logger.debug(f"Froze decoder layer {i}")
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def fine_tune(
        self,
        train_data_path: Path,
        val_data_path: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """Main fine-tuning loop"""
        from trainer.finetune.dataset import create_finetune_dataloader
        
        # Create dataloaders
        train_loader = create_finetune_dataloader(
            data_path=train_data_path,
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            max_length=self.config.max_seq_len,
            task_type=self.config.task_type,
            shuffle=True
        )
        
        val_loader = None
        if val_data_path and val_data_path.exists():
            val_loader = create_finetune_dataloader(
                data_path=val_data_path,
                tokenizer=self.tokenizer,
                batch_size=self.config.batch_size,
                max_length=self.config.max_seq_len,
                task_type=self.config.task_type,
                shuffle=False
            )
        
        # Setup optimizer (lower LR for fine-tuning)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Warmup scheduler
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = min(self.config.warmup_steps, total_steps // 4)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Loss function
        criterion = nn.CrossEntropyLoss(ignore_index=-100)  # -100 is ignore index
        
        # Fine-tuning loop
        self.model.train()
        global_step = 0
        best_loss = float('inf')
        
        print(f"\n🎯 Starting Fine-Tuning")
        print(f"   Task: {self.config.task_type}")
        print(f"   Epochs: {self.config.epochs}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   LoRA: {'Yes' if self.config.use_lora else 'No'}")
        print(f"   Freeze layers: {self.config.freeze_layers}")
        print("="*50)
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.model(input_ids, input_ids)
                
                # Calculate loss
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                
                # Update weights
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # Log progress
                if batch_idx % 10 == 0:
                    current_loss = loss.item() * self.config.gradient_accumulation_steps
                    lr = scheduler.get_last_lr()[0]
                    print(f"   Epoch {epoch+1}/{self.config.epochs} | "
                          f"Step {batch_idx}/{len(train_loader)} | "
                          f"Loss: {current_loss:.4f} | LR: {lr:.2e}")
            
            # Epoch end
            avg_loss = epoch_loss / len(train_loader)
            epoch_duration = time.time() - epoch_start
            
            # Validation
            val_loss = None
            if val_loader:
                val_loss = self._evaluate(val_loader, criterion)
                print(f"\n📊 Epoch {epoch+1} completed:")
                print(f"   Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"   Duration: {epoch_duration:.1f}s")
            else:
                print(f"\n📊 Epoch {epoch+1} completed:")
                print(f"   Train Loss: {avg_loss:.4f} | Duration: {epoch_duration:.1f}s")
            
            # Save checkpoint
            if output_dir and (epoch + 1) % self.config.save_every_epochs == 0:
                self.save(output_dir / f"finetuned_epoch_{epoch+1}.pt")
            
            # Track best loss
            if val_loss and val_loss < best_loss:
                best_loss = val_loss
                if output_dir:
                    self.save(output_dir / "finetuned_best.pt")
        
        print(f"\n✅ Fine-tuning complete! Best loss: {best_loss:.4f}")
    
    def _evaluate(self, val_loader, criterion) -> float:
        """Run evaluation"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(val_loader)
    
    def save(self, path: Path):
        """Save fine-tuned model"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'learning_rate': self.config.learning_rate,
                'use_lora': self.config.use_lora,
                'task_type': self.config.task_type
            }
        }, path)
        logger.info(f"Fine-tuned model saved to {path}")
    
    def generate_response(self, instruction: str, max_tokens: int = 100) -> str:
        """Generate response from fine-tuned model"""
        self.model.eval()
        
        # Format prompt
        prompt = f"<BOS>Instruction: {instruction}\n\nResponse: "
        input_ids = self.tokenizer.encode(prompt)
        src = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.greedy_decode(
                src, bos_id=2, eos_id=3, max_len=max_tokens
            )[0].tolist()
        
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response