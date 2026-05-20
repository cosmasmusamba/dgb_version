"""
trainer/core/training_loop.py
================================
Core training loop — fully decoupled from model_trainer.py entry point.

FIXES APPLIED IN v3.0.0
------------------------
B4: learning_rate now read from TrainingConfig.from_cfg() — not hardcoded
T4: seed applied to torch / random / numpy at loop start
T6: num_workers read from config — defaults to 0 until multi-file corpus
    eval_every_steps wired into eval loop
    validation split support via split_cleaned_files()
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from modules.utils.streaming import get_training_hub, StreamEvent
from modules.utils.metrics_logger import MetricsLogger
from modules.utils.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class MovingAverage:
    """Exponential moving average for loss trend analytics."""
    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self._value: Optional[float] = None
        self._count = 0

    def update(self, value: float) -> float:
        self._count += 1
        if self._value is None:
            self._value = value
        else:
            self._value = self._alpha * value + (1 - self._alpha) * self._value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value

    @property
    def count(self) -> int:
        return self._count


class WarmupScheduler:
    """Linear warmup → inverse-sqrt decay (Transformer paper schedule)."""

    def __init__(self, optimizer, d_model: int, warmup_steps: int) -> None:
        self._opt  = optimizer
        self._d    = d_model
        self._wu   = max(warmup_steps, 1)
        self._step = 0

    def step(self) -> float:
        self._step += 1
        lr = self._d ** -0.5 * min(
            self._step ** -0.5,
            self._step * self._wu ** -1.5,
        )
        for pg in self._opt.param_groups:
            pg["lr"] = lr
        return lr

    @property
    def current_lr(self) -> float:
        s = max(self._step, 1)
        return self._d ** -0.5 * min(s ** -0.5, s * self._wu ** -1.5)

    @property
    def step_count(self) -> int:
        return self._step

    def restore(self, step: int) -> None:
        self._step = step


@dataclass
class TrainingConfig:
    """All hyperparameters needed by the training loop — all read from config."""
    epochs:             int   = 30
    warmup_steps:       int   = 1_000
    grad_clip:          float = 1.0
    log_every_steps:    int   = 1
    save_every_epochs:  int   = 5
    eval_every_steps:   int   = 500     # FIX: now wired
    pad_idx:            int   = 0
    mixed_precision:    bool  = False
    d_model:            int   = 256
    learning_rate:      float = 3e-4    # FIX B4: read from config
    weight_decay:       float = 0.01
    seed:               int   = 42      # FIX T4: applied in loop
    num_workers:        int   = 0       # FIX T6: 0 until multi-file
    validation_split:   float = 0.1

    @classmethod
    def from_cfg(cls, cfg) -> "TrainingConfig":
        tr = cfg.training
        tf = cfg.transformer
        ds = cfg.dataset
        return cls(
            epochs=tr.epochs,
            warmup_steps=tr.warmup_steps,
            grad_clip=tr.grad_clip,
            log_every_steps=tr.log_every_steps,
            save_every_epochs=tr.save_every_epochs,
            eval_every_steps=tr.eval_every_steps,
            pad_idx=tf.pad_idx,
            mixed_precision=tr.mixed_precision,
            d_model=tf.d_model,
            learning_rate=tr.learning_rate,   # FIX B4
            weight_decay=tr.weight_decay,
            seed=tr.seed,                     # FIX T4
            num_workers=tr.num_workers,       # FIX T6
            validation_split=getattr(ds, "validation_split", 0.1),
        )


def split_cleaned_files(
    cleaned_dir: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """
    Split cleaned text files into train/val sets.
    Returns (train_files, val_files).
    With only 1 file returns (all_files, []) with a warning.
    """
    from modules.utils.file_handler import list_files
    rng   = random.Random(seed)
    files = sorted(list_files(cleaned_dir, glob="*.txt"))
    if not files:
        raise FileNotFoundError(f"No cleaned .txt files in {cleaned_dir}")
    if len(files) < 2:
        logger.warning(
            "Only %d cleaned file(s) — cannot create validation split. "
            "Add more Wikipedia dump files and re-run the pipeline.",
            len(files),
        )
        return files, []
    shuffled = list(files)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def _eval_pass(model, tokenizer, val_files, device, pad_idx, batch_size, num_workers):
    """Run a full pass over validation files, return avg val_loss."""
    if not val_files:
        return None
    from trainer.core.dataset_loader import StreamingTextDataset, build_streaming_loader
    dataset = StreamingTextDataset(
        tokenizer=tokenizer,
        cleaned_dir=val_files[0].parent,
        max_seq_len=512,
    )
    # Override file list to only val files
    dataset._files = sorted(val_files)
    loader = build_streaming_loader(dataset, batch_size=max(4, batch_size // 2),
                                    num_workers=0, pad_id=pad_idx)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_loss = 0.0
    n_batches  = 0
    model.eval()
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask = model.make_padding_mask(src)
            tgt_mask = model.make_padding_mask(tgt)
            logits   = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            loss     = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            total_loss += loss.item()
            n_batches  += 1
    model.train()
    return total_loss / max(n_batches, 1)


def run_training_loop(
    *,
    model,
    tokenizer,
    cleaned_dir:      Path,
    device:           "torch.device",
    train_cfg:        "TrainingConfig",
    checkpoint_mgr,
    resource_handle,
    metrics_logger:   "MetricsLogger",
    unified_log,
    progress,
    run_id:           str = "",
    model_id:         str = "dgb1",
) -> float:
    """
    Execute the full training loop.
    Returns best_loss achieved during this run.
    """
    from trainer.core.dataset_loader import StreamingTextDataset, build_streaming_loader
    from modules.utils.error_handler import GradientError
    from modules.logging_config import TrainingLogger, set_log_stage
    from modules.utils.streaming import StreamEvent

    set_log_stage("model_training")
    hub      = get_training_hub()
    mem_mgr  = MemoryManager(device=str(device))
    loss_ema = MovingAverage(alpha=0.1)
    tlog     = TrainingLogger()

    # FIX T4 — apply seed for reproducibility
    torch.manual_seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)
    torch.backends.cudnn.deterministic = True
    logger.info("Seed applied: %d", train_cfg.seed)

    # Mixed precision scaler
    scaler = None
    if train_cfg.mixed_precision and "cuda" in str(device):
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Mixed precision (fp16) enabled")

    # FIX B4 — LR from config, not hardcoded
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,       # FIX B4
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = WarmupScheduler(
        optimizer, d_model=train_cfg.d_model, warmup_steps=train_cfg.warmup_steps
    )
    criterion = nn.CrossEntropyLoss(ignore_index=train_cfg.pad_idx)

    # Restore optimizer + scheduler from checkpoint
    ckpt_path = checkpoint_mgr.latest_model_checkpoint()
    if ckpt_path and progress.global_step > 0:
        from transformer.utils.model_helpers import load_checkpoint
        load_checkpoint(ckpt_path, model, optimizer, device=device)
        scheduler.restore(progress.global_step)
        logger.info("Optimizer + scheduler restored  step=%d  epoch=%d",
                    progress.global_step, progress.epoch)

    # Train/val split
    train_files, val_files = split_cleaned_files(
        cleaned_dir, val_fraction=train_cfg.validation_split, seed=train_cfg.seed
    )
    logger.info("Dataset split: %d train files  %d val files", len(train_files), len(val_files))

    start_epoch   = progress.epoch
    global_step   = progress.global_step
    best_loss     = progress.best_loss
    prev_batch_sz = resource_handle.batch_size

    logger.info(
        "Training: epochs=%d  start=%d  device=%s  lr=%.2e  warmup=%d",
        train_cfg.epochs, start_epoch, device,
        train_cfg.learning_rate, train_cfg.warmup_steps,
    )

    for epoch in range(start_epoch + 1, train_cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        t_epoch    = time.time()
        set_log_stage("model_training")

        cur_batch_sz = resource_handle.batch_size
        if cur_batch_sz != prev_batch_sz:
            logger.info("Batch size: %d → %d (pressure=%s)",
                        prev_batch_sz, cur_batch_sz, resource_handle.pressure)
            prev_batch_sz = cur_batch_sz

        tlog.epoch_start(epoch, train_cfg.epochs,
                         scheduler.current_lr, cur_batch_sz, resource_handle.grad_accum)

        def _on_line(fi, li):
            progress.clean_line_idx = li
            progress.clean_file_idx = fi

        dataset = StreamingTextDataset(
            tokenizer=tokenizer,
            cleaned_dir=cleaned_dir,
            max_seq_len=512,
            start_file_idx=0,
            start_line_idx=0,
            on_line=_on_line,
        )
        # Restrict to train files only
        dataset._files = sorted(train_files)

        loader = build_streaming_loader(
            dataset,
            batch_size=cur_batch_sz,
            num_workers=resource_handle.num_workers,
            pin_memory=resource_handle.pin_memory,
            pad_id=train_cfg.pad_idx,
        )

        start_batch = progress.batch_idx if epoch == start_epoch + 1 else 0
        optimizer.zero_grad()

        for batch_idx, (src, tgt) in enumerate(loader):
            if batch_idx < start_batch:
                continue

            accum = resource_handle.grad_accum
            src, tgt = src.to(device), tgt.to(device)
            src_mask = model.make_padding_mask(src)
            tgt_mask = model.make_padding_mask(tgt)

            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
                    loss   = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1)) / accum
                scaler.scale(loss).backward()
            else:
                logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
                loss   = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1)) / accum
                loss.backward()

            loss_val    = loss.item() * accum
            epoch_loss += loss_val
            n_batches  += 1
            ema_loss    = loss_ema.update(loss_val)

            is_update = (batch_idx + 1) % accum == 0
            if is_update:
                if scaler:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip).item()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip).item()
                    optimizer.step()

                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    msg = f"NaN/Inf gradient  epoch={epoch}  batch={batch_idx}"
                    logger.critical(msg)
                    raise GradientError(msg)

                lr = scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                checkpoint_mgr.on_batch_done(progress, epoch, batch_idx, global_step, loss_val)

                if global_step % train_cfg.log_every_steps == 0:
                    unified_log.batch(
                        epoch=epoch, batch=batch_idx, step=global_step,
                        loss=loss_val, lr=lr, grad_norm=grad_norm,
                        ema_loss=ema_loss, batch_size=cur_batch_sz, accum=accum,
                        pressure=resource_handle.pressure,
                        ram_avail=resource_handle.ram_available_gb,
                    )
                    metrics_logger.log_step(
                        epoch=epoch, step=global_step,
                        loss=loss_val, lr=lr, grad_norm=grad_norm,
                    )
                    tlog.step(global_step, loss_val, lr, grad_norm, ema_loss)
                    hub.publish(StreamEvent.metric(
                        epoch=epoch, step=global_step,
                        loss=round(loss_val, 4), lr=round(lr, 6),
                        grad_norm=round(grad_norm, 4), ema_loss=round(ema_loss, 4),
                    ))

                # Validation eval
                if train_cfg.eval_every_steps > 0 and global_step % train_cfg.eval_every_steps == 0:
                    val_loss = _eval_pass(model, tokenizer, val_files, device,
                                         train_cfg.pad_idx, cur_batch_sz, train_cfg.num_workers)
                    if val_loss is not None:
                        ppl = math.exp(min(val_loss, 20))
                        tlog.eval(epoch, val_loss, ppl)
                        metrics_logger.log_step(epoch=epoch, step=global_step,
                                                loss=loss_val, lr=lr, val_loss=val_loss)
                        hub.publish(StreamEvent.metric(
                            epoch=epoch, step=global_step, val_loss=round(val_loss, 4),
                            perplexity=round(ppl, 2),
                        ))

            mem_mgr.check(global_step)

        # ── Epoch end ──────────────────────────────────────────────────
        avg_loss     = epoch_loss / max(n_batches, 1)
        duration_sec = time.time() - t_epoch
        progress.global_step = global_step

        val_loss = _eval_pass(model, tokenizer, val_files, device,
                              train_cfg.pad_idx, cur_batch_sz, train_cfg.num_workers)

        tlog.epoch_end(epoch, train_cfg.epochs, avg_loss,
                       min(avg_loss, best_loss), duration_sec)

        unified_log.epoch(
            epoch=epoch, avg_loss=avg_loss, best_loss=min(avg_loss, best_loss),
            duration_sec=duration_sec, n_batches=n_batches,
            ema_loss=loss_ema.value or avg_loss, batch_size=cur_batch_sz,
            val_loss=val_loss,
        )

        metrics_logger.log_epoch(epoch, avg_loss, val_loss=val_loss, duration_sec=duration_sec)
        checkpoint_mgr.on_epoch_done(progress, epoch, avg_loss, model, optimizer)

        if avg_loss < best_loss:
            best_loss = avg_loss

        hub.publish(StreamEvent.progress(
            phase="model_training", epoch=epoch, total_epochs=train_cfg.epochs,
            avg_loss=round(avg_loss, 4), val_loss=round(val_loss, 4) if val_loss else None,
            best_loss=round(best_loss, 4), duration_sec=round(duration_sec, 1),
        ))

        progress.batch_idx = 0

    unified_log.pipeline(
        f"Training complete  best_loss={best_loss:.4f}  total_steps={global_step}",
        stage="model_training",
    )
    return best_loss
