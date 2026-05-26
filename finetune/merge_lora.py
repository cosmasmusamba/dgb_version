#!/usr/bin/env python3
"""
Merge a LoRA fine‑tuned checkpoint into a plain DGBTransformer model.
Uses your existing LoRAAdapter without triggering automatic base model load.
"""
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import logging
from configs.loader import get_config
from modules.utils.path_resolver import get_path_resolver
from transformer.core.transformer_model import DGBTransformer
from transformer.utils.model_helpers import save_checkpoint
from tokenizer.dgb_tokenizer import DGBTokenizer
from finetune.core.lora_adapter import LoRAAdapter

logger = logging.getLogger(__name__)

def main():
    cfg = get_config()
    resolver = get_path_resolver(cfg.project.model_id, cfg)

    # Path to the LoRA fine‑tuned checkpoint
    lora_ckpt = resolver.models_dir() / "20260526115118_finetune_complete.pt"
    if not lora_ckpt.exists():
        logger.warning(f"LoRA checkpoint not found: {lora_ckpt}")
        return 1

    # Load tokenizer for vocab size
    tokenizer = DGBTokenizer.from_pretrained(resolver.tokenizer_dir())
    vocab_size = tokenizer.vocab_size

    # Create a clean base model (plain DGBTransformer)
    base_model = DGBTransformer(
        vocab_size=vocab_size,
        d_model=cfg.transformer.d_model,
        n_heads=cfg.transformer.n_heads,
        n_encoder_layers=cfg.transformer.n_encoder_layers,
        n_decoder_layers=cfg.transformer.n_decoder_layers,
        d_ff=cfg.transformer.d_ff,
        dropout=cfg.transformer.dropout,
        max_seq_len=cfg.transformer.max_seq_len,
        pad_idx=cfg.transformer.pad_idx,
        tie_embeddings=cfg.transformer.tie_embeddings,
    )

    # Create LoRA adapter with the base model (no automatic checkpoint load)
    adapter = LoRAAdapter(config=cfg, model=base_model)   # pass model explicitly
    # Load the LoRA fine‑tuned checkpoint (contains LoRA weights)
    adapter.load_checkpoint(lora_ckpt, load_optimizer=False)

    # Merge LoRA weights into the base linear layers
    adapter.merge_and_unload()

    # Save the merged model as a plain checkpoint
    merged_path = resolver.models_dir() / "dgb1_finetuned_merged.pt"
    save_checkpoint(
        path=merged_path,
        model=adapter.model,
        epoch=3,                     # adjust as needed
        loss=0.2276,
        extra={"note": "Merged from LoRA finetune run 20260526115118"},
        ctx=None,
    )
    logger.info(f"Merged model saved to {merged_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())