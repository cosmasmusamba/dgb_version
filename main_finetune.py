#!/usr/bin/env python
"""
main_finetune.py
================
Fine-tune a pre-trained DGB model on custom instruction data.

Usage:
    python main_finetune.py --train datasets/finetune/chat_data.jsonl
    python main_finetune.py --train data.jsonl --lora --epochs 5
"""

import argparse
import logging
import torch
from pathlib import Path

from configs.loader import get_config
from modules.utils.path_resolver import init_path_resolver
from tokenizer.dgb_tokenizer import DGBTokenizer
from transformer.core.transformer_model import DGBTransformer
from transformer.utils.model_helpers import load_model
from finetune.finetuner import ModelFineTuner, FineTuningConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_sample_data():
    """Create sample fine-tuning data if none exists"""
    sample_data = Path("datasets/finetune/sample_chat.jsonl")
    
    if sample_data.exists():
        return sample_data
    
    sample_data.parent.mkdir(parents=True, exist_ok=True)
    
    examples = [
        {"instruction": "What is artificial intelligence?", 
         "response": "Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans."},
        {"instruction": "Explain machine learning in simple terms",
         "response": "Machine learning is a type of AI that allows computers to learn from data without being explicitly programmed. It's like teaching a child by showing them many examples."},
        {"instruction": "What are the benefits of renewable energy?",
         "response": "Renewable energy sources like solar and wind are sustainable, reduce pollution, create jobs, and help fight climate change."},
        {"instruction": "How does a neural network work?",
         "response": "A neural network is a computer system modeled after the human brain. It has layers of interconnected nodes that process information and learn patterns from data."}
    ]
    
    import json
    with open(sample_data, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"✅ Created sample data at {sample_data}")
    return sample_data


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DGB model")
    parser.add_argument("--train", type=str, help="Training data path (jsonl)")
    parser.add_argument("--val", type=str, help="Validation data path (optional)")
    parser.add_argument("--output", type=str, default="checkpoints/dgb1/finetuned", 
                       help="Output directory")
    parser.add_argument("--lora", action="store_true", help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--freeze", type=int, default=0, help="Number of layers to freeze")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--task", type=str, default="chat", 
                       choices=["chat", "conversation", "completion"], help="Task type")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎯 DGB Fine-Tuning")
    print("="*60)
    
    # Get or create training data
    train_path = Path(args.train) if args.train else prepare_sample_data()
    if not train_path.exists():
        print(f"❌ Training data not found: {train_path}")
        return
    
    # Load config and model
    cfg = get_config()
    model_id = cfg.project.model_id
    path_resolver = init_path_resolver(model_id=model_id, cfg=cfg)
    
    # Load tokenizer
    tokenizer_dir = path_resolver.tokenizer_dir(create=False)
    if not tokenizer_dir.exists():
        print(f"❌ Tokenizer not found. Train tokenizer first:")
        print(f"   python main_train_tokenizer.py")
        return
    
    tokenizer = DGBTokenizer.from_pretrained(tokenizer_dir)
    print(f"✅ Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    # Load pre-trained model
    models_dir = path_resolver.models_dir(create=False)
    model_files = sorted(models_dir.glob("*_best_model.pt"))
    if not model_files:
        model_files = sorted(models_dir.glob("*_epoch_*.pt"))
    
    if not model_files:
        print(f"❌ No pre-trained model found. Train model first:")
        print(f"   python model_trainer.py")
        return
    
    latest_model = model_files[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Loading model from {latest_model.name}")
    
    model = load_model(str(latest_model), device, vocab_size=tokenizer.vocab_size)
    
    # Fine-tuning config
    ft_config = FineTuningConfig({
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'use_lora': args.lora,
        'freeze_layers': args.freeze,
        'task_type': args.task,
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'gradient_accumulation_steps': 4,
        'lora_r': 8,
        'lora_alpha': 32,
        'save_every_epochs': 1
    })
    
    # Create fine-tuner
    finetuner = ModelFineTuner(model, tokenizer, ft_config)
    
    # Run fine-tuning
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    val_path = Path(args.val) if args.val else None
    
    finetuner.fine_tune(
        train_data_path=train_path,
        val_data_path=val_path,
        output_dir=output_dir
    )
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"   Model saved to: {output_dir}")
    print(f"\nTest with:")
    print(f"   python -c \"from main_finetune import test; test()\"")


def test():
    """Quick test function for fine-tuned model"""
    import torch
    from configs.loader import get_config
    from modules.utils.path_resolver import init_path_resolver
    from tokenizer.dgb_tokenizer import DGBTokenizer
    from transformer.core.transformer_model import DGBTransformer
    from trainer.finetune.finetuner import ModelFineTuner, FineTuningConfig
    
    cfg = get_config()
    model_id = cfg.project.model_id
    path_resolver = init_path_resolver(model_id=model_id, cfg=cfg)
    
    tokenizer = DGBTokenizer.from_pretrained(path_resolver.tokenizer_dir())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load fine-tuned model
    ft_path = Path("checkpoints/dgb1/finetuned/finetuned_best.pt")
    if not ft_path.exists():
        ft_path = Path("checkpoints/dgb1/finetuned/finetuned_epoch_3.pt")
    
    if ft_path.exists():
        checkpoint = torch.load(ft_path, map_location='cpu')
        
        transformer_cfg = cfg.transformer
        model = DGBTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=transformer_cfg.d_model,
            n_heads=transformer_cfg.n_heads,
            n_encoder_layers=transformer_cfg.n_encoder_layers,
            n_decoder_layers=transformer_cfg.n_decoder_layers,
            d_ff=transformer_cfg.d_ff,
            dropout=transformer_cfg.dropout,
            max_seq_len=transformer_cfg.max_seq_len,
            pad_idx=transformer_cfg.pad_idx,
            tie_embeddings=transformer_cfg.tie_embeddings,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print("\n🤖 Testing fine-tuned model:")
        print("="*50)
        
        test_prompts = [
            "What is artificial intelligence?",
            "Explain machine learning",
            "What are the benefits of AI?"
        ]
        
        for prompt in test_prompts:
            # Simple generation
            input_ids = tokenizer.encode(f"<BOS>Instruction: {prompt}\n\nResponse: ")
            src = torch.tensor([input_ids], dtype=torch.long, device=device)
            
            with torch.no_grad():
                output = model.greedy_decode(src, bos_id=2, eos_id=3, max_len=100)[0].tolist()
            
            response = tokenizer.decode(output, skip_special_tokens=True)
            print(f"\n📝 {prompt}")
            print(f"✨ {response[:200]}...")
    else:
        print("No fine-tuned model found. Run fine-tuning first.")

if __name__ == "__main__":
    main()