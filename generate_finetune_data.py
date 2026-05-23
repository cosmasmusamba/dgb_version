#!/usr/bin/env python
"""
generate_finetune_data.py
==========================
Generate fine-tuning data from your own training corpus.
No external AI dependencies - uses pattern matching on your existing data.

Usage:
    python generate_finetune_data.py
    python generate_finetune_data.py --max-files 100 --examples-per-file 15
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from configs.loader import get_config
from modules.utils.path_resolver import init_path_resolver
from finetune.data_generator import build_finetune_dataset_from_corpus, FineTuneDatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Generate fine-tuning data from training corpus")
    parser.add_argument("--max-files", type=int, default=50, 
                       help="Maximum number of cleaned files to process")
    parser.add_argument("--examples-per-file", type=int, default=20,
                       help="Maximum examples to generate per file")
    parser.add_argument("--min-text-length", type=int, default=200,
                       help="Minimum text length to process (characters)")
    parser.add_argument("--sample-only", action="store_true",
                       help="Only create sample examples, don't process corpus")
    parser.add_argument("--output", type=str, default="datasets/finetune",
                       help="Output directory for generated data")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("📚 DGB Fine-Tuning Data Generator (Standalone)")
    print("="*60)
    print("No external AI dependencies - using your training corpus")
    print("="*60)
    
    cfg = get_config()
    model_id = cfg.project.model_id
    path_resolver = init_path_resolver(model_id=model_id, cfg=cfg)
    
    cleaned_dir = path_resolver.cleaned_dir(create=False)
    output_dir = Path(args.output)
    
    if args.sample_only:
        # Create sample examples only
        builder = FineTuneDatasetBuilder()
        sample_path = builder.create_sample_instruction_examples(output_dir / "sample")
        print(f"\n✅ Created sample examples at: {sample_path}")
        print(f"\nRun fine-tuning with:")
        print(f"  python main_finetune.py --train {sample_path}")
        
    else:
        # Build dataset from corpus
        if not cleaned_dir.exists():
            print(f"\n❌ Cleaned directory not found: {cleaned_dir}")
            print("   Run data pipeline first:")
            print("   python main_pipeline.py --stage dataset_clean")
            return
        
        # Check if there are any cleaned files
        txt_files = list(cleaned_dir.glob("*.txt"))
        if not txt_files:
            print(f"\n❌ No cleaned .txt files found in {cleaned_dir}")
            print("   Run data pipeline first:")
            print("   python main_pipeline.py --stage dataset_clean")
            return
        
        print(f"\n📁 Found {len(txt_files)} cleaned files in {cleaned_dir}")
        print(f"📁 Output directory: {output_dir}")
        print(f"\n⚙️  Settings:")
        print(f"   Max files: {args.max_files}")
        print(f"   Max examples per file: {args.examples_per_file}")
        print(f"   Min text length: {args.min_text_length}")
        print("\n🔄 Generating instruction-response pairs...")
        
        # Override generator settings
        generator = FineTuneDatasetBuilder()
        generator.generator._max_examples_per_file = args.examples_per_file
        
        stats = generator.build_from_cleaned_data(
            cleaned_dir=cleaned_dir,
            output_dir=output_dir,
            max_files=args.max_files,
            max_examples_per_file=args.examples_per_file,
            min_text_length=args.min_text_length
        )
        
        print(f"\n✅ Generation complete!")
        print(f"\n📊 Statistics:")
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Total examples: {stats['total_examples']}")
        print(f"   Train examples: {int(stats['total_examples'] * 0.9)}")
        print(f"   Validation examples: {int(stats['total_examples'] * 0.1)}")
        
        if stats['examples_by_source']:
            print(f"\n   Examples by source:")
            for source, count in stats['examples_by_source'].items():
                print(f"      {source}: {count}")
        
        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"
        
        print(f"\n📁 Generated files:")
        print(f"   Train: {train_path}")
        print(f"   Validation: {val_path}")
        print(f"   Stats: {output_dir / 'dataset_stats.json'}")
        
        print(f"\n🚀 Run fine-tuning with:")
        print(f"   python main_finetune.py --train {train_path} --val {val_path} --lora")


if __name__ == "__main__":
    main()