"""
trainer/finetune/data_generator.py
=====================================
Standalone fine-tuning dataset generator from your own text corpus.
No external dependencies - uses your existing training data.

Creates instruction-response pairs by:
1. Extracting key sentences from your training corpus
2. Generating natural language instructions from text patterns
3. Creating Q&A pairs from factual statements
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

import logging
logger = logging.getLogger(__name__)


class InstructionGenerator:
    """
    Generate instruction-response pairs from raw text.
    No AI dependencies - uses pattern matching and text transformations.
    """
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Define instruction templates for different text patterns"""
        
        # Question templates for factual statements
        self.fact_templates = [
            "What is {entity}?",
            "Explain {entity} in simple terms",
            "What do you know about {entity}?",
            "Tell me about {entity}",
            "Can you describe {entity}?",
            "What does {entity} mean?",
            "Define {entity}"
        ]
        
        # Definition patterns
        self.definition_patterns = [
            (r"(\w+) is (?:a|an|the) (.+?)(?:\.|;|, and)", "What is {entity}?"),
            (r"(\w+) refers to (.+?)(?:\.|;)", "What does {entity} refer to?"),
            (r"(\w+) means (.+?)(?:\.|;)", "What does {entity} mean?"),
            (r"(\w+) are (.+?)(?:\.|;)", "What are {entity}?"),
        ]
        
        # Causal patterns
        self.causal_patterns = [
            (r"(\w+) causes (.+?)(?:\.|;)", "What causes {entity}?"),
            (r"(\w+) leads to (.+?)(?:\.|;)", "What does {entity} lead to?"),
            (r"(\w+) results in (.+?)(?:\.|;)", "What results from {entity}?"),
        ]
        
        # Example templates for instructions
        self.instruction_templates = [
            "Write a short explanation of {topic}",
            "Summarize the key points about {topic}",
            "What are the main characteristics of {topic}?",
            "Explain how {topic} works",
            "Why is {topic} important?",
            "Describe the benefits of {topic}",
            "What are the different types of {topic}?",
            "Compare and contrast aspects of {topic}",
            "What should someone know about {topic}?",
            "Provide an overview of {topic}"
        ]
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract potential entities (capitalized words/phrases) from text"""
        entities = []
        
        # Extract capitalized phrases (potential entities)
        capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        entities.extend([c for c in capitalized if len(c) > 2 and len(c.split()) <= 4])
        
        # Extract common noun phrases (simple heuristic)
        noun_phrases = re.findall(r'\b(?:the|a|an)\s+([a-z]+(?:\s+[a-z]+){0,3})\b', text.lower())
        entities.extend([np for np in noun_phrases if len(np.split()) <= 3 and len(np) > 4])
        
        # Remove duplicates
        seen = set()
        unique_entities = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                unique_entities.append(e)
        
        return unique_entities[:10]  # Limit to 10 per sentence
    
    def extract_key_sentences(self, text: str, max_sentences: int = 5) -> List[str]:
        """Extract important sentences from text"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20 and len(s.strip()) < 300]
        
        # Score sentences by length and keyword presence
        keyword_weights = ['important', 'key', 'essential', 'primarily', 'typically', 
                          'usually', 'often', 'significant', 'major', 'main']
        
        scored = []
        for s in sentences:
            score = len(s)
            for kw in keyword_weights:
                if kw in s.lower():
                    score += 50
            scored.append((score, s))
        
        scored.sort(reverse=True)
        return [s for _, s in scored[:max_sentences]]
    
    def generate_from_fact(self, text: str) -> Optional[Tuple[str, str]]:
        """Generate Q&A from factual statements"""
        for pattern, template in self.definition_patterns:
            match = re.search(pattern, text)
            if match:
                entity = match.group(1)
                definition = match.group(2)
                question = template.format(entity=entity)
                response = f"{entity} is {definition}."
                return (question, response)
        return None
    
    def generate_from_causal(self, text: str) -> Optional[Tuple[str, str]]:
        """Generate Q&A from causal relationships"""
        for pattern, template in self.causal_patterns:
            match = re.search(pattern, text)
            if match:
                entity = match.group(1)
                effect = match.group(2)
                question = template.format(entity=entity)
                response = f"{entity} leads to {effect}."
                return (question, response)
        return None
    
    def generate_from_topics(self, sentences: List[str], entities: List[str]) -> List[Tuple[str, str]]:
        """Generate examples from extracted topics"""
        examples = []
        
        for entity in entities[:5]:  # Use top entities
            for template in self.instruction_templates[:3]:  # Use a few templates
                instruction = template.format(topic=entity)
                
                # Find relevant sentence that contains the entity
                response = None
                for sent in sentences:
                    if entity.lower() in sent.lower():
                        response = sent
                        break
                
                if response:
                    examples.append((instruction, response))
        
        return examples
    
    def generate_from_text(self, text: str, max_examples_per_text: int = 10) -> List[Dict]:
        """
        Generate instruction-response pairs from a text
        """
        examples = []
        
        # Clean and prepare text
        text = text.strip()
        if len(text) < 100:
            return examples
        
        # Extract components
        entities = self.extract_entities(text)
        key_sentences = self.extract_key_sentences(text)
        
        # Method 1: Extract definitions
        fact_example = self.generate_from_fact(text)
        if fact_example:
            question, response = fact_example
            examples.append({
                "instruction": question,
                "response": response,
                "source": "definition"
            })
        
        # Method 2: Extract causal relationships
        causal_example = self.generate_from_causal(text)
        if causal_example:
            question, response = causal_example
            examples.append({
                "instruction": question,
                "response": response,
                "source": "causal"
            })
        
        # Method 3: Generate from topics and sentences
        topic_examples = self.generate_from_topics(key_sentences, entities)
        for instruction, response in topic_examples:
            if len(examples) < max_examples_per_text:
                examples.append({
                    "instruction": instruction,
                    "response": response,
                    "source": "topic"
                })
        
        # Method 4: Simple Q&A from key sentences
        for sent in key_sentences[:3]:
            if len(examples) < max_examples_per_text:
                # Create a simple instruction
                words = sent.split()[:10]
                preview = ' '.join(words) + "..."
                examples.append({
                    "instruction": f"Explain: {preview}",
                    "response": sent,
                    "source": "extract"
                })
        
        return examples


class FineTuneDatasetBuilder:
    """
    Build fine-tuning dataset from your own text corpus.
    No external dependencies - uses only your training data.
    """
    
    def __init__(self, tokenizer=None):
        self.generator = InstructionGenerator(tokenizer)
    
    def build_from_cleaned_data(
        self,
        cleaned_dir: Path,
        output_dir: Path,
        max_files: int = 100,
        max_examples_per_file: int = 20,
        min_text_length: int = 200
    ) -> Dict:
        """
        Build fine-tuning dataset from cleaned text files
        
        Args:
            cleaned_dir: Directory containing cleaned .txt files
            output_dir: Output directory for JSONL files
            max_files: Maximum number of files to process
            max_examples_per_file: Max examples per text file
            min_text_length: Minimum text length to process
        """
        cleaned_dir = Path(cleaned_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all cleaned text files
        text_files = sorted(cleaned_dir.glob("*.txt"))
        if not text_files:
            raise FileNotFoundError(f"No cleaned text files found in {cleaned_dir}")
        
        # Limit files
        text_files = text_files[:max_files]
        logger.info(f"Processing {len(text_files)} cleaned files")
        
        all_examples = []
        stats = {
            "files_processed": 0,
            "files_skipped_too_short": 0,
            "total_examples": 0,
            "examples_by_source": Counter()
        }
        
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                if len(text) < min_text_length:
                    stats["files_skipped_too_short"] += 1
                    continue
                
                # Generate examples
                examples = self.generator.generate_from_text(text, max_examples_per_file)
                
                if examples:
                    all_examples.extend(examples)
                    stats["files_processed"] += 1
                    
                    for ex in examples:
                        stats["examples_by_source"][ex.get("source", "unknown")] += 1
                    
                    logger.debug(f"Generated {len(examples)} examples from {file_path.name}")
                
            except Exception as e:
                logger.warning(f"Failed to process {file_path.name}: {e}")
        
        # Shuffle examples
        random.shuffle(all_examples)
        stats["total_examples"] = len(all_examples)
        
        # Split into train/val
        split_idx = int(len(all_examples) * 0.9)
        train_examples = all_examples[:split_idx]
        val_examples = all_examples[split_idx:]
        
        # Save datasets
        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"
        
        with open(train_path, 'w', encoding='utf-8') as f:
            for ex in train_examples:
                # Remove source field for training
                clean_ex = {k: v for k, v in ex.items() if k != 'source'}
                f.write(json.dumps(clean_ex) + '\n')
        
        with open(val_path, 'w', encoding='utf-8') as f:
            for ex in val_examples:
                clean_ex = {k: v for k, v in ex.items() if k != 'source'}
                f.write(json.dumps(clean_ex) + '\n')
        
        # Save statistics
        stats_path = output_dir / "dataset_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset built: {len(train_examples)} train, {len(val_examples)} val")
        
        return stats
    
    def create_sample_instruction_examples(self, output_dir: Path) -> Path:
        """
        Create sample instruction examples for demonstration/testing
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pre-defined high-quality examples that work with any training data
        sample_examples = [
            {
                "instruction": "What is a seizure?",
                "response": "A seizure is a sudden, uncontrolled electrical disturbance in the brain that can cause changes in behavior, movements, feelings, and consciousness."
            },
            {
                "instruction": "Explain how the brain works",
                "response": "The brain is the control center of the body. It receives signals from sensory organs and sends instructions to muscles and organs through neurons, which communicate via electrical and chemical signals."
            },
            {
                "instruction": "What are the symptoms of epilepsy?",
                "response": "Epilepsy symptoms vary but can include temporary confusion, staring spells, uncontrollable jerking movements, loss of consciousness, and fear or anxiety."
            },
            {
                "instruction": "How is a seizure diagnosed?",
                "response": "Seizures are diagnosed through medical history review, neurological examination, EEG (electroencephalogram) to record brain activity, and sometimes MRI or CT scans to identify brain abnormalities."
            },
            {
                "instruction": "What treatments are available for epilepsy?",
                "response": "Epilepsy treatments include anti-seizure medications (AEDs), ketogenic diet, vagus nerve stimulation (VNS), and in some cases, brain surgery to remove the area causing seizures."
            }
        ]
        
        output_file = output_dir / "sample_instructions.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for ex in sample_examples:
                f.write(json.dumps(ex) + '\n')
        
        logger.info(f"Created {len(sample_examples)} sample examples at {output_file}")
        return output_file


def build_finetune_dataset_from_corpus(
    cleaned_dir: Path,
    output_dir: Path,
    max_files: int = 50
) -> Path:
    """
    Main function to build fine-tuning dataset from your training corpus
    
    Usage:
        from trainer.finetune.data_generator import build_finetune_dataset_from_corpus
        
        build_finetune_dataset_from_corpus(
            cleaned_dir=Path("datasets/dgb1/cleaned"),
            output_dir=Path("datasets/finetune/from_corpus")
        )
    """
    builder = FineTuneDatasetBuilder()
    
    stats = builder.build_from_cleaned_data(
        cleaned_dir=cleaned_dir,
        output_dir=output_dir,
        max_files=max_files
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Files processed: {stats['files_processed']}")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Examples by source: {dict(stats['examples_by_source'])}")
    
    return output_dir