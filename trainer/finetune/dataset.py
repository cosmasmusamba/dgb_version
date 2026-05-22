"""
trainer/finetune/dataset.py
============================
Fine-tuning dataset with instruction-response formatting
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class InstructionDataset(Dataset):
    """
    Dataset for instruction fine-tuning.
    
    Supports:
    - Chat format: {"instruction": "...", "response": "..."}
    - Conversation format: {"messages": [{"role": "user", "content": "..."}, ...]}
    - Alpaca format: {"instruction": "...", "input": "...", "output": "..."}
    """
    
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_length: int = 512,
        task_type: str = "chat",
        add_special_tokens: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.add_special_tokens = add_special_tokens
        
        self.data = self._load_data(data_path)
        logger.info(f"Loaded {len(self.data)} fine-tuning examples from {data_path}")
    
    def _load_data(self, data_path: Path) -> List[Dict]:
        """Load data from JSONL or JSON file"""
        data = []
        
        if data_path.suffix == '.jsonl':
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        elif data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    data = loaded
                else:
                    data = [loaded]
        else:
            # Try to read as text file (one example per line)
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        data.append({"instruction": line.strip(), "response": ""})
        
        return data
    
    def _format_example(self, example: Dict) -> str:
        """Format example based on task type"""
        
        if self.task_type == "chat":
            # Standard instruction-response format
            instruction = example.get('instruction', '')
            response = example.get('response', '')
            input_text = example.get('input', '')
            
            if input_text:
                text = f"<BOS>Instruction: {instruction}\n\nInput: {input_text}\n\nResponse: {response}<EOS>"
            else:
                text = f"<BOS>Instruction: {instruction}\n\nResponse: {response}<EOS>"
        
        elif self.task_type == "conversation":
            # Multi-turn conversation format
            messages = example.get('messages', [])
            text_parts = ["<BOS>"]
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                text_parts.append(f"{role}: {content}\n")
            text_parts.append("<EOS>")
            text = ''.join(text_parts)
        
        elif self.task_type == "completion":
            # Simple text completion (predict next tokens)
            text = example.get('text', '')
            if not text:
                text = f"<BOS>{example.get('prompt', '')}<EOS>"
        
        else:
            # Default: just use response
            text = example.get('response', example.get('text', ''))
        
        return text
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Format text
        text = self._format_example(example)
        
        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=self.add_special_tokens
        )
        
        # Create input_ids and labels (for causal LM)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        
        # Optional: Mask out instruction part (only train on response)
        # This requires tracking token positions - simpler to train on all
        
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


def create_finetune_dataloader(
    data_path: Path,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    task_type: str = "chat",
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create DataLoader for fine-tuning"""
    
    dataset = InstructionDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        task_type=task_type
    )
    
    def collate_fn(batch):
        """Pad sequences in batch"""
        max_len = max(item['input_ids'].size(0) for item in batch)
        
        input_ids = []
        labels = []
        attention_masks = []
        
        for item in batch:
            # Pad input_ids
            pad_len = max_len - item['input_ids'].size(0)
            input_ids.append(torch.cat([item['input_ids'], torch.zeros(pad_len, dtype=torch.long)]))
            
            # Pad labels (use -100 for ignore)
            labels.append(torch.cat([item['labels'], torch.full((pad_len,), -100, dtype=torch.long)]))
            
            # Pad attention masks
            attention_masks.append(torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)]))
        
        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'attention_mask': torch.stack(attention_masks)
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )