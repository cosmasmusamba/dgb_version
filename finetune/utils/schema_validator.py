"""
finetune/utils/schema_validator.py
Validate dataset schema for fine-tuning.
Supports multiple formats: instruction/output, input/output, text, or messages.
"""
from typing import Dict, Any, List, Iterable


def validate_dataset(entries: Iterable[Dict[str, Any]], required_fields: List[str] = None) -> None:
    """
    Validate dataset entries for fine-tuning.
    
    Supported formats:
    - instruction + output (chat format)
    - input + output (legacy format)
    - text (single field for continued pretraining)
    - messages (conversation format)
    
    Args:
        entries: Iterable of dataset entries
        required_fields: Override default required fields
    
    Raises:
        ValueError: If validation fails
    """
    entries_list = list(entries)
    if not entries_list:
        raise ValueError("Dataset is empty")
    
    # Check first entry to determine format
    first_entry = entries_list[0]
    
    # Supported field patterns
    instruction_output = "instruction" in first_entry and "output" in first_entry
    input_output = "input" in first_entry and "output" in first_entry
    text_only = "text" in first_entry
    messages_format = "messages" in first_entry
    
    if instruction_output:
        required = {"instruction", "output"}
        format_type = "instruction/output"
    elif input_output:
        required = {"input", "output"}
        format_type = "input/output"
    elif text_only:
        required = {"text"}
        format_type = "text"
    elif messages_format:
        required = {"messages"}
        format_type = "messages"
    else:
        available = set(first_entry.keys())
        raise ValueError(
            f"Dataset entry missing required fields. "
            f"Expected one of: instruction+output, input+output, text, or messages. "
            f"Got: {available}"
        )
    
    # Validate all entries have the required fields
    for i, entry in enumerate(entries_list):
        missing = required - set(entry.keys())
        if missing:
            raise ValueError(f"Dataset entry {i} missing fields: {missing}")
        
        # Additional validation for non-empty values
        for field in required:
            if not entry.get(field):
                raise ValueError(f"Dataset entry {i} has empty {field} field")
    
    # Log validation results
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        f"Dataset validation passed: format={format_type}, "
        f"entries={len(entries_list)}, fields={required}"
    )


def convert_dataset_format(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert dataset entry to unified format for training.
    
    Returns dict with:
    - input_ids: tokenized input text
    - labels: tokenized output text
    """
    if "instruction" in entry and "output" in entry:
        # Chat format: combine instruction and output
        return {
            "input": entry["instruction"],
            "output": entry["output"],
        }
    elif "input" in entry and "output" in entry:
        # Already in correct format
        return entry
    elif "text" in entry:
        # For continued pretraining
        return {
            "input": entry["text"],
            "output": entry["text"],  # Self-supervised
        }
    elif "messages" in entry:
        # Conversation format - convert to string
        messages = entry["messages"]
        input_text = "\n".join([m["content"] for m in messages if m["role"] == "user"])
        output_text = "\n".join([m["content"] for m in messages if m["role"] == "assistant"])
        return {
            "input": input_text,
            "output": output_text,
        }
    else:
        return entry