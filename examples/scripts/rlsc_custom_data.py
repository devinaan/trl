#!/usr/bin/env python3
"""
Advanced RLSC example with custom data preparation.

This example demonstrates:
1. Creating custom prompt-only datasets
2. Converting conversational data to prompt format
3. Data preprocessing and validation
4. Using different confidence calculation methods

Usage:
python examples/scripts/rlsc_custom_data.py
"""

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import RLSCTrainer, RLSCConfig

def create_custom_prompt_dataset():
    """Create a custom prompt-only dataset."""
    prompts = [
        "Explain the concept of machine learning in simple terms:",
        "Write a short story about a robot learning to paint:",
        "Describe the benefits of renewable energy:",
        "What are the key principles of good software design?",
        "How does photosynthesis work?",
        "Compare and contrast classical and quantum computing:",
        "Explain the importance of biodiversity:",
        "What are the main causes of climate change?",
        "Describe the process of protein synthesis:",
        "How do neural networks learn from data?",
    ]
    
    return Dataset.from_dict({"prompt": prompts})

def convert_conversational_to_prompt(dataset):
    """Convert conversational dataset to prompt-only format."""
    def extract_prompt(example):
        messages = example["messages"]
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if user_messages:
            return {"prompt": user_messages[-1]["content"]}
        return {"prompt": ""}
    
    return dataset.map(extract_prompt, remove_columns=dataset.column_names)

def validate_dataset(dataset):
    """Validate dataset format and content."""
    print("Dataset validation:")
    print(f"  Columns: {dataset.column_names}")
    print(f"  Size: {len(dataset)} examples")
    
    if "prompt" not in dataset.column_names:
        raise ValueError("Dataset must contain 'prompt' column")
    
    empty_prompts = sum(1 for example in dataset if not example["prompt"].strip())
    if empty_prompts > 0:
        print(f"  Warning: {empty_prompts} empty prompts found")
    
    avg_length = sum(len(example["prompt"]) for example in dataset) / len(dataset)
    print(f"  Average prompt length: {avg_length:.1f} characters")
    
    print(f"  Sample prompt: {dataset[0]['prompt'][:100]}...")
    print()

def train_with_different_configs(dataset, model, tokenizer):
    """Train with different RLSC configurations."""
    configs = [
        {
            "name": "entropy_conservative",
            "confidence_method": "entropy",
            "confidence_threshold": 0.7,
            "beta": 0.05,
        },
        {
            "name": "max_prob_aggressive", 
            "confidence_method": "max_prob",
            "confidence_threshold": 0.3,
            "beta": 0.2,
        },
        {
            "name": "variance_balanced",
            "confidence_method": "variance",
            "confidence_threshold": 0.5,
            "beta": 0.1,
        },
    ]
    
    for config_dict in configs:
        print(f"Training with {config_dict['name']} configuration...")
        
        config = RLSCConfig(
            output_dir=f"./rlsc_custom_{config_dict['name']}",
            confidence_method=config_dict["confidence_method"],
            confidence_threshold=config_dict["confidence_threshold"],
            beta=config_dict["beta"],
            max_steps=10,
            per_device_train_batch_size=1,
            learning_rate=1e-6,
            logging_steps=5,
            eval_strategy="no",
        )
        
        trainer = RLSCTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        
        trainer.train()
        print(f"  {config_dict['name']} training completed\n")

def main():
    print("=== RLSC Custom Data Preparation Example ===\n")
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("1. Creating custom prompt dataset...")
    custom_dataset = create_custom_prompt_dataset()
    validate_dataset(custom_dataset)
    
    print("2. Converting conversational data...")
    try:
        conv_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:50]")
        prompt_dataset = convert_conversational_to_prompt(conv_dataset)
        validate_dataset(prompt_dataset)
    except Exception as e:
        print(f"  Could not load conversational dataset: {e}")
        print("  Using custom dataset instead\n")
        prompt_dataset = custom_dataset
    
    print("3. Training with different configurations...")
    train_with_different_configs(custom_dataset, model, tokenizer)
    
    print("=== Custom Data Example Complete ===")

if __name__ == "__main__":
    main()
