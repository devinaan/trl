#!/usr/bin/env python3
"""
Basic example of using RLSCTrainer for confidence-based fine-tuning.

This example demonstrates:
1. Loading a prompt-only dataset
2. Setting up RLSCTrainer with different confidence methods
3. Training with confidence as reward signal

Usage:
python examples/scripts/rlsc_basic.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/tldr \
    --confidence_method entropy \
    --confidence_threshold 0.5 \
    --beta 0.1 \
    --output_dir ./rlsc_basic_output
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import RLSCTrainer, RLSCConfig

def main():
    print("=== RLSC Basic Example ===\n")
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    dataset = load_dataset("trl-lib/tldr", split="train")
    print(f"Dataset format: {dataset.column_names}")
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Example prompt: {dataset[0]['prompt'][:100]}...")
    print()
    
    config = RLSCConfig(
        output_dir="./rlsc_basic_output",
        confidence_method="entropy",
        confidence_threshold=0.5,
        beta=0.1,
        max_steps=100,
        per_device_train_batch_size=2,
        learning_rate=1e-6,
        logging_steps=10,
        save_steps=50,
        eval_strategy="no",
    )
    
    print(f"Training configuration:")
    print(f"  Confidence method: {config.confidence_method}")
    print(f"  Confidence threshold: {config.confidence_threshold}")
    print(f"  Beta (scaling factor): {config.beta}")
    print(f"  Max steps: {config.max_steps}")
    print()
    
    trainer = RLSCTrainer(
        model=model,
        args=config,
        train_dataset=dataset.select(range(100)),
        processing_class=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    trainer.save_model()
    print(f"Model saved to {config.output_dir}")

def demo_confidence_methods():
    """Demonstrate different confidence calculation methods."""
    print("=== Confidence Methods Demo ===\n")
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    dataset = load_dataset("trl-lib/tldr", split="train").select(range(10))
    
    confidence_methods = ["entropy", "max_prob", "variance"]
    
    for method in confidence_methods:
        print(f"Testing {method} confidence method...")
        
        config = RLSCConfig(
            output_dir=f"./rlsc_demo_{method}",
            confidence_method=method,
            confidence_threshold=0.5,
            beta=0.1,
            max_steps=5,
            per_device_train_batch_size=1,
            learning_rate=1e-6,
            logging_steps=1,
            eval_strategy="no",
        )
        
        trainer = RLSCTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        
        trainer.train()
        print(f"  {method} training completed\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_confidence_methods()
    else:
        main()
