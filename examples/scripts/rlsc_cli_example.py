#!/usr/bin/env python3
"""
CLI-compatible RLSC example script.

This script can be run directly or through the TRL CLI:

Direct usage:
python examples/scripts/rlsc_cli_example.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/tldr \
    --confidence_method entropy \
    --output_dir ./rlsc_cli_output

TRL CLI usage:
trl rlsc \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/tldr \
    --confidence_method entropy \
    --output_dir ./rlsc_cli_output
"""

import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import RLSCConfig, RLSCTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

def main(script_args, training_args, model_args):
    """Main training function following TRL patterns."""
    
    print("=== RLSC CLI Example ===")
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Dataset: {script_args.dataset_name}")
    print(f"Confidence method: {training_args.confidence_method}")
    print(f"Confidence threshold: {training_args.confidence_threshold}")
    print(f"Beta: {training_args.beta}")
    print(f"Output directory: {training_args.output_dir}")
    print()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code
    )
    
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    print(f"Loaded dataset: {script_args.dataset_name}")
    print(f"Training examples: {len(dataset[script_args.dataset_train_split])}")
    
    if training_args.eval_strategy != "no":
        print(f"Evaluation examples: {len(dataset[script_args.dataset_test_split])}")
    print()
    
    trainer = RLSCTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    
    print("Starting RLSC training...")
    trainer.train()
    
    trainer.save_model(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")
    
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        print("Model pushed to Hugging Face Hub")

def make_parser(subparsers: argparse._SubParsersAction = None):
    """Create argument parser following TRL patterns."""
    dataclass_types = (ScriptArguments, RLSCConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("rlsc", help="Run the RLSC training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser

if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
