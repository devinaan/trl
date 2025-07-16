"""
Example script for training with UFTTrainer (Uncertainty-guided Fine-Tuning).

This script demonstrates how to use UFTTrainer with SFT hints for uncertainty-guided training.
"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import UFTTrainer, UFTConfig


def create_sample_dataset():
    """Create a sample dataset with sft_answer column for UFT training."""
    data = [
        {
            "prompt": "What is the capital of France?",
            "sft_answer": "The capital of France is Paris, which is located in the north-central part of the country."
        },
        {
            "prompt": "Explain photosynthesis.",
            "sft_answer": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen."
        },
        {
            "prompt": "What is 2 + 2?",
            "sft_answer": "2 + 2 equals 4. This is a basic arithmetic operation."
        },
        {
            "prompt": "How do you make coffee?",
            "sft_answer": "To make coffee, grind coffee beans, add hot water, and let it brew for several minutes."
        },
    ]
    return Dataset.from_list(data)


def reward_function(completions, **kwargs):
    """Simple reward function that rewards longer, more detailed completions."""
    return [len(completion.split()) / 10.0 for completion in completions]


def main():
    model_name = "microsoft/DialoGPT-small"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = create_sample_dataset()
    
    config = UFTConfig(
        output_dir="./uft_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        logging_steps=1,
        uft_regularization_weight=0.1,
        use_sft_hints=True,
        sft_answer_column="sft_answer",
        uncertainty_threshold=0.5,
        max_prompt_length=256,
        max_completion_length=128,
        num_generations=4,
    )
    
    trainer = UFTTrainer(
        model=model,
        reward_funcs=reward_function,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
