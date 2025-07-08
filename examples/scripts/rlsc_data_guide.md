# RLSC Data Preparation Guide

This guide explains how to prepare and use data with the RLSCTrainer (Reinforcement Learning via Self Confidence).

## Overview

RLSCTrainer uses the model's own confidence as a reward signal for fine-tuning. It expects **prompt-only datasets** where each example contains a prompt that the model should complete.

## Dataset Format Requirements

### Standard Format (Recommended)
```python
{
    "prompt": "Explain the concept of machine learning:"
}
```

### Conversational Format
```python
{
    "prompt": [
        {"role": "user", "content": "Explain the concept of machine learning:"}
    ]
}
```

## Data Preparation Examples

### 1. Using Existing TRL Datasets

```python
from datasets import load_dataset
from trl import RLSCTrainer, RLSCConfig

# Load prompt-only dataset
dataset = load_dataset("trl-lib/tldr", split="train")
print(dataset[0])  # Shows: {"prompt": "SUBREDDIT: r/..."}

trainer = RLSCTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    args=RLSCConfig(),
    train_dataset=dataset,
)
```

### 2. Creating Custom Datasets

```python
from datasets import Dataset

# Create from list of prompts
prompts = [
    "Write a story about:",
    "Explain how to:",
    "What are the benefits of:",
]

dataset = Dataset.from_dict({"prompt": prompts})
```

### 3. Converting Other Formats

```python
# Convert from conversational format
def extract_prompts(example):
    messages = example["messages"]
    user_msg = [m for m in messages if m["role"] == "user"][-1]
    return {"prompt": user_msg["content"]}

dataset = dataset.map(extract_prompts)
```

### 4. Converting from Text Files

```python
# Load from text file (one prompt per line)
with open("prompts.txt", "r") as f:
    prompts = [line.strip() for line in f if line.strip()]

dataset = Dataset.from_dict({"prompt": prompts})
```

### 5. Converting from JSON/CSV

```python
import pandas as pd

# From CSV
df = pd.read_csv("data.csv")
dataset = Dataset.from_pandas(df[["prompt"]])

# From JSON
import json
with open("data.json", "r") as f:
    data = json.load(f)
dataset = Dataset.from_dict({"prompt": [item["text"] for item in data]})
```

## Confidence Methods

RLSCTrainer supports three confidence calculation methods:

### 1. Entropy (Default)
- **How it works**: Lower entropy = higher confidence
- **Best for**: General use cases, balanced confidence estimation
- **Formula**: `confidence = exp(-entropy)`

```python
config = RLSCConfig(confidence_method="entropy")
```

### 2. Max Probability
- **How it works**: Uses maximum token probability as confidence
- **Best for**: Fast computation, simple confidence measure
- **Formula**: `confidence = max(softmax(logits))`

```python
config = RLSCConfig(confidence_method="max_prob")
```

### 3. Variance
- **How it works**: Lower variance = higher confidence
- **Best for**: Detecting uncertainty in probability distributions
- **Formula**: `confidence = exp(-variance)`

```python
config = RLSCConfig(confidence_method="variance")
```

## Configuration Parameters

```python
config = RLSCConfig(
    # Core RLSC parameters
    confidence_threshold=0.5,    # Confidence threshold for reward calculation
    confidence_method="entropy", # Method: "entropy", "max_prob", "variance"
    beta=0.1,                   # Temperature for confidence scaling
    
    # Length constraints
    max_prompt_length=512,      # Maximum prompt length in tokens
    max_completion_length=256,  # Maximum completion length in tokens
    
    # Training parameters
    learning_rate=1e-6,         # Lower learning rates often work better
    per_device_train_batch_size=2,
    max_steps=1000,
    logging_steps=10,
    
    # Output
    output_dir="./rlsc_output",
)
```

## Parameter Tuning Guidelines

### Confidence Threshold
- **Low (0.1-0.3)**: More aggressive training, higher rewards
- **Medium (0.4-0.6)**: Balanced approach (recommended starting point)
- **High (0.7-0.9)**: Conservative training, only high-confidence completions rewarded

### Beta (Scaling Factor)
- **Low (0.01-0.05)**: Gentle confidence-based adjustments
- **Medium (0.1-0.2)**: Standard scaling (recommended)
- **High (0.5+)**: Strong confidence influence (use carefully)

### Learning Rate
- Start with **1e-6** to **1e-5** (lower than standard fine-tuning)
- RLSC can be sensitive to learning rate due to confidence-based rewards

## Common Issues and Solutions

### Issue: "KeyError: 'prompt'"
**Solution**: Ensure your dataset has a "prompt" column.
```python
# Check dataset columns
print(dataset.column_names)

# Rename column if needed
dataset = dataset.rename_column("text", "prompt")
```

### Issue: Empty or very short prompts
**Solution**: Filter your dataset to remove empty prompts:
```python
dataset = dataset.filter(lambda x: len(x["prompt"].strip()) > 10)
```

### Issue: Memory issues with large datasets
**Solution**: Use streaming datasets or reduce batch size:
```python
dataset = load_dataset("your-dataset", streaming=True)
config = RLSCConfig(per_device_train_batch_size=1)
```

### Issue: Training loss becomes negative
**Solution**: This is expected! RLSC uses policy gradient loss which can be negative.

### Issue: Very high or low confidence values
**Solution**: Adjust confidence threshold and beta:
```python
# For high confidence values
config = RLSCConfig(confidence_threshold=0.8, beta=0.05)

# For low confidence values  
config = RLSCConfig(confidence_threshold=0.2, beta=0.2)
```

## Best Practices

1. **Start with small datasets** (100-1000 examples) to test your setup
2. **Use appropriate confidence thresholds** (0.3-0.7 typically work well)
3. **Monitor training logs** to ensure confidence calculations are reasonable
4. **Experiment with different confidence methods** for your specific use case
5. **Validate your data format** before training
6. **Use lower learning rates** than standard fine-tuning
7. **Check confidence distributions** during training

## Example Datasets

Compatible datasets from Hugging Face Hub:
- `trl-lib/tldr` - Summarization prompts
- `trl-lib/ultrafeedback_binarized` - Can be converted to prompt-only
- `HuggingFaceH4/ultrachat_200k` - Can be converted from conversational format

## Data Quality Tips

1. **Diverse prompts**: Include varied prompt types and lengths
2. **Clear instructions**: Prompts should be clear and well-formed
3. **Appropriate length**: Not too short (>10 chars) or too long (<1000 chars)
4. **Consistent format**: All prompts should follow similar structure
5. **Quality over quantity**: Better to have fewer high-quality prompts

## Monitoring Training

Watch these metrics during training:
- **Loss**: Can be negative (this is normal for RLSC)
- **Confidence values**: Should be reasonable (0.0-1.0 range)
- **Reward distribution**: Check if rewards are too extreme

```python
# Add logging to monitor confidence
import logging
logging.basicConfig(level=logging.INFO)
```

For more examples, see the [TRL documentation](https://huggingface.co/docs/trl/dataset_formats) and the example scripts in this directory.
