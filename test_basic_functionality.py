#!/usr/bin/env python3

print("Testing basic RLSC functionality...")

try:
    from trl import RLSCTrainer, RLSCConfig
    print("✓ Main imports work")
except ImportError as e:
    print(f"✗ Main imports failed: {e}")
    exit(1)

try:
    config = RLSCConfig()
    print(f"✓ RLSCConfig created successfully")
    print(f"  confidence_threshold: {config.confidence_threshold}")
    print(f"  confidence_method: {config.confidence_method}")
    print(f"  beta: {config.beta}")
except Exception as e:
    print(f"✗ RLSCConfig creation failed: {e}")
    exit(1)

try:
    from trl.scripts.rlsc import make_parser
    parser = make_parser()
    print("✓ CLI script imports and parser creation work")
except ImportError as e:
    print(f"✗ CLI script imports failed: {e}")
    exit(1)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_name = "hf-internal-testing/tiny-random-gpt2"
    
    print(f"Testing trainer instantiation with {model_name}...")
    
    trainer = RLSCTrainer(
        model=model_name,
        args=RLSCConfig(
            output_dir="./test_output",
            max_steps=1,
            per_device_train_batch_size=1,
            logging_steps=1,
        ),
        train_dataset=None,  # We'll skip actual training
    )
    print("✓ RLSCTrainer instantiation successful")
    
    dummy_logits = torch.randn(2, 10, 1000)  # batch_size=2, seq_len=10, vocab_size=1000
    confidence = trainer.calculate_confidence(dummy_logits)
    print(f"✓ Confidence calculation works, shape: {confidence.shape}")
    
except Exception as e:
    print(f"✗ Trainer instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("✓ All basic functionality tests passed!")
