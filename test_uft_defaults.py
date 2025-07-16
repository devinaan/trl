#!/usr/bin/env python3
"""
Test script to verify UFTConfig works with new default values.
"""

def test_uft_config_defaults():
    """Test UFTConfig instantiation with new defaults."""
    from trl import UFTConfig, UFTTrainer
    
    print("Testing UFTConfig with new defaults...")
    config = UFTConfig(
        output_dir='./test_output',
        bf16=False,
        fp16=False,
        no_cuda=True
    )
    
    print(f"uft_regularization_weight: {config.uft_regularization_weight}")
    print(f"uncertainty_threshold: {config.uncertainty_threshold}")
    print(f"use_sft_hints: {config.use_sft_hints}")
    print(f"sft_answer_column: {config.sft_answer_column}")
    
    assert config.uft_regularization_weight == 0.01, f"Expected 0.01, got {config.uft_regularization_weight}"
    assert config.uncertainty_threshold == 0.0, f"Expected 0.0, got {config.uncertainty_threshold}"
    assert config.use_sft_hints == True, f"Expected True, got {config.use_sft_hints}"
    assert config.sft_answer_column == "sft_answer", f"Expected 'sft_answer', got {config.sft_answer_column}"
    
    print("✓ UFTConfig instantiation successful with correct defaults!")
    print("✓ UFTTrainer class available:", UFTTrainer is not None)
    
    return True

if __name__ == "__main__":
    test_uft_config_defaults()
    print("All tests passed!")
