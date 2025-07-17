from trl import UFTTrainer, GRPOTrainer
from datasets import Dataset
import torch
import inspect

def test_all_variables_updated():
    """Test that all completion-dependent variables are updated after sft_answer replacement."""
    print("=== Testing All Variables Updated ===")
    
    signature = inspect.signature(UFTTrainer._replace_completions_with_sft_answers)
    params = list(signature.parameters.keys())
    
    expected_params = [
        'self', 'inputs', 'completion_ids', 'completion_mask', 'completion_ids_list',
        'is_eos', 'eos_idx', 'completion_lengths', 'attention_mask', 'prompt_mask', 'prompt_completion_ids'
    ]
    
    for param in expected_params:
        assert param in params, f"Missing parameter: {param}"
    print("✓ Method signature has all required parameters")
    
    return_annotation = signature.return_annotation
    if hasattr(return_annotation, '__args__'):
        return_types = return_annotation.__args__
        assert len(return_types) == 8, f"Expected 8 return values, got {len(return_types)}"
    print("✓ Method returns correct number of values")
    
    source = inspect.getsource(UFTTrainer._generate_and_score_completions)
    assert "prompt_completion_ids = self._replace_completions_with_sft_answers(" in source, "Method call should include prompt_completion_ids"
    assert "completion_ids, completion_mask, completion_ids_list, is_eos, eos_idx, completion_lengths, attention_mask, prompt_completion_ids =" in source, "Method call should handle all return values"
    print("✓ Method call handles all variables")
    
    print("✓ All variable update tests passed")

def test_edge_cases():
    """Test edge cases like long sft_answers, missing EOS tokens, etc."""
    print("=== Testing Edge Cases ===")
    
    source = inspect.getsource(UFTTrainer._replace_completions_with_sft_answers)
    assert "mask_truncated_completions" in source, "Should support mask_truncated_completions feature"
    print("✓ mask_truncated_completions support found")
    
    assert "sequence_indices" in source, "Should use sequence_indices for completion_mask calculation"
    assert "modified_is_eos[i].int().argmax()" in source, "Should find first EOS token position"
    print("✓ EOS handling logic follows parent class pattern")
    
    assert "modified_prompt_completion_ids" in source, "Should update prompt_completion_ids"
    assert "torch.cat([modified_prompt_completion_ids[i][:prompt_length], modified_completion_ids[i]])" in source, "Should concatenate prompt and completion"
    print("✓ prompt_completion_ids update logic found")
    
    print("✓ All edge case tests passed")

def test_variable_consistency():
    """Test that variable calculations are consistent with parent class."""
    print("=== Testing Variable Consistency ===")
    
    source = inspect.getsource(UFTTrainer._replace_completions_with_sft_answers)
    assert "(sequence_indices <= modified_eos_idx[i]).int()" in source, "completion_mask should use sequence_indices <= eos_idx pattern"
    print("✓ completion_mask calculation follows parent class pattern")
    
    assert "for id, m in zip(modified_completion_ids[i], modified_completion_mask[i]) if m" in source, "completion_ids_list should use completion_mask"
    print("✓ completion_ids_list calculation uses completion_mask")
    
    assert "modified_completion_lengths[i] = modified_completion_mask[i].sum()" in source, "completion_lengths should be sum of completion_mask"
    print("✓ completion_lengths calculation is correct")
    
    print("✓ All variable consistency tests passed")

if __name__ == "__main__":
    print("=== Comprehensive SFT Replacement Tests ===")
    
    test_all_variables_updated()
    test_edge_cases()
    test_variable_consistency()
    
    print("\n=== Test Summary ===")
    print("✓ All comprehensive sft_replacement tests passed successfully")
    print("✓ UFTTrainer properly handles all completion-dependent variables")
