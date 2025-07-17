from trl import UFTTrainer, GRPOTrainer
from datasets import Dataset
import torch

def test_train_eval_mode_distinction():
    """Test that sft_answer replacement only occurs in training mode."""
    print("=== Testing Train/Eval Mode Distinction ===")
    
    import inspect
    source = inspect.getsource(UFTTrainer._generate_and_score_completions)
    
    assert "self.model.training" in source, "Training mode check should be present"
    print("✓ Training mode check found in _generate_and_score_completions")
    
    assert "if self.model.training:" in source, "sft_answer replacement should be conditional on training mode"
    print("✓ sft_answer replacement is conditional on training mode")
    
    lines = source.split('\n')
    training_check_found = False
    replacement_call_found = False
    
    for i, line in enumerate(lines):
        if "if self.model.training:" in line:
            training_check_found = True
            for j in range(i+1, min(i+5, len(lines))):
                if "_replace_completions_with_sft_answers" in lines[j]:
                    replacement_call_found = True
                    break
            break
    
    assert training_check_found and replacement_call_found, "sft_answer replacement should be inside training mode check"
    print("✓ sft_answer replacement is properly nested within training mode check")
    
    print("✓ All train/eval mode distinction tests passed")
    return True

if __name__ == "__main__":
    test_train_eval_mode_distinction()
    print("✓ Train/eval mode distinction validation complete")
