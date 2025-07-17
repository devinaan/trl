from trl import UFTTrainer, GRPOTrainer

def test_basic_functionality():
    print("=== Simple UFTTrainer Validation ===")
    
    print("✓ UFTTrainer imported successfully")
    print("✓ GRPOTrainer imported successfully")
    
    assert issubclass(UFTTrainer, GRPOTrainer)
    print("✓ UFTTrainer correctly inherits from GRPOTrainer")
    
    assert hasattr(UFTTrainer, '_generate_and_score_completions')
    assert hasattr(UFTTrainer, '_replace_completions_with_sft_answers')
    print("✓ UFTTrainer has expected methods")
    
    parent_method = GRPOTrainer._generate_and_score_completions
    child_method = UFTTrainer._generate_and_score_completions
    assert parent_method != child_method
    print("✓ UFTTrainer properly overrides _generate_and_score_completions")
    
    print("✓ All basic validation tests passed")
    print("✓ UFTTrainer implementation validation complete")

if __name__ == "__main__":
    test_basic_functionality()
