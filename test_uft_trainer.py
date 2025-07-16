from trl import UFTTrainer
from datasets import Dataset


def test_uft_trainer_import():
    print("✓ UFTTrainer imported successfully")


def test_uft_trainer_basic():
    data = {
        "prompt": ["What is 2+2?", "What is the capital of France?"],
        "sft_answer": ["4", "Paris"]
    }
    dataset = Dataset.from_dict(data)
    
    def dummy_reward_func(completions, **kwargs):
        return [1.0] * len(completions)
    
    try:
        trainer = UFTTrainer(
            model="gpt2",
            reward_funcs=dummy_reward_func,
            train_dataset=dataset,
        )
        print("✓ UFTTrainer instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Error instantiating UFTTrainer: {e}")
        return False


if __name__ == "__main__":
    test_uft_trainer_import()
    success = test_uft_trainer_basic()
    if success:
        print("✓ All tests passed")
    else:
        print("✗ Some tests failed")
