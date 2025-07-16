# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Union

import torch

from .grpo_trainer import GRPOTrainer


class UFTTrainer(GRPOTrainer):
    """
    Trainer that extends GRPOTrainer to support SFT answer replacement.
    
    When training data contains 'sft_answer' field, one of the num_generations
    completions will be replaced with the sft_answer before reward calculation.
    
    This allows for incorporating supervised fine-tuning answers directly into
    the GRPO training process, treating the sft_answer as if it was one of the
    generated completions for reward calculation and advantage computation.
    """

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Generate completions and optionally replace one with sft_answer when available.
        
        This method first calls the parent's generation method to produce num_generations
        completions per prompt, then replaces the first completion with sft_answer
        when the input data contains this field.
        
        Args:
            inputs: List of input dictionaries that may contain 'sft_answer' field
            
        Returns:
            Dictionary containing generated completions with sft_answer replacements
        """
        results = super()._generate_and_score_completions(inputs)
        
        self._replace_with_sft_answers(inputs, results)
        
        return results

    def _replace_with_sft_answers(
        self, 
        inputs: list[dict[str, Union[torch.Tensor, Any]]], 
        results: dict[str, Union[torch.Tensor, Any]]
    ) -> None:
        """
        Replace one completion with sft_answer when available.
        
        For each input that contains an 'sft_answer' field, this method replaces
        the first generated completion with the tokenized sft_answer. The replacement
        maintains the same tensor structure and masking as the original completions.
        
        Args:
            inputs: Original input data that may contain sft_answer
            results: Results from _generate_and_score_completions to modify in-place
        """
        device = self.accelerator.device
        
        has_sft_answer = [
            'sft_answer' in inp and inp['sft_answer'] is not None 
            for inp in inputs
        ]
        
        if not any(has_sft_answer):
            return
        
        completion_ids = results["completion_ids"]
        completion_mask = results["completion_mask"]
        
        for i, (inp, has_sft) in enumerate(zip(inputs, has_sft_answer)):
            if not has_sft:
                continue
                
            sft_answer = inp['sft_answer']
            
            if isinstance(sft_answer, str):
                sft_tokens = self.processing_class(
                    text=sft_answer, 
                    return_tensors="pt", 
                    padding=False, 
                    add_special_tokens=False
                )["input_ids"].squeeze(0)
            else:
                sft_tokens = torch.tensor(sft_answer, device=device)
            
            max_len = completion_ids.size(1)
            if len(sft_tokens) > max_len:
                sft_tokens = sft_tokens[:max_len]
            else:
                pad_length = max_len - len(sft_tokens)
                if pad_length > 0:
                    sft_tokens = torch.cat([
                        sft_tokens, 
                        torch.full((pad_length,), self.processing_class.pad_token_id, device=device)
                    ])
            
            completion_ids[i] = sft_tokens.to(device)
            
            actual_length = len(sft_tokens) if len(sft_tokens) <= max_len else max_len
            completion_mask[i] = torch.zeros(max_len, device=device)
            completion_mask[i][:actual_length] = 1
