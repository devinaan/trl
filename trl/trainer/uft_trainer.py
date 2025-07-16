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

import torch
import warnings
from typing import Any, Union, Optional, List, Dict

from transformers import PreTrainedTokenizerBase

from .grpo_trainer import GRPOTrainer
from .uft_config import UFTConfig
from ..data_utils import maybe_apply_chat_template


class UFTTrainer(GRPOTrainer):
    """
    Trainer for Unified Fine-Tuning (UFT) method.
    
    This trainer extends GRPOTrainer to incorporate SFT guidance through dynamic prompt generation
    and uncertainty-based regularization.
    
    Example:
    
    ```python
    from datasets import Dataset
    from trl import UFTTrainer, UFTConfig
    
    dataset = Dataset.from_list([
        {
            "prompt": "What is the capital of France?",
            "sft_answer": "The capital of France is Paris."
        }
    ])
    
    def reward_func(completions, **kwargs):
        return [float(len(completion)) for completion in completions]
    
    trainer = UFTTrainer(
        model="microsoft/DialoGPT-small",
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=UFTConfig(
            uft_regularization_weight=0.1,
            use_sft_hints=True,
            sft_answer_column="sft_answer"
        )
    )
    
    trainer.train()
    ```
    """
    
    _tag_names = ["trl", "uft"]
    
    def __init__(
        self,
        model=None,
        reward_funcs=None,
        args: UFTConfig = None,
        train_dataset=None,
        eval_dataset=None,
        processing_class: PreTrainedTokenizerBase = None,
        reward_processing_classes=None,
        **kwargs,
    ):
        if args is None:
            args = UFTConfig()
        elif not isinstance(args, UFTConfig):
            raise ValueError("UFTTrainer requires UFTConfig")
            
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            **kwargs,
        )
        
        self.uft_regularization_weight = args.uft_regularization_weight
        self.use_sft_hints = args.use_sft_hints
        self.sft_answer_column = args.sft_answer_column
        self.uncertainty_threshold = args.uncertainty_threshold
        
    def _generate_dynamic_prompt_with_hints(self, example: Dict, sft_hint: str = None) -> Dict:
        """
        Generate dynamic prompts that incorporate SFT hints for unified fine-tuning.
        
        Args:
            example: Original example from dataset
            sft_hint: SFT answer to use as hint
            
        Returns:
            Modified example with hint-augmented prompt
        """
        if not self.use_sft_hints or not sft_hint:
            return example
            
        original_prompt = example.get("prompt", "")
        
        if isinstance(original_prompt, str):
            hint_text = f"\n\nHint: Consider this approach: {sft_hint}"
            example["prompt"] = original_prompt + hint_text
        elif isinstance(original_prompt, list):
            hint_message = {
                "role": "system", 
                "content": f"Hint: Consider this approach: {sft_hint}"
            }
            example["prompt"] = [hint_message] + original_prompt
            
        return example
        
    def _generate_and_score_completions(
        self, inputs: List[Dict[str, Union[torch.Tensor, Any]]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Override to implement dynamic prompt generation with SFT hints.
        """
        sft_hints = []
        for example in inputs:
            sft_hint = example.get(self.sft_answer_column, None)
            sft_hints.append(sft_hint)
            
        if self.use_sft_hints:
            modified_inputs = []
            for example, sft_hint in zip(inputs, sft_hints):
                modified_example = self._generate_dynamic_prompt_with_hints(example, sft_hint)
                modified_inputs.append(modified_example)
            inputs = modified_inputs
            
        result = super()._generate_and_score_completions(inputs)
        
        result["sft_hints"] = sft_hints
        
        return result
        
    def _compute_uncertainty_weights(self, per_token_logps: torch.Tensor, entropies: torch.Tensor = None) -> torch.Tensor:
        """
        Compute uncertainty-based weights for regularization.
        
        Args:
            per_token_logps: Per-token log probabilities
            entropies: Per-token entropies (optional)
            
        Returns:
            Uncertainty weights for each token
        """
        if entropies is not None:
            uncertainty = entropies
        else:
            uncertainty = -per_token_logps
            
        uncertainty_normalized = torch.sigmoid(uncertainty)
        
        weights = torch.where(
            uncertainty_normalized > self.uncertainty_threshold,
            uncertainty_normalized,
            torch.zeros_like(uncertainty_normalized)
        )
        
        return weights
        
    def _compute_uft_regularization(
        self, 
        per_token_logps: torch.Tensor, 
        completion_mask: torch.Tensor,
        sft_hints: List = None,
        entropies: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute UFT regularization term based on uncertainty and SFT guidance for unified fine-tuning.
        
        Args:
            per_token_logps: Per-token log probabilities
            completion_mask: Mask for completion tokens
            sft_hints: List of SFT hints for each example
            entropies: Per-token entropies (optional)
            
        Returns:
            UFT regularization loss
        """
        if not self.use_sft_hints or self.uft_regularization_weight == 0.0:
            return torch.tensor(0.0, device=per_token_logps.device)
            
        uncertainty_weights = self._compute_uncertainty_weights(per_token_logps, entropies)
        
        regularization_loss = -per_token_logps * uncertainty_weights * completion_mask
        
        return regularization_loss.sum() / completion_mask.sum().clamp(min=1.0)
        
    def _compute_loss(self, model, inputs):
        """
        Override to add UFT regularization term to the GRPO loss.
        """
        base_loss = super()._compute_loss(model, inputs)
        
        completion_mask = inputs["completion_mask"]
        
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        
        if self.token_entropy_percentile_threshold > 0.0:
            logps_and_entropies = self._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, compute_entropy=True
            )
            per_token_logps = logps_and_entropies["logps"]
            entropies = logps_and_entropies.get("entropies", None)
        else:
            per_token_logps = self._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep
            )["logps"]
            entropies = None
        
        sft_hints = inputs.get("sft_hints", [])
        
        uft_regularization = self._compute_uft_regularization(
            per_token_logps, completion_mask, sft_hints, entropies
        )
        
        total_loss = base_loss + self.uft_regularization_weight * uft_regularization
        
        mode = "train" if self.model.training else "eval"
        if hasattr(self, '_metrics'):
            if f"uft_regularization" not in self._metrics[mode]:
                self._metrics[mode]["uft_regularization"] = []
            self._metrics[mode]["uft_regularization"].append(
                self.accelerator.gather(uft_regularization).mean().item()
            )
        
        return total_loss
