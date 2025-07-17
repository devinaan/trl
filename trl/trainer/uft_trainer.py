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

import warnings
from typing import Any, Union

import torch
from accelerate.utils import broadcast_object_list, gather, gather_object

from ..data_utils import is_conversational, maybe_apply_chat_template
from ..extras.profiling import profiling_context
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
        Generate completions and replace with sft_answer before reward calculation.
        
        This method completely overrides the parent implementation to ensure that
        sft_answer replacement occurs before reward and advantage calculation.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )

        if self.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    from vllm import GuidedDecodingParams
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                
                from vllm import SamplingParams
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    orig_size = len(prompts_text)
                    all_prompts_text = gather_object(prompts_text)
                    if self.accelerator.is_main_process:
                        all_prompts_text = all_prompts_text[: orig_size * self.accelerator.num_processes]
                    else:
                        all_prompts_text = [None] * (orig_size * self.accelerator.num_processes)
                    all_prompts_text = broadcast_object_list(all_prompts_text, from_process=0)
                    
                    with profiling_context(self, "vLLM.generate"):
                        all_outputs = self.vllm_engine.generate(all_prompts_text, sampling_params)
                    
                    all_completion_ids = []
                    for output in all_outputs:
                        completion_ids_single = output.outputs[0].token_ids
                        all_completion_ids.append(completion_ids_single)
                    
                    process_slice = slice(
                        self.accelerator.process_index * orig_size,
                        (self.accelerator.process_index + 1) * orig_size,
                    )
                    completion_ids = all_completion_ids[process_slice]
                else:
                    with profiling_context(self, "vLLM.generate"):
                        outputs = self.vllm_engine.generate(prompts_text, sampling_params)
                    
                    completion_ids = []
                    for output in outputs:
                        completion_ids_single = output.outputs[0].token_ids
                        completion_ids.append(completion_ids_single)

        elif hasattr(self, 'use_transformers_paged') and self.use_transformers_paged:
            from transformers import GenerationConfig
            generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                eos_token_id=self.processing_class.eos_token_id,
                pad_token_id=self.processing_class.pad_token_id,
                **(self.args.generation_kwargs or {}),
            )

            with profiling_context(self, "transformers_paged.generate"):
                prompt_completion_ids = self.model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=False,
                )

            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            completion_ids = [completion_ids[i].tolist() for i in range(completion_ids.size(0))]

        else:
            from transformers import GenerationConfig
            generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                eos_token_id=self.processing_class.eos_token_id,
                pad_token_id=self.processing_class.pad_token_id,
                **(self.args.generation_kwargs or {}),
            )

            with profiling_context(self, "transformers.generate"):
                prompt_completion_ids = self.model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=False,
                )

            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            completion_ids = [completion_ids[i].tolist() for i in range(completion_ids.size(0))]

        if not self.use_vllm or (hasattr(self, 'vllm_mode') and self.vllm_mode == "colocate"):
            from ..trainer.utils import pad
            completion_ids = pad(completion_ids, self.processing_class.pad_token_id, device)

        if hasattr(self, 'use_transformers_paged') and self.use_transformers_paged:
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            if not self.use_vllm:
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                prompt_length = prompt_ids.size(1)
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
            else:
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        completion_lengths = completion_mask.sum(1)

        if hasattr(self, 'mask_truncated_completions') and self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps_and_entropies(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )["logps"]
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps_and_entropies(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )["logps"]
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps_and_entropies(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep
                        )["logps"]
            else:
                ref_per_token_logps = None

        if self.model.training:
            completion_ids, completion_mask, completion_ids_list, is_eos, eos_idx, completion_lengths, attention_mask, prompt_completion_ids = self._replace_completions_with_sft_answers(
                inputs, completion_ids, completion_mask, completion_ids_list, is_eos, eos_idx, completion_lengths, attention_mask, prompt_mask, prompt_completion_ids
            )

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            from .grpo_trainer import nanstd
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

    def _replace_completions_with_sft_answers(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        completion_ids_list: list[list[int]],
        is_eos: torch.Tensor,
        eos_idx: torch.Tensor,
        completion_lengths: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_mask: torch.Tensor,
        prompt_completion_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Replace completions with sft_answer when available.
        
        This method modifies completion_ids, completion_mask, completion_ids_list,
        and all dependent variables (is_eos, eos_idx, completion_lengths, attention_mask, prompt_completion_ids)
        to replace the first completion for each prompt that has sft_answer.
        """
        device = self.accelerator.device
        
        has_sft_answer = [
            'sft_answer' in inp and inp['sft_answer'] is not None 
            for inp in inputs
        ]
        
        if not any(has_sft_answer):
            return completion_ids, completion_mask, completion_ids_list, is_eos, eos_idx, completion_lengths, attention_mask, prompt_completion_ids
        
        modified_completion_ids = completion_ids.clone()
        modified_completion_mask = completion_mask.clone()
        modified_completion_ids_list = completion_ids_list.copy()
        modified_is_eos = is_eos.clone()
        modified_eos_idx = eos_idx.clone()
        modified_completion_lengths = completion_lengths.clone()
        modified_attention_mask = attention_mask.clone()
        modified_prompt_completion_ids = prompt_completion_ids.clone()
        
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
                sft_tokens = torch.tensor(sft_answer, dtype=torch.long, device=device)
            
            max_len = completion_ids.size(1)
            if len(sft_tokens) > max_len:
                sft_tokens = sft_tokens[:max_len]
                actual_length = max_len
            else:
                actual_length = len(sft_tokens)
                pad_length = max_len - len(sft_tokens)
                if pad_length > 0:
                    sft_tokens = torch.cat([
                        sft_tokens, 
                        torch.full((pad_length,), self.processing_class.pad_token_id, dtype=torch.long, device=device)
                    ])
            
            modified_completion_ids[i] = sft_tokens.to(device)
            
            modified_is_eos[i] = modified_completion_ids[i] == self.processing_class.eos_token_id
            
            if modified_is_eos[i].any():
                modified_eos_idx[i] = modified_is_eos[i].int().argmax()
            else:
                modified_eos_idx[i] = max_len  # No EOS found, use max length
            
            sequence_indices = torch.arange(max_len, device=device)
            modified_completion_mask[i] = (sequence_indices <= modified_eos_idx[i]).int()
            
            # Update completion_ids_list based on new completion_mask
            modified_completion_ids_list[i] = [
                id.item() for id, m in zip(modified_completion_ids[i], modified_completion_mask[i]) if m
            ]
            
            # Update completion_lengths
            modified_completion_lengths[i] = modified_completion_mask[i].sum()
            
            # Update attention_mask (concatenate prompt_mask with new completion_mask)
            modified_attention_mask[i] = torch.cat([prompt_mask[i], modified_completion_mask[i]])
            
            prompt_length = modified_prompt_completion_ids.size(1) - modified_completion_ids.size(1)
            modified_prompt_completion_ids[i] = torch.cat([modified_prompt_completion_ids[i][:prompt_length], modified_completion_ids[i]])
        
        if hasattr(self, 'mask_truncated_completions') and self.mask_truncated_completions:
            for i in range(modified_completion_ids.size(0)):
                if not modified_is_eos[i].any():  # No EOS found = truncated
                    modified_completion_mask[i] = torch.zeros_like(modified_completion_mask[i])
                    modified_attention_mask[i] = torch.cat([prompt_mask[i], modified_completion_mask[i]])
                    modified_completion_lengths[i] = 0
        
        return modified_completion_ids, modified_completion_mask, modified_completion_ids_list, modified_is_eos, modified_eos_idx, modified_completion_lengths, modified_attention_mask, modified_prompt_completion_ids
