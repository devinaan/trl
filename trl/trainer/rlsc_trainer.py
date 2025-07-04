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
import torch.nn.functional as F
from typing import Optional, Union, Any, Dict, List
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)

from ..data_utils import maybe_apply_chat_template
from .rlsc_config import RLSCConfig
from .utils import (
    disable_dropout_in_model,
    generate_model_card,
)


class RLSCTrainer(Trainer):
    """
    Trainer for Reinforcement Learning via Self Confidence (RLSC) method.
    
    This method uses the model's own confidence as a reward signal, eliminating the need
    for external reward models or preference data.

    Example:

    ```python
    from datasets import load_dataset
    from trl import RLSCTrainer, RLSCConfig

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = RLSCTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        args=RLSCConfig(),
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        args ([`RLSCConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoTokenizer.from_pretrained`].
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use.
    """

    _tag_names = ["trl", "rlsc"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: Optional[RLSCConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        **kwargs,
    ):
        if args is None:
            args = RLSCConfig()

        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path)
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        if args.disable_dropout:
            disable_dropout_in_model(model)

        self.confidence_threshold = args.confidence_threshold
        self.confidence_method = args.confidence_method
        self.beta = args.beta
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            **kwargs,
        )

    def calculate_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculate confidence based on the specified method."""
        if self.confidence_method == "entropy":
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            confidence = torch.exp(-entropy)
        elif self.confidence_method == "max_prob":
            confidence = F.softmax(logits, dim=-1).max(dim=-1)[0]
        elif self.confidence_method == "variance":
            probs = F.softmax(logits, dim=-1)
            variance = torch.var(probs, dim=-1)
            confidence = torch.exp(-variance)
        else:
            raise ValueError(f"Unknown confidence method: {self.confidence_method}")
        
        return confidence

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute RLSC loss using confidence as reward signal."""
        outputs = model(**inputs)
        logits = outputs.logits
        
        confidence = self.calculate_confidence(logits)
        
        rewards = (confidence - self.confidence_threshold) * self.beta
        
        labels = inputs.get("labels", inputs["input_ids"])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_rewards = rewards[..., :-1].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        selected_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        loss = -(selected_log_probs * shift_rewards).mean()
        
        return (loss, outputs) if return_outputs else loss

    def create_model_card(self, model_name: Optional[str] = None, **kwargs):
        """Create model card for RLSC trainer."""
        model_card = generate_model_card(
            trainer_name="RLSC",
            trainer_citation="@article{rlsc2024, title={Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models}}",
            paper_title="Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models",
            paper_id="2506.06395",
            **kwargs
        )
        return model_card
