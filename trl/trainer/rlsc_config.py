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

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class RLSCConfig(TrainingArguments):
    r"""
    Configuration class for the [`RLSCTrainer`].

    This class includes only the parameters that are specific to RLSC training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Parameters:
        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a
            string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model for consistent confidence calculation.
        confidence_threshold (`float`, *optional*, defaults to `0.5`):
            Threshold for confidence-based reward calculation.
        confidence_method (`str`, *optional*, defaults to `"entropy"`):
            Method to calculate confidence. Options: "entropy", "max_prob", "variance".
        beta (`float`, *optional*, defaults to `0.1`):
            Temperature parameter for confidence scaling.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the completion.
    """

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    logging_steps: float = field(
        default=10,
        metadata={"help": "Log every X updates steps."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."},
    )

    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a string."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model for consistent confidence calculation."},
    )

    confidence_threshold: float = field(
        default=0.5,
        metadata={"help": "Threshold for confidence-based reward calculation."},
    )
    confidence_method: str = field(
        default="entropy",
        metadata={
            "help": "Method to calculate confidence. Options: 'entropy', 'max_prob', 'variance'.",
        },
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "Temperature parameter for confidence scaling."},
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the completion."},
    )
