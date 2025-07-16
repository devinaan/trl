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

from .grpo_config import GRPOConfig


@dataclass
class UFTConfig(GRPOConfig):
    r"""
    Configuration class for the [`UFTTrainer`].
    
    This class extends [`GRPOConfig`] with UFT-specific parameters for unified fine-tuning.
    
    Parameters:
        uft_regularization_weight (`float`, *optional*, defaults to `0.1`):
            Weight for the UFT regularization term that incorporates SFT guidance.
        use_sft_hints (`bool`, *optional*, defaults to `True`):
            Whether to use SFT answer hints during dynamic prompt generation.
        sft_answer_column (`str`, *optional*, defaults to `"sft_answer"`):
            Column name containing SFT teacher signals to use as hints.
        uncertainty_threshold (`float`, *optional*, defaults to `0.5`):
            Threshold for uncertainty-based weighting in the regularization term.
    """
    
    uft_regularization_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for the UFT regularization term that incorporates SFT guidance."}
    )
    use_sft_hints: bool = field(
        default=True,
        metadata={"help": "Whether to use SFT answer hints during dynamic prompt generation."}
    )
    sft_answer_column: str = field(
        default="sft_answer",
        metadata={"help": "Column name containing SFT teacher signals to use as hints."}
    )
    uncertainty_threshold: float = field(
        default=0.5,
        metadata={"help": "Threshold for uncertainty-based weighting in the regularization term."}
    )
