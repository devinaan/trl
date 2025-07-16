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
    
    Default values are chosen to be conservative and follow established patterns in the TRL library:
    - Regularization weight (0.01) follows the pattern of auxiliary loss coefficients in other trainers
    - Token threshold (0.0) applies regularization to all tokens by default for maximum stability
    - SFT hints are enabled by default as they are the core feature of UFT
    
    Parameters:
        uft_regularization_weight (`float`, *optional*, defaults to `0.01`):
            Weight for the UFT regularization term that incorporates SFT guidance.
            Default value follows common practice for auxiliary loss terms in preference optimization.
        use_sft_hints (`bool`, *optional*, defaults to `True`):
            Whether to use SFT answer hints during dynamic prompt generation.
            Enabled by default as this is the core feature of UFT.
        sft_answer_column (`str`, *optional*, defaults to `"sft_answer"`):
            Column name containing SFT teacher signals to use as hints.
            Standard column name for SFT teacher signals in UFT datasets.
        token_threshold (`float`, *optional*, defaults to `0.0`):
            Threshold for token-based weighting in the regularization term.
            Default of 0.0 applies regularization to all tokens; increase to focus on high-weight tokens only.
    """
    
    uft_regularization_weight: float = field(
        default=0.01,
        metadata={"help": "Weight for the UFT regularization term that incorporates SFT guidance. Default follows common practice for auxiliary loss terms."}
    )
    use_sft_hints: bool = field(
        default=True,
        metadata={"help": "Whether to use SFT answer hints during dynamic prompt generation. Core UFT feature enabled by default."}
    )
    sft_answer_column: str = field(
        default="sft_answer",
        metadata={"help": "Column name containing SFT teacher signals to use as hints. Standard UFT dataset column name."}
    )
    token_threshold: float = field(
        default=0.0,
        metadata={"help": "Threshold for token-based weighting in the regularization term. 0.0 applies to all tokens, higher values focus on high-weight tokens."}
    )
