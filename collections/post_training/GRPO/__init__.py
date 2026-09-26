"""GRPO: group-relative policy optimization with verifiable rewards.

    rewards.py       grading rule -> scalar reward (self-contained, runnable)
    rollout.py       batched sampling that returns token ids + masks
    grpo_advantages.py  group-relative advantages from rewards
    grpo_logprobs.py    per-token log-probs of a sequence under the model
    grpo_loss.py     clipped objective (pure math)
    grpo_data.py     prompt-only dataset (train rollouts and the test-set eval)

The training pipeline that wires these together lives in
collections/qwen3/4_Qwen3_Gsm8k_GRPO.py.
"""

from .grpo_data import PromptDataset, prompt_collate_fn
from .grpo_advantages import group_advantages
from .grpo_logprobs import token_logprobs
from .grpo_loss import grpo_loss
from .rewards import (
    DEFAULT_WEIGHTS, gsm8k_reward, parse_completion,
    reward_groups,
)
from .rollout import RolloutBatch, sample_rollouts

__all__ = [
    "PromptDataset", "prompt_collate_fn",
    "group_advantages", "grpo_loss",
    "token_logprobs",
    "DEFAULT_WEIGHTS", "gsm8k_reward", "parse_completion",
    "reward_groups",
    "RolloutBatch", "sample_rollouts",
]
