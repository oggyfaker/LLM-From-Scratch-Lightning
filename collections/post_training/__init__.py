"""Post-training (RLHF-style) methods that sit on top of an SFT checkpoint.

    GRPO/   group-relative policy optimization with verifiable rewards

Each method is a self-contained package: its own loss math, rollout sampling,
reward rule and prompt dataset, so a training script imports one namespace and
nothing in collections/qwen3/ has to change.
"""
