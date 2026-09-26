# -----------
# ADVANTAGES
# -----------
def group_advantages(rewards, group_size, scale_by_std=False, eps=1e-4):
    """(N,) rewards -> (N,) advantages, normalized WITHIN each group of G.

    Rollouts must be laid out contiguously per prompt: [p0 x G, p1 x G, ...].

    scale_by_std=False is Dr. GRPO. Binary-ish rewards give a tiny group std, and
    dividing by it inflates the advantage of near-unanimous groups - the very
    groups that carry the least information.

    Returns (advantages, degenerate) where degenerate is (N/G,) and True for a
    group whose rollouts all scored the same: its advantages are exactly 0, so
    it contributes no gradient no matter how much compute the rollouts cost.
    """
    grouped = rewards.view(-1, group_size).float()
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    # Bessel-corrected, as TRL / Unsloth scale it: (r - mean) / (std + 1e-4)
    std = grouped.std(dim=1, unbiased=True, keepdim=True)
    if scale_by_std:
        centered = centered / (std + eps)
    return centered.reshape(-1), (std.squeeze(1) <= 1e-8)
