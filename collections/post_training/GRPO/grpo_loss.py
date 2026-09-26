import torch


# ------------
# LOSS PIECES
# ------------
def loss_normalizer(loss_type, n_seq, max_completion_len, num_tokens, grad_accum=1):
    """What the summed per-token loss of ONE generation call is divided by.

        dr_grpo  n_seq * max_completion_len * grad_accum. max_completion_len is
                 the CONSTANT generation budget (MAX_NEW_TOKENS), not the
                 batch's longest rollout: a batch where everything stopped at
                 50 tokens would otherwise get an 8x larger gradient than one
                 where a rollout ran to 400.
        bnpo     num_tokens * grad_accum: a plain mean over every completion
                 token of the call (TRL / Unsloth "bnpo"). num_tokens counts
                 the WHOLE rollout batch, so a chunked backward divides every
                 chunk by the same number and sums to the whole-batch loss.

    grad_accum turns the accumulated gradient into a mean over the generation
    calls of one optimizer step rather than a sum.
    """
    if loss_type == "dr_grpo":
        return float(n_seq * max_completion_len * grad_accum)
    if loss_type == "bnpo":
        return float(max(1, num_tokens) * grad_accum)
    raise ValueError(f"Unknown loss_type: {loss_type}. Use 'bnpo' or 'dr_grpo'.")


def grpo_loss(policy_logprobs, advantages, completion_mask, max_completion_len,
              num_rollouts=None, grad_accum=1, old_logprobs=None,
              clip_eps_low=0.2, clip_eps_high=0.28,
              loss_type="dr_grpo", num_tokens=None,
              ref_logprobs=None, kl_beta=0.0, return_stats=False):
    """-> loss (or (loss, stats) with return_stats). All tensors are (B, T-1)
    except advantages, which is (B,).

        num_rollouts   N of the WHOLE rollout batch (dr_grpo). Pass it when
                       backwarding chunk by chunk, so every chunk divides by the
                       same N. Defaults to this call's batch size.
        num_tokens     completion tokens of the WHOLE rollout batch (bnpo).
                       Defaults to this call's mask.
        ref_logprobs   log-probs under the frozen reference policy. With
                       kl_beta > 0 the k3 estimator of KL(pi_theta || pi_ref)
                       is added per token, as TRL / Unsloth do.

    stats: policy_loss (the loss without the KL term) and kl_per_seq, the
    token-mean KL of every row (None without ref_logprobs), both detached.
    """
    mask = completion_mask.float()
    advantages = advantages.detach().unsqueeze(1)

    if old_logprobs is None:
        # Strictly on-policy: rho == 1 by construction. Keeping the exp() of a
        # zero difference preserves the gradient path through policy_logprobs.
        ratio = torch.exp(policy_logprobs - policy_logprobs.detach())
    else:
        ratio = torch.exp(policy_logprobs - old_logprobs)

    clipped = ratio.clamp(1.0 - clip_eps_low, 1.0 + clip_eps_high)
    per_token = -torch.min(ratio * advantages, clipped * advantages)

    normalizer = loss_normalizer(
        loss_type,
        n_seq=num_rollouts or completion_mask.shape[0],
        max_completion_len=max_completion_len,
        num_tokens=int(mask.sum().item()) if num_tokens is None else num_tokens,
        grad_accum=grad_accum,
    )
    policy_loss = (per_token * mask).sum() / normalizer
    loss = policy_loss

    kl_per_seq = None
    if ref_logprobs is not None:
        # k3 = exp(ref - pi) - (ref - pi) - 1 >= 0: unbiased, low variance.
        log_ratio = ref_logprobs.detach() - policy_logprobs
        per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0
        if kl_beta:
            loss = loss + kl_beta * (per_token_kl * mask).sum() / normalizer
        kl_per_seq = ((per_token_kl * mask).sum(1) / mask.sum(1).clamp_min(1.0)).detach()

    if return_stats:
        return loss, {"policy_loss": policy_loss.detach(), "kl_per_seq": kl_per_seq}
    return loss
