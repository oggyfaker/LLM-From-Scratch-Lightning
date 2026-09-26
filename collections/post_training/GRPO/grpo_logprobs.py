import torch


def token_logprobs(model, sequences, pad_mask=None, temperature=1.0):
    """(B,T) ids -> (B,T-1) logprob of sequences[:,1:] under the model.

    Uses logsumexp rather than log_softmax: log_softmax would allocate a second
    (B, T-1, 151936) tensor, which at B=8 / T=640 is another 1.6 GB before the
    backward graph is even built.

    temperature divides the logits to match how the rollouts were sampled. The
    behaviour policy is pi_theta at temperature tau, so scoring at tau is what
    makes the ratio well defined.
    """
    logits = model(sequences, pad_mask=pad_mask)[:, :-1].float()
    if temperature != 1.0:
        logits = logits / temperature
    targets = sequences[:, 1:]
    selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return selected - torch.logsumexp(logits, dim=-1)
