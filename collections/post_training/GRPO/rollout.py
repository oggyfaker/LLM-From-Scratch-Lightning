import torch
from dataclasses import dataclass

@dataclass
class RolloutBatch:
    sequences: torch.Tensor          # (N, T+M) prompt + completion ids
    pad_mask: torch.Tensor           # (N, T+M) True on padding (left and right)
    logprob_mask: torch.Tensor       # (N, T+M-1) True on generated tokens, EOS included (Shifted M-1)
    texts: list                      # decoded completions
    finish_reasons: list             # "stop" | "length"
    gen_lens: list                   # generated tokens per row
    records: list                    # source record, repeated group_size times
    group_size: int
    prompt_len: int                  # padded prompt width T
    max_completion_len: int          # M

    @property
    def n_rollouts(self):
        return self.sequences.shape[0]


def top_p_filter(probas, top_p):
    if top_p is None or top_p >= 1.0:
        return probas
    sorted_probas, sorted_idx = torch.sort(probas, dim=-1, descending=True)
    prefix = torch.cumsum(sorted_probas, dim=-1) - sorted_probas
    keep = prefix < top_p
    keep[:, 0] = True
    kept = torch.where(keep, sorted_probas, torch.zeros_like(sorted_probas))
    filtered = torch.zeros_like(probas).scatter(-1, sorted_idx, kept)
    return filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class KVCache:
    def __init__(self, n_layers):
        self.keys = [None] * n_layers
        self.values = [None] * n_layers

    def get(self, layer_idx):
        return self.keys[layer_idx], self.values[layer_idx]

    def update(self, layer_idx, keys, values):
        self.keys[layer_idx] = keys
        self.values[layer_idx] = values

    def reset(self):
        self.keys = [None] * len(self.keys)
        self.values = [None] * len(self.values)


@torch.no_grad()
def sample_rollouts(
    model, tokenizer, prompts, 
    records, group_size, max_new_tokens,
    temperature=1.0, top_p=1.0, generator=None
):
    """Sample group_size completions for each prompt -> RolloutBatch.
    """
    if temperature <= 0 and group_size > 1:
        raise ValueError(
            "Greedy decoding gives identical rollouts and zero advantage; "
            "set temperature > 0 whenever group_size > 1.")

    # 0. Device 
    device = next(model.parameters()).device
    amp = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )

    # 1. Setup EOT and PAD.  
    ## Qwen3 with eos_ids {151645, 151643}; 
    # eos_token_id 151645  '<|im_end|>' 
    # pad_token_id 151643  '<|endoftext|>'
    eos_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    pad_id = tokenizer.pad_token_id


    # 2. Repeat n_times (group_size) of batch prompts.
    ## Batched-prompts defined by PROMPTS_PER_STEP = 4. It mean each prompt in batch repeat PROMPTS_PER_STEPxtimes
    ## Total prompts input is n_batch x n_rollouts
    flat_prompts, flat_records = [], []
    for prompt, record in zip(prompts, records):
        flat_prompts.extend([prompt] * group_size)
        flat_records.extend([record] * group_size)


    # 3. Init tensorids for padding and prompts  
    ## N - total number prompts input
    ## T - max_length that prompts can have 
    ## input_ids: Base max_length can have ,create new tensor contain all promppts ids 
    ## pad_mask: same above , but the bool-matrix tensor 
    encoded = [tokenizer.encode(p) for p in flat_prompts]
    N, T = len(encoded), max(len(e) for e in encoded)
    input_ids = torch.full((N, T), pad_id, dtype=torch.long, device=device)
    pad_mask = torch.ones((N, T), dtype=torch.bool, device=device)


    # 4. Left-padding: batching all prompts into 1 tensor 
    ## input_ids - Make sure alll the final tokens of prompts appear at the end of list 
    ## this techniqe helps the model easy gather the next tokens for all prompts
    ## pad_mask - have True for padding position and False for text prompts. 
    ## Padding mask - ensure attention not calculate on padding value 
    ## Example: N=2, T=3
    ##   encoded   = [[11, 12, 13],
    ##                [21]]
    ##   input_ids = [[ 11,  12,  13],       ends at col 2
    ##                [pad, pad,  21]]       ends at col 2
    ##   pad_mask  = [[  F,   F,   F],
    ##                [  T,   T,   F]]
    for i, ids in enumerate(encoded):
        input_ids[i, T - len(ids):] = torch.tensor(ids, dtype=torch.long, device=device)
        pad_mask[i, T - len(ids):] = False


    # 5. Generation budget
    ## This a number of new tokens need to be generated. Must matched SFT max_new_tokens 
    ## (not include the length input prompts, just count the new tokens need generated)
    ## Beyond SFT new tokens length, you're in untrained territory. 
    ## You'd be sampling from a region with no learned structure and then reinforcing whatever comes out.
    budget_length = max(1, max_new_tokens)


    # 6. Prefill: Load KVcache full tokens + Generate new next tokens  
    ## Calculate the KVcache for all exist input tokens 
    cache = KVCache(len(model.transformer_blocks))
    with amp:
        logits = model(input_ids, cache=cache, start_pos=0, pad_mask=pad_mask,
                       logits_last_only=True)[:, -1]


    # 7. Decode loop: generate new tokens until EOS or the budget runs out 
    out_ids = [[] for _ in range(N)]
    finished = [False] * N
    for _ in range(budget_length):
        # --- Step 7.1: Get next token predicted for batch prompts (Sampling Train - Greedy Validation)
        if temperature > 0:
            probas = torch.softmax(logits.float() / temperature, dim=-1)
            probas = top_p_filter(probas, top_p)
            next_token = torch.multinomial(probas, num_samples=1, generator=generator).squeeze(-1)
        else:
            next_token = logits.argmax(dim=-1)

        # --- Step 7.2: Loop each sentence - Append new_token (include: normal tokens and EOS) & Bypass finished ones 
        for i, token_id in enumerate(next_token.tolist()):
             
            if not finished[i]: # Skip row finished 
                out_ids[i].append(token_id)
                finished[i] = token_id in eos_ids 

        # --- Step 7.3: If all prompts are finished, stop generate   
        if all(finished):
            break

        # --- Step 7.4: Process new tokens generated for Finished-Row and Unfinshed-Row
        ## Some of row can finished earlier , and can't be removed from the batch, damage KVcache.  
        ## next_token: So still keep finished-row with unfinished-row in batch. 
        ## Finshied-row: use (pad_token '<|endoftext|>') replace (eos_token_id '<|im_end|>') and new_tokens after EOS. Also untouch tokens before EOS.
        ## Unfinished-row: append token get from sampling. 
        ## Note: next_tokens is the tensor just have 1 token in single row. (**Thanks for KVcache without use full tokens input**)
        ## pad_mask: Tensor included prior pad-mask + 1 False (unmask) each row. Always +1 False (finished or unfinished row cause the next_tokens is covering attention) 
        next_token = torch.where(
            torch.tensor(finished, device=device),
            torch.full_like(next_token, pad_id), 
            next_token
        )
        pad_mask = torch.cat([
            pad_mask, 
            torch.zeros(N, 1, dtype=torch.bool, device=device)
        ], dim=1)

        # --- Step 7.5: Foward new tokens - start from the current position by length-1 
        with amp:
            logits = model(next_token[:, None], cache=cache,
                           start_pos=pad_mask.shape[1] - 1, pad_mask=pad_mask,
                           logits_last_only=True)[:, -1]    
    cache.reset()


    # 8. Right-padding: pack all completions into 1 tensor block (N, M)
    ## Each row stops at a different step, so out_ids have different lengths 
    ## So need 1 rectangle tensor to train on.
    ## Example: 
    ## N=2, T=3 (prompt block from step 4 = [[11, 12, 13], [pad, pad, 21]])
    ##   out_ids   = [[41, 42, 43, eos],
    ##                [51, eos]]
    ##   gen_lens  = [4, 2]
    ##   M         = max(4, 2) = 4    
    ## After the loop:
    ##   gen_block       = [[ 41,  42,  43, eos],
    ##                      [ 51, eos, pad, pad]] --> all tokens are coming from out_ids 
    ##   completion_mask = [[F, F, F | T, T, T, T],
    ##                      [F, F, F | T, T, F, F]] --> All False before '|' is for inputs_ids, after is out_ids
    gen_lens = [len(ids) for ids in out_ids]
    M = max(max(gen_lens), 1)
    gen_block = torch.full((N, M), pad_id, dtype=torch.long, device=device)
    completion_mask = torch.zeros((N, T + M), dtype=torch.bool, device=device)
    for i, ids in enumerate(out_ids):
        if ids:
            gen_block[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            completion_mask[i, T:T + len(ids)] = True

    
    # 9. Glue prompt + completion into the final (N, T+M) sequence and rebuild the pad-mask
    ## sequences - exactly what token_logprobs() re-forwards later with cache=None to get grad-able logprobs
    ## full_pad_mask - THROW AWAY the pad_mask grown inside the decode loop (step 4) and rebuild it.
    ## Example (continue from step 8): N=2, T=3, M=4
    ##   input_ids       = [[ 11,  12,  13],
    ##                      [pad, pad,  21]]
    ##   gen_block       = [[ 41,  42,  43, eos],
    ##                      [ 51, eos, pad, pad]]
    ##   sequences       = [[ 11,  12,  13 |  41,  42,  43, eos],
    ##                      [pad, pad,  21 |  51, eos, pad, pad]]          shape (N, T+M) = (2, 7)
    ##   full_pad_mask   = [[F, F, F | F, F, F, F],
    ##                      [T, T, F | F, F, T, T]] --> True on the left pad of prompt AND the right pad of completion
    sequences = torch.cat([input_ids, gen_block], dim=1)
    full_pad_mask = torch.cat(
        [pad_mask[:, :T], ~completion_mask[:, T:]], dim=1)


    # 10. Decode text + finish reason, the two things the reward function actually eats
    texts = [tokenizer.decode(ids) for ids in out_ids]
    finish_reasons = ["stop" if finished[i] else "length" for i in range(N)]
    

    # 11. Return usage:
    ## sequences: Re-forwarded through the model to get gradiant log-probs of each token.
    ## pad_mask:  Passed alongside sequences so attention ignores the left-pad (prompt) and right-pad (completion tail). Without it, padding leaks into the log-probs.
    ## logprob_mask: It's completion_mask[:, 1:], shifted by 1 because the logit at position t predicts token t+1, so token_logprobs returns (N, T+M-1). This shifted version is what the loss and the token counts actually use.
    ## max_completion_len: 
    return RolloutBatch(
        sequences=sequences, pad_mask=full_pad_mask, 
        logprob_mask=completion_mask[:, 1:], max_completion_len=M,
        texts=texts, finish_reasons=finish_reasons, gen_lens=gen_lens, records=flat_records,
        group_size=group_size, prompt_len=T
    )
