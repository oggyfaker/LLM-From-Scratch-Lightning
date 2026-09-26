"""
GRPO rollout debugger — run one training step's data path and look at every tensor.
────────────────────────────────────────────────────────────────────────────────
Loads the merged SFT checkpoint, takes a few GSM8K train prompts, samples
GROUP_SIZE completions each, and prints the RolloutBatch plus the rewards and
advantages the training step would compute from it. Then it walks the masks of
one rollout column by column and replays backward_loss_chunk chunk by chunk.

Before all that, TOY_WALKTHROUGH runs backward_loss_chunk line by line on a
hand-made 8-row batch whose tensors are small enough to print whole.

Nothing here trains. Every parameter is inline in __main__.

    python 5_GRPO_Debug.py
"""
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLLECTIONS_DIR = PROJECT_ROOT / "collections"
MODELS_DIR = Path(__file__).resolve().parent / "models"
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(COLLECTIONS_DIR))
sys.path.append(str(MODELS_DIR))

from data.Gsm8k.data_utils import load_gsm8k_json, format_prompt_only
from post_training.GRPO import (
    group_advantages, sample_rollouts, gsm8k_reward, parse_completion,
    token_logprobs, grpo_loss, RolloutBatch,
)
from qwen3_dense import Qwen3Model, from_local_pth, QWEN_14B_CFG
from qwen_tokenizer import Qwen3Tokenizer


def show_completion(text, head=420, tail=160):
    text = text.replace("\n", "\\n")
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n        ... [{len(text) - head - tail} chars] ...\n        {text[-tail:]}"


def tok(tokenizer, token_id, width=16):
    text = repr(tokenizer.decode([token_id]))
    return text if len(text) <= width else text[:width - 4] + "...'"


def flag(value):
    return "T" if value else "F"


def layout_bar(pad_row, completion_row, width=48):
    """One char per ~(T+M)/width columns: '.' pad, 'p' prompt, 'G' generated."""
    cols = torch.linspace(0, len(pad_row) - 1, width).round().long().tolist()
    return "".join("." if pad_row[c] else "G" if completion_row[c] else "p" for c in cols)


def zoom_columns(edges, last, radius=2):
    """Columns within `radius` of each edge, clipped to [0, last]."""
    cols = set()
    for edge in edges:
        cols.update(range(edge - radius, edge + radius))
    return sorted(c for c in cols if 0 <= c <= last)


def token_role(p, left_pad, T, gen_len, stopped):
    if p < left_pad:
        return "L-pad"
    if p < T:
        return "prompt"
    if p < T + gen_len:
        return "EOS" if stopped and p == T + gen_len - 1 else "gen"
    return "R-pad"


def column_note(p, T, gen_len, width):
    if p == width - 1:
        return "no column: nothing comes after the last token"
    if p == T - 1:
        return "<- FIRST scored: last prompt token predicts gen[0]"
    if p == T + gen_len - 2:
        return "<- LAST scored: predicts the final generated token"
    if p == T + gen_len - 1:
        return "   final token predicts padding: not scored"
    return ""


def print_gap(prev, p):
    if prev is not None and p != prev + 1:
        print(f"  {'...':>5}")


# ---------------------------------------------------------------------------
# TOY WALKTHROUGH: backward_loss_chunk on a batch small enough to print whole
# ---------------------------------------------------------------------------
CELL = 7   # printed width of one token column


def build_toy_rollouts(tokenizer, prompts, answers, completions, group_size, device):
    """Hand-written completions in the exact layout sample_rollouts returns:
    prompts left-padded to T, completions (EOS included) right-padded to M."""
    pad_id = tokenizer.pad_token_id
    prompt_ids = [tokenizer.encode(p) for p in prompts for _ in range(group_size)]
    completion_ids = [tokenizer.encode(c) for c in completions]
    N = len(completion_ids)
    T = max(len(ids) for ids in prompt_ids)
    M = max(len(ids) for ids in completion_ids)

    sequences = torch.full((N, T + M), pad_id, dtype=torch.long)
    pad_mask = torch.ones((N, T + M), dtype=torch.bool)
    completion_mask = torch.zeros((N, T + M), dtype=torch.bool)
    for i, (p_ids, c_ids) in enumerate(zip(prompt_ids, completion_ids)):
        sequences[i, T - len(p_ids):T] = torch.tensor(p_ids)
        sequences[i, T:T + len(c_ids)] = torch.tensor(c_ids)
        pad_mask[i, T - len(p_ids):T + len(c_ids)] = False
        completion_mask[i, T:T + len(c_ids)] = True

    return RolloutBatch(
        sequences=sequences.to(device), pad_mask=pad_mask.to(device),
        logprob_mask=completion_mask[:, 1:].to(device),
        texts=completions, finish_reasons=["stop"] * N,
        gen_lens=[len(ids) for ids in completion_ids],
        records=[{"answer": a} for a in answers for _ in range(group_size)],
        group_size=group_size, prompt_len=T, max_completion_len=M,
    )


def cell_text(tokenizer, token_id, is_pad):
    if is_pad:
        return "<pad>"
    if token_id == tokenizer.eos_token_id:
        return "<eos>"
    text = tokenizer.decode([token_id]).replace(" ", "␣").replace("\n", "\\n")
    return text if len(text) < CELL else text[:CELL - 2] + "…"


def grid_line(label, values, T, shift=False):
    """One tensor row in token columns, '|' between prompt and completion.
    shift=True is for the T+M-1 wide tensors (logprob_mask, logprobs, grad):
    their column p is drawn under token p+1, the token it scores."""
    values = ([""] if shift else []) + [str(v) for v in values]
    cells = [f"{v:>{CELL}}" for v in values]
    return f"    {label:<14}{''.join(cells[:T])}  |{''.join(cells[T:])}"


def indent(tensor, pad=6):
    return "\n".join(" " * pad + line for line in str(tensor.detach().cpu()).splitlines())


def toy_walkthrough(model, tokenizer, device, chunk_size, max_new_tokens, grad_accum,
                    clip_eps_low, clip_eps_high, temperature, reward_weights):
    group_size = 4
    prompts, answers = ["2+3=", "12+30="], ["5", "42"]
    completions = [
        # group 0: all four right -> equal rewards -> every advantage is 0
        "\\boxed{5}<|im_end|>",
        "So \\boxed{5}<|im_end|>",
        "\\boxed{5}<|im_end|>",
        "5, so \\boxed{5}<|im_end|>",
        # group 1: right, wrong, unboxed, right -> advantages differ
        "\\boxed{42}<|im_end|>",
        "\\boxed{32}<|im_end|>",
        "42<|im_end|>",
        "\\boxed{42}<|im_end|>",
    ]
    rollouts = build_toy_rollouts(tokenizer, prompts, answers, completions, group_size, device)
    N, T, M = rollouts.n_rollouts, rollouts.prompt_len, rollouts.max_completion_len
    rewards = torch.tensor([
        gsm8k_reward(text, record["answer"], "stop", thinking_mode=False,
                     weights=reward_weights).total
        for text, record in zip(completions, rollouts.records)])
    advantages, _ = group_advantages(rewards, group_size)
    advantages = advantages.to(device)
    mask = rollouts.logprob_mask
    denom = N * max_new_tokens * grad_accum
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    torch.set_printoptions(linewidth=200)

    def text_row(i):
        return [cell_text(tokenizer, t, p) for t, p in
                zip(rollouts.sequences[i].tolist(), rollouts.pad_mask[i].tolist())]

    header = grid_line("column", range(T + M), T)
    section = f"    {'':<14}{f'prompt (T={T})':^{T * CELL}}  |{f'completion (M={M})':^{M * CELL}}"

    print()
    print("=" * 78)
    print("TOY WALKTHROUGH: backward_loss_chunk, one line at a time")
    print("=" * 78)
    print("A hand-made batch in exactly the layout sample_rollouts returns, small enough")
    print("to print every tensor whole. Real token ids, real model, real grpo_loss.")
    print(f"2 prompts x G={group_size} = N={N} rollouts, LOGPROB_CHUNK={chunk_size}. "
          f"Thinking is off so rows stay short.")

    # ---- STEP 0: the inputs
    print()
    print("STEP 0  the inputs: rollouts (sample_rollouts) and advantages (group_advantages)")
    print("-" * 78)
    for k, (prompt, answer) in enumerate(zip(prompts, answers)):
        ids = tokenizer.encode(prompt)
        print(f"  prompt {k}: {prompt!r:<9} answer {answer:<3} ids {ids}  ({len(ids)} tokens)")
    print(f"  -> T = longest prompt = {T}. Shorter prompts are LEFT-padded so every prompt "
          f"ends at column {T - 1}.")
    print()
    for i, text in enumerate(completions):
        print(f"  row {i}  group {i // group_size}  {text!r:<28} {rollouts.gen_lens[i]} tokens  "
              f"reward {rewards[i]:.2f}  advantage {advantages[i].item() + 0.0:+.2f}")
    means = rewards.view(-1, group_size).mean(1).tolist()
    print(f"  -> M = longest completion = {M}. Shorter completions are RIGHT-padded.")
    print("  -> advantage = reward - mean reward of its own group: "
          + ", ".join(f"group {g} mean {m:.2f}" for g, m in enumerate(means)))

    print(f"\n  rollouts.sequences {tuple(rollouts.sequences.shape)}, "
          f"rollouts.pad_mask {tuple(rollouts.pad_mask.shape)}, "
          f"rollouts.logprob_mask {tuple(mask.shape)}")
    print("  <pad> = <|endoftext|> used as filler (pad_mask 1). "
          "<eos> = <|im_end|>, the stop token the model generated.")
    print("  logprob_mask is one column shorter: its column p is drawn under token p+1, "
          "the token it scores.")
    print("  So its 1s sit exactly under the generated tokens, <eos> included.\n")
    print(section)
    print(header)
    for i in range(N):
        print(f"  row {i}  (group {i // group_size}, advantage {advantages[i].item() + 0.0:+.2f})")
        print(grid_line("text", text_row(i), T))
        print(grid_line("id", rollouts.sequences[i].tolist(), T))
        print(grid_line("pad_mask", rollouts.pad_mask[i].long().tolist(), T))
        print(grid_line("logprob_mask", mask[i].long().tolist(), T, shift=True))

    # ---- STEP 1: the loop
    starts = list(range(0, N, chunk_size))
    print()
    print("STEP 1  mask, total_loss = rollouts.logprob_mask, 0.0")
    print(f"        for start in range(0, rollouts.n_rollouts, LOGPROB_CHUNK)"
          f"  =  range(0, {N}, {chunk_size})  ->  start = {starts}")
    print("-" * 78)
    print("  A chunk is a block of whole ROWS (rollouts). Columns are never split: every")
    print("  chunk keeps all T+M-1 logprob columns. Advantages were computed per group")
    print("  BEFORE chunking, so a group spread over two chunks is still correct.\n")
    cols = lambda values: "".join(f"{v:>4}" for v in values)
    print(f"    {'row':<7}{cols(range(N))}")
    print(f"    {'group':<7}{cols(i // group_size for i in range(N))}")
    print(f"    {'chunk':<7}{cols(i // chunk_size for i in range(N))}")

    total_loss = 0.0
    for start in starts:
        chunk = slice(start, start + chunk_size)
        chunk_mask, chunk_adv = mask[chunk], advantages[chunk]
        rows = list(range(N))[chunk]

        print()
        print(f"---- start = {start} " + "-" * (64 - len(str(start))))
        print(f"  chunk = slice(start, start + LOGPROB_CHUNK) = slice({start}, {start + chunk_size})"
              f"   -> rows {rows}")

        print(f"\n  chunk_mask = mask[chunk]      shape {tuple(chunk_mask.shape)}   "
              f"(bool, True printed as 1)")
        print(indent(chunk_mask.long()))
        print(f"      raw column p scores token p+1, so the 1s start at column {T - 1} (the last prompt")
        print(f"      token, whose logit predicts the first generated token at column {T}):")
        for k, r in enumerate(rows):
            scored = [text_row(r)[p + 1] for p in range(T + M - 1) if chunk_mask[k, p]]
            print(f"      row {r}: {len(scored)} ones -> scores {' '.join(scored)}")

        print(f"\n  chunk_adv = advantages[chunk]      shape {tuple(chunk_adv.shape)}")
        print(indent(chunk_adv + 0.0))

        mask_sum, adv_max = int(chunk_mask.sum()), chunk_adv.abs().max().item()
        print("\n  if chunk_mask.sum() == 0 or chunk_adv.abs().max() == 0:")
        print(f"         {mask_sum} == 0 -> {mask_sum == 0}"
              f"            {adv_max:.4f} == 0 -> {adv_max == 0}")
        if mask_sum == 0 or adv_max == 0:
            group = rows[0] // group_size
            print("      continue      <- SKIP this chunk")
            print(f"      Every row here has advantage 0: group {group} scored "
                  f"{[round(v, 2) for v in rewards[group * group_size:(group + 1) * group_size].tolist()]},")
            print("      all equal, so reward - mean = 0. The loss would be exactly 0 and so would")
            print("      the gradient: no forward, no backward, no VRAM spent on these rows.")
            continue
        print("      False -> this chunk is forwarded and backpropagated")

        sequences = rollouts.sequences[chunk]
        pad_mask = rollouts.pad_mask[chunk]
        print(f"\n  sequences = rollouts.sequences[chunk]      shape {tuple(sequences.shape)}")
        print(indent(sequences))
        print(f"\n  pad_mask = rollouts.pad_mask[chunk]        shape {tuple(pad_mask.shape)}   "
              f"(True printed as 1)")
        print(indent(pad_mask.long()))

        with torch.no_grad(), amp:
            logprobs = token_logprobs(model, sequences, pad_mask=pad_mask, temperature=temperature)
        # A leaf tensor stands in for the network: its .grad is exactly what
        # manual_backward pushes back through token_logprobs into the LoRA weights.
        logprobs = logprobs.float().requires_grad_()
        print(f"\n  logprobs = token_logprobs(self.model, sequences, pad_mask, temperature)"
              f"      shape {tuple(logprobs.shape)}")
        print("      column p = log P(token p+1 | tokens 0..p), drawn under token p+1."
              " Every column is computed;")
        print("      chunk_mask decides which ones reach the loss.\n")
        print(section)
        print(header)
        for k, r in enumerate(rows):
            print(f"  row {r}")
            print(grid_line("text", text_row(r), T))
            print(grid_line("logprobs", [f"{v:.2f}" for v in logprobs[k].tolist()], T, shift=True))
            print(grid_line("chunk_mask", chunk_mask[k].long().tolist(), T, shift=True))

        loss = grpo_loss(
            logprobs, chunk_adv, chunk_mask,
            max_completion_len=max_new_tokens, num_rollouts=N, grad_accum=grad_accum,
            clip_eps_low=clip_eps_low, clip_eps_high=clip_eps_high,
        )
        print(f"\n  loss = grpo_loss(logprobs, chunk_adv, chunk_mask, max_completion_len={max_new_tokens},")
        print(f"                   num_rollouts={N}, grad_accum={grad_accum}, "
              f"clip_eps_low={clip_eps_low}, clip_eps_high={clip_eps_high})")
        print("      ratio     = exp(logprobs - logprobs.detach()) = 1 on every column (on-policy)")
        print("      per_token = -min(ratio * adv, clip(ratio) * adv) = -adv")
        print("      loss      = sum(per_token * chunk_mask) / (num_rollouts x max_completion_len x grad_accum)")
        for k, r in enumerate(rows):
            n, a = int(chunk_mask[k].sum()), chunk_adv[k].item() + 0.0
            print(f"          row {r}: -({a:+.4f}) x {n} ones in chunk_mask = {-a * n + 0.0:+.4f}")
        total = -(chunk_adv * chunk_mask.sum(1)).sum().item() + 0.0
        print(f"          {total:+.4f} / ({N} x {max_new_tokens} x {grad_accum} = {denom}) "
              f"= {total / denom:+.4e}")
        print(f"      grpo_loss returned {loss.item():+.4e}")

        loss.backward()
        print("\n  self.manual_backward(loss)")
        print(f"      gradient reaching each logprob, x{denom} so it is readable "
              f"(raw value = shown / {denom}):\n")
        print(section)
        print(header)
        for k, r in enumerate(rows):
            grad = [(v * denom) + 0.0 for v in logprobs.grad[k].tolist()]
            print(f"  row {r}  (advantage {chunk_adv[k].item() + 0.0:+.2f})")
            print(grid_line("text", text_row(r), T))
            print(grid_line(f"grad x{denom}", ["0" if v == 0 else f"{v:+.2f}" for v in grad],
                            T, shift=True))
        for k, r in enumerate(rows):
            a = chunk_adv[k].item()
            verdict = "MORE" if a > 0 else "LESS"
            print(f"      row {r}: gradient {-a:+.2f}/{denom} on its {int(chunk_mask[k].sum())} "
                  f"tokens -> the optimizer step makes them {verdict} likely")
        print("      prompt and pad columns get 0: chunk_mask is 0 there. In training this")
        print("      gradient keeps flowing back through the model into the LoRA A/B weights.")

        total_loss += loss.item()
        print(f"\n  total_loss += loss.item()      ->  total_loss = {total_loss:+.4e}")

    print()
    print("-" * 78)
    print(f"  return total_loss = {total_loss:+.4e}")
    print()
    print("  Skips come from chunk_adv being all 0 (a group whose rollouts all scored the")
    print("  same). chunk_mask.sum() == 0 is only a guard: sample_rollouts always generates")
    print("  at least one token per row. A chunk with SOME zero-advantage rows still runs;")
    print("  those rows just add 0 to the loss.")
    print("  Why chunk at all: token_logprobs builds float32 logits of shape")
    print("  (rows, T+M-1, 151936). At training size (32 rollouts x ~640 columns) that is")
    print(f"  ~{32 * 639 * 151936 * 4 / 1e9:.0f} GB for all rows at once, "
          f"~{4 * 639 * 151936 * 4 / 1e9:.1f} GB per chunk of 4, before the backward graph.")


if __name__ == '__main__':

    # ------------------------------------------------------------ PARAMETERS
    CHECKPOINT = (PROJECT_ROOT / "collections/qwen3/logs/Qwen3_Gsm8k_14B_QLoRA_4bit_think"
                  / "20_09_26/version_0/model_pretrained/00-0.4334-87.11.pth")
    MODEL_CFG = QWEN_14B_CFG

    NUM_PROMPTS = 2           # prompts drawn from train.json
    PROMPT_OFFSET = 0         # which slice of train.json
    GROUP_SIZE = 4            # G rollouts per prompt  ->  N = NUM_PROMPTS * G
    MAX_NEW_TOKENS = 384      # completion cap; GSM8K thinking ground truth maxes at 346
    TEMPERATURE = 1.0         # must be > 0 or all G rollouts are identical
    TOP_P = 1.0
    INCLUDE_THINKING = True
    REWARD_WEIGHTS = {"correct": 1.0, "format": 0.2}
    SEED = 1001
    DEVICE = "cuda"
    DTYPE = torch.bfloat16    # 14B bf16 = ~28 GB; the training script runs it in 4-bit

    # backward_loss_chunk replay. Training uses G=8 / chunk=4, i.e. each group
    # spans 2 chunks; G=4 / chunk=2 keeps that shape at debug size.
    LOGPROB_CHUNK = 2
    GRAD_ACCUM = 4
    CLIP_EPS_LOW = 0.2
    CLIP_EPS_HIGH = 0.28
    MASK_ROW = None           # rollout to zoom into; None = auto-pick one with padding and gradient
    TOY_WALKTHROUGH = True    # backward_loss_chunk line by line on a hand-made batch

    torch.manual_seed(SEED)

    # ------------------------------------------------------ MODEL & TOKENIZER
    print("=" * 78)
    print("LOADING")
    print("=" * 78)
    tokenizer = Qwen3Tokenizer(str(MODELS_DIR / "tokenizer.json"))
    print(f"tokenizer   eos_ids={sorted({tokenizer.eos_token_id, tokenizer.pad_token_id})}  "
          f"pad_id={tokenizer.pad_token_id}  (derived inside sample_rollouts)")

    t0 = time.time()
    model = Qwen3Model(MODEL_CFG)
    model = from_local_pth(model, str(CHECKPOINT))
    model = model.to(device=DEVICE, dtype=DTYPE).eval()
    model.dtype = DTYPE
    params = sum(p.numel() for p in model.parameters())
    print(f"model       {params/1e9:.2f}B params, {DTYPE}, loaded in {time.time()-t0:.1f}s")
    print(f"gpu         {torch.cuda.memory_allocated()/1024**3:.1f} GB allocated")

    # ------------------------------------------------------ TOY WALKTHROUGH
    if TOY_WALKTHROUGH:
        toy_walkthrough(
            model, tokenizer, DEVICE,
            chunk_size=LOGPROB_CHUNK, max_new_tokens=MAX_NEW_TOKENS, grad_accum=GRAD_ACCUM,
            clip_eps_low=CLIP_EPS_LOW, clip_eps_high=CLIP_EPS_HIGH,
            temperature=TEMPERATURE, reward_weights=REWARD_WEIGHTS,
        )

    # ------------------------------------------------------------- PROMPTS
    records = load_gsm8k_json(str(PROJECT_ROOT / "data" / "Gsm8k" / "train.json"))
    records = records[PROMPT_OFFSET:PROMPT_OFFSET + NUM_PROMPTS]
    prompts = [format_prompt_only(r["question"], include_thinking=INCLUDE_THINKING)
               for r in records]

    print()
    print("=" * 78)
    print("PROMPTS")
    print("=" * 78)
    for i, (prompt, record) in enumerate(zip(prompts, records)):
        print(f"[{i}] answer={record['answer']!r}  prompt_tokens={len(tokenizer.encode(prompt))}")
        print(f"    {record['question']}")
    print(f"\nN = {NUM_PROMPTS} prompts x G={GROUP_SIZE} = {NUM_PROMPTS * GROUP_SIZE} rollouts")

    # ------------------------------------------------------------- ROLLOUTS
    print()
    print("=" * 78)
    print("sample_rollouts(...)")
    print("=" * 78)
    t0 = time.time()
    outputs = sample_rollouts(
        model, tokenizer, prompts, records,
        group_size=GROUP_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    elapsed = time.time() - t0

    # RolloutBatch only carries logprob_mask = completion_mask[:, 1:]. Column 0 is
    # always a prompt token, so putting a False column back in front recovers it.
    completion_mask = torch.cat(
        [torch.zeros_like(outputs.logprob_mask[:, :1]), outputs.logprob_mask], dim=1)

    print(f"elapsed            {elapsed:.1f}s  "
          f"({sum(outputs.gen_lens)} tokens, {sum(outputs.gen_lens)/elapsed:.0f} tok/s)")
    print(f"peak gpu           {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")
    print()
    print(f"sequences          {tuple(outputs.sequences.shape)}  {outputs.sequences.dtype}")
    print(f"pad_mask           {tuple(outputs.pad_mask.shape)}  {outputs.pad_mask.dtype}")
    print(f"completion_mask    {tuple(completion_mask.shape)}  {completion_mask.dtype}  "
          f"(rebuilt here, not stored in RolloutBatch)")
    print(f"logprob_mask       {tuple(outputs.logprob_mask.shape)}   <- completion_mask[:, 1:]")
    print(f"prompt_len  T      {outputs.prompt_len}")
    print(f"max_completion_len M {outputs.max_completion_len}")
    print(f"group_size         {outputs.group_size}")
    print(f"gen_lens           {outputs.gen_lens}")
    print(f"finish_reasons     {outputs.finish_reasons}")
    print(f"records            {len(outputs.records)} entries, "
          f"answers={[r['answer'] for r in outputs.records]}")
    print(f"scored tokens      {int(outputs.logprob_mask.sum())} "
          f"(= sum(gen_lens) = {sum(outputs.gen_lens)})")
    print(f"mask overlap       {int((outputs.pad_mask & completion_mask).sum())} "
          f"(must be 0)")

    # --------------------------------------------------------------- REWARDS
    print()
    print("=" * 78)
    print("COMPLETIONS + REWARDS")
    print("=" * 78)
    all_rewards = []
    for g in range(0, outputs.n_rollouts, GROUP_SIZE):
        group = slice(g, g + GROUP_SIZE)
        answer = outputs.records[g]["answer"]
        # gsm8k_reward per rollout (not reward_groups) to keep label/prediction
        reward_outs = [
            gsm8k_reward(text, answer, finish_reason,
                         thinking_mode=INCLUDE_THINKING, weights=REWARD_WEIGHTS)
            for text, finish_reason in zip(outputs.texts[group],
                                           outputs.finish_reasons[group])
        ]
        all_rewards.extend(o.total for o in reward_outs)

        print(f"\n{'-'*78}")
        print(f"GROUP {g // GROUP_SIZE}  |  ground truth = {answer!r}")
        print(f"{'-'*78}")
        for k, (text, reward) in enumerate(zip(outputs.texts[group], reward_outs)):
            row = g + k
            parsed = parse_completion(text, outputs.finish_reasons[row], INCLUDE_THINKING)
            print(f"\n  [rollout {k}] len={outputs.gen_lens[row]:>3} "
                  f"finish={outputs.finish_reasons[row]:<6} "
                  f"label={reward.label:<7} pred={str(reward.prediction):<8} "
                  f"reward={reward.total:.2f}"
                  + (f"  bad_reason={reward.bad_reason}" if reward.bad_reason else ""))
            print(f"      </think>={parsed.n_think_close}  boxed_after_think={parsed.n_boxed}")
            print(f"      {show_completion(text)}")
        group_r = torch.tensor([o.total for o in reward_outs])
        print(f"\n  group reward_mean={group_r.mean():.2f}  "
              f"reward_std={group_r.std(unbiased=False):.2f}")

    # ------------------------------------------------------------ ADVANTAGES
    print()
    print("=" * 78)
    print("ADVANTAGES (what the loss actually multiplies the log-probs by)")
    print("=" * 78)
    rewards = torch.tensor(all_rewards)
    advantages, degenerate = group_advantages(rewards, GROUP_SIZE, scale_by_std=False)
    for g in range(0, len(rewards), GROUP_SIZE):
        idx = g // GROUP_SIZE
        group_r = rewards[g:g + GROUP_SIZE]
        print(f"group {idx}: rewards={[round(v, 2) for v in group_r.tolist()]}  "
              f"mean={group_r.mean():.3f}  ->  "
              f"advantages={[round(v, 3) for v in advantages[g:g+GROUP_SIZE].tolist()]}"
              + ("   <- DEGENERATE: zero gradient" if degenerate[idx] else ""))
    print(f"\ndegenerate groups: {int(degenerate.sum())}/{len(degenerate)}")
    print(f"reward mean {rewards.mean():.3f}  std {rewards.std(unbiased=False):.3f}")

    # ----------------------------------------------------------------- MASKS
    N, T = outputs.n_rollouts, outputs.prompt_len
    width = outputs.sequences.shape[1]               # T + M
    mask = outputs.logprob_mask

    print()
    print("=" * 78)
    print("MASKS (one row = one rollout = prompt + completion)")
    print("=" * 78)
    print(f"sequences        (N, T+M)   = {tuple(outputs.sequences.shape)}: "
          f"prompt left-padded to T={T}, completion right-padded to M={width - T}")
    print("pad_mask         (N, T+M)   True = padding -> attention ignores it")
    print("completion_mask  (N, T+M)   True = token the policy sampled (EOS included)")
    print("logprob_mask     (N, T+M-1) = completion_mask[:, 1:] -> the only mask the loss uses")
    print()
    print("Why the shift: token_logprobs keeps logits[:, :-1] and targets sequences[:, 1:],")
    print("because the logit at position p predicts token p+1. The last logit predicts")
    print("nothing and token 0 has nothing predicting it, so it returns T+M-1 columns and")
    print("column p holds log P(sequences[p+1] | sequences[:p+1]).")

    print(f"\nlayout of every row  ('.' pad, 'p' prompt, 'G' generated; 1 char ~ {width / 48:.0f} tokens)")
    print(f"  {'row':>3} {'grp':>3} {'L-pad':>6} {'prompt':>6} {'gen':>4} {'R-pad':>5} "
          f"{'scored':>6}  {'finish':<6}")
    for i in range(N):
        left = int(outputs.pad_mask[i, :T].sum())
        right = int(outputs.pad_mask[i, T:].sum())
        print(f"  {i:>3} {i // GROUP_SIZE:>3} {left:>6} {T - left:>6} {outputs.gen_lens[i]:>4} "
              f"{right:>5} {int(mask[i].sum()):>6}  {outputs.finish_reasons[i]:<6} "
              f"{layout_bar(outputs.pad_mask[i], completion_mask[i])}")

    # Prefer a row that carries gradient, then the one with the most padding, so
    # the zoom shows left pad, prompt, completion and right pad in one row.
    if MASK_ROW is None:
        score = outputs.pad_mask.sum(1).cpu() + (advantages != 0) * width
        zoom_row = int(score.argmax())
    else:
        zoom_row = MASK_ROW
    zoom_left = int(outputs.pad_mask[zoom_row, :T].sum())
    zoom_gen = outputs.gen_lens[zoom_row]
    zoom_seq = outputs.sequences[zoom_row].tolist()
    zoom_stop = outputs.finish_reasons[zoom_row] == "stop"

    print(f"\nzoom on row {zoom_row} (group {zoom_row // GROUP_SIZE}, "
          f"finish={outputs.finish_reasons[zoom_row]}) around its edges")
    print(f"  {'':<50}| column p of token_logprobs + logprob_mask")
    print(f"  {'pos':>4}  {'role':<6} {'id':>6}  {'token':<16}  {'pad':^3}  {'comp':^4}  |"
          f"  {'scores p+1':<16}  {'lp_mask':^7}")
    prev = None
    edges = (zoom_left, T, T + zoom_gen)
    for p in zoom_columns(edges, last=width - 1) + ([width - 1] if width - 1 > T + zoom_gen + 1 else []):
        print_gap(prev, p)
        prev = p
        line = (f"  {p:>4}  {token_role(p, zoom_left, T, zoom_gen, zoom_stop):<6} "
                f"{zoom_seq[p]:>6}  {tok(tokenizer, zoom_seq[p]):<16}  "
                f"{flag(outputs.pad_mask[zoom_row, p]):^3}  {flag(completion_mask[zoom_row, p]):^4}  |")
        if p < width - 1:
            line += (f"  {tok(tokenizer, zoom_seq[p + 1]):<16}  {flag(mask[zoom_row, p]):^7}  "
                     f"{column_note(p, T, zoom_gen, width)}")
        else:
            line += f"  {'-':<16}  {'-':^7}  {column_note(p, T, zoom_gen, width)}"
        print(line)
    print("\nPad and EOS can share the id <|endoftext|>; the masks, not the ids, tell them apart.")

    print("\nchecks")
    print(f"  logprob_mask == completion_mask[:, 1:]       "
          f"{bool(torch.equal(mask, completion_mask[:, 1:]))}")
    print(f"  pad_mask & completion_mask overlap          "
          f"{int((outputs.pad_mask & completion_mask).sum())}")
    print(f"  logprob_mask.sum(1) == gen_lens             "
          f"{mask.sum(1).tolist() == outputs.gen_lens}")

    # ------------------------------------------------------ CHUNKED BACKWARD
    print()
    print("=" * 78)
    print("backward_loss_chunk (replayed chunk by chunk; no weights change)")
    print("=" * 78)
    advantages = advantages.to(DEVICE)
    denom = N * MAX_NEW_TOKENS * GRAD_ACCUM
    loss_kwargs = dict(max_completion_len=MAX_NEW_TOKENS, grad_accum=GRAD_ACCUM,
                       clip_eps_low=CLIP_EPS_LOW, clip_eps_high=CLIP_EPS_HIGH)
    chunks = [slice(s, min(s + LOGPROB_CHUNK, N)) for s in range(0, N, LOGPROB_CHUNK)]
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    print(f"Chunks slice ROWS (rollouts), never columns: every chunk keeps all "
          f"T+M-1={width - 1} columns.")
    print(f"LOGPROB_CHUNK={LOGPROB_CHUNK} -> {len(chunks)} chunks for N={N} "
          f"(GROUP_SIZE={GROUP_SIZE}). Advantages were normalised over whole")
    print("groups BEFORE chunking, so a group split across chunks is still correct.")
    print()
    cell = lambda values: "".join(f"{v:>8}" for v in values)
    print(f"  {'row':<10}{cell(range(N))}")
    print(f"  {'group':<10}{cell(i // GROUP_SIZE for i in range(N))}")
    print(f"  {'chunk':<10}{cell(i // LOGPROB_CHUNK for i in range(N))}")
    print(f"  {'reward':<10}{cell(f'{v:.2f}' for v in rewards.tolist())}")
    print(f"  {'advantage':<10}{cell(f'{v:+.3f}' for v in advantages.tolist())}")
    print(f"  {'scored':<10}{cell(mask.sum(1).tolist())}")
    print(f"\nloss denominator = N x MAX_NEW_TOKENS x GRAD_ACCUM = "
          f"{N} x {MAX_NEW_TOKENS} x {GRAD_ACCUM} = {denom}")
    print("Every chunk divides by the whole-batch N, so the chunk losses add up to the full loss.")

    total_loss = 0.0
    all_logprobs = torch.zeros(mask.shape, device=DEVICE)
    all_grads = torch.zeros(mask.shape, device=DEVICE)
    for c, chunk in enumerate(chunks):
        rows = range(N)[chunk]
        chunk_mask, chunk_adv = mask[chunk], advantages[chunk]
        n_scored = chunk_mask.sum(1)
        skip = chunk_mask.sum() == 0 or chunk_adv.abs().max() == 0

        print(f"\n{'-'*78}")
        print(f"CHUNK {c}  rows {rows[0]}..{rows[-1]}  "
              f"-> slice({chunk.start}, {chunk.stop}) on mask, advantages, sequences, pad_mask")
        print(f"{'-'*78}")
        print(f"  chunk_mask      {tuple(chunk_mask.shape)}  scored per row {n_scored.tolist()} "
              f"-> sum {int(n_scored.sum())}")
        print(f"  chunk_adv       {[round(v, 3) for v in chunk_adv.tolist()]}  "
              f"|adv|.max() = {chunk_adv.abs().max():.3f}")
        if skip:
            print("  skip check      mask.sum() == 0 or |adv|.max() == 0 -> continue: "
                  "training never forwards these rows")
        else:
            print("  skip check      mask.sum() > 0 and |adv|.max() > 0 -> forward + backward")

        sequences, pad_mask = outputs.sequences[chunk], outputs.pad_mask[chunk]
        with torch.no_grad(), amp:
            logprobs = token_logprobs(model, sequences, pad_mask=pad_mask,
                                      temperature=TEMPERATURE)
        # A leaf tensor stands in for the network: its .grad is exactly the signal
        # manual_backward would push back through token_logprobs into the LoRA weights.
        logprobs = logprobs.float().requires_grad_()
        loss = grpo_loss(logprobs, chunk_adv, chunk_mask, num_rollouts=N, **loss_kwargs)
        loss.backward()
        all_logprobs[chunk], all_grads[chunk] = logprobs.detach(), logprobs.grad

        print(f"  sequences       {tuple(sequences.shape)}   pad_mask {tuple(pad_mask.shape)}"
              + ("   (forwarded here only to show skipping is safe)" if skip else ""))
        print(f"  token_logprobs  {tuple(logprobs.shape)}   every column is computed; "
              f"chunk_mask picks the ones that count")
        print("  grpo_loss       on-policy ratio = exp(lp - lp.detach()) = 1, "
              "so per-token loss = -adv on scored columns")
        for i, row in enumerate(rows):
            n_i, adv_i = int(n_scored[i]), chunk_adv[i].item() + 0.0
            lp_mean = (logprobs[i].detach() * chunk_mask[i]).sum().item() / max(n_i, 1)
            print(f"                  row {row}: -({adv_i:+.3f}) x {n_i:>3} tokens = "
                  f"{-adv_i * n_i + 0.0:+9.3f}   (mean scored logprob {lp_mean:.3f})")
        contrib = -(chunk_adv * n_scored).sum().item() + 0.0
        print(f"                  sum {contrib:+.3f} / {denom} = {contrib / denom:+.6f}")
        print(f"  loss            {loss.item():+.6f}   (its value ignores the logprobs; "
              f"only the gradient uses them)")

        for i, row in enumerate(rows):
            grad_on = logprobs.grad[i][chunk_mask[i]]
            grad_off = logprobs.grad[i][~chunk_mask[i]]
            adv_i = chunk_adv[i].item()
            direction = "UP" if adv_i > 0 else "DOWN" if adv_i < 0 else "nowhere"
            off = grad_off.abs().max().item()
            print(f"  dloss/dlogprob  row {row}: "
                  f"{(grad_on[0].item() if len(grad_on) else 0.0) + 0.0:+.2e} on its {len(grad_on)} "
                  f"scored columns, {'0' if off == 0 else f'up to {off:.1e}'} on the other "
                  f"{len(grad_off)} -> sampled tokens pushed {direction}")

        if zoom_row in rows:
            i = zoom_row - chunk.start
            print(f"\n  zoom row {zoom_row}: the columns where scoring starts and stops")
            print(f"  {'col':>5}  {'scores token':<16}  {'logprob':>8}  {'lp_mask':^7}  "
                  f"{'dloss/dlogprob':>14}")
            prev = None
            for p in zoom_columns((T - 1, T + zoom_gen - 1), last=width - 2):
                print_gap(prev, p)
                prev = p
                print(f"  {p:>5}  {tok(tokenizer, zoom_seq[p + 1]):<16}  "
                      f"{logprobs[i, p].item():>8.3f}  {flag(chunk_mask[i, p]):^7}  "
                      f"{logprobs.grad[i, p].item() + 0.0:>+14.2e}")
            print()

        if skip:
            print(f"  manual_backward skipped: loss {loss.item():+.6f}, "
                  f"max |grad| {logprobs.grad.abs().max().item():.0e}, so nothing is lost")
        else:
            total_loss += loss.item()
            print("  manual_backward(loss)   not run here; in training it sends the gradients "
                  "above into LoRA A/B")
            print(f"  total_loss      {total_loss:+.6f}")

    # ------------------------------------------------ CHUNKED == WHOLE BATCH
    print()
    print("=" * 78)
    print("CHUNKED == WHOLE BATCH")
    print("=" * 78)
    whole_logprobs = all_logprobs.clone().requires_grad_()
    whole = grpo_loss(whole_logprobs, advantages, mask, num_rollouts=N, **loss_kwargs)
    whole.backward()
    per_chunk_n = sum(grpo_loss(all_logprobs[c], advantages[c], mask[c], **loss_kwargs).item()
                      for c in chunks)
    print(f"total_loss summed over chunks          {total_loss:+.6f}")
    print(f"grpo_loss over all {N} rows at once     {whole.item():+.6f}")
    print(f"max |gradient difference|              "
          f"{(whole_logprobs.grad - all_grads).abs().max().item():.1e}")
    print(f"chunks without num_rollouts=N          {per_chunk_n:+.6f}   <- each chunk divides by "
          f"its own row count, not N")
    print(f"\nTraining repeats this GRAD_ACCUM={GRAD_ACCUM} times (one per generation call) while")
    print(".grad keeps adding up, then takes one optimizer step. The /GRAD_ACCUM in the")
    print("denominator turns that sum into a mean.")
