"""Prompt-only dataset for GRPO.

GRPO never sees the ground-truth chain of thought - it only needs a prompt to
roll out from and an answer to grade against - so there are no targets, no
label masking and no CE collate here.

The prompt itself comes from data/Gsm8k/data_utils.format_prompt_only, the exact
function the SFT scripts use. Reimplementing it would silently shift the prompt
between an SFT run and a GRPO run and make the two accuracies incomparable.

The GRPO script builds two of these: all of train.json for rollouts and all of
test.json for the greedy validation rounds. Nothing is held out or sampled.
"""

import sys
from pathlib import Path

from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.Gsm8k.data_utils import format_prompt_only   # noqa: E402


class PromptDataset(Dataset):
    """GSM8K records -> (prompt, record) pairs.

    Filtered on PROMPT length, not on ground-truth length: the CoT is never fed
    in, so the SFT filter (prompt + CoT + answer <= MAX_SEQ_LEN) would drop
    perfectly usable prompts. What matters is that prompt + max_new_tokens fits.
    """

    def __init__(self, records, tokenizer, max_prompt_len=None, include_thinking=True):
        self.prompts, self.records = [], []
        for rec in records:
            prompt = format_prompt_only(rec["question"], include_thinking=include_thinking)
            if max_prompt_len is not None and len(tokenizer.encode(prompt)) > max_prompt_len:
                continue
            self.prompts.append(prompt)
            self.records.append(rec)

    def __getitem__(self, index):
        return self.prompts[index], self.records[index]

    def __len__(self):
        return len(self.prompts)


def prompt_collate_fn(batch):
    """-> (list[str] prompts, list[dict] records). Nothing is tensorised here:
    sample_rollouts tokenises and pads, because only it knows the group width."""
    prompts, records = zip(*batch)
    return list(prompts), list(records)


if __name__ == "__main__":
    import json
    sys.path.append(str(PROJECT_ROOT / "collections" / "qwen3" / "models"))
    from qwen_tokenizer import Qwen3Tokenizer

    tokenizer = Qwen3Tokenizer(
        str(PROJECT_ROOT / "collections" / "qwen3" / "models" / "tokenizer.json"))
    with open(PROJECT_ROOT / "data" / "Gsm8k" / "train.json") as f:
        records = json.load(f)

    dataset = PromptDataset(records, tokenizer, max_prompt_len=256, include_thinking=True)
    lengths = [len(tokenizer.encode(p)) for p in dataset.prompts]
    lengths.sort()
    print(f"kept {len(dataset)} prompts | prompt tokens: "
          f"min {lengths[0]} p50 {lengths[len(lengths)//2]} "
          f"p99 {lengths[int(len(lengths)*0.99)]} max {lengths[-1]}")

    prompt, record = dataset[0]
    print(f"\nanswer to grade against: {record['answer']!r}")
    print(f"prompt ends with: {prompt[-60:]!r}")
    prompts, recs = prompt_collate_fn([dataset[i] for i in range(4)])
    print(f"collate -> {len(prompts)} prompts, {len(recs)} records")
