import json
import torch
from torch.utils.data import Dataset


BASE_PROMPT = "Solve the math problem. Put the final answer inside \\boxed{}."


# --------------------
# PROMPT CHAT TEMPLATE
# --------------------
def format_chat_prompt(question: str, answer: str, thinking: str = None,
                       base_prompt: str = BASE_PROMPT) -> str:
    """Format a single sample into Qwen3 chat template string.

    Direct mode (thinking=None):
        <|im_start|>
        system: {base_prompt}
        <|im_end|>

        <|im_start|>
        user: {question}
        <|im_end|>

        <|im_start|>
        assistant \\boxed{answer}
        <|im_end|>

    Thinking mode (thinking given):
        <|im_start|>
            system: {base_prompt}
        <|im_end|>

        <|im_start|>
            user: {question}
        <|im_end|>

        <|im_start|>
            assistant
            <think> {thinking} </think>
            \\boxed{answer}
        <|im_end|>
    """
    return (
        format_prompt_only(question, include_thinking=thinking is not None,
                           base_prompt=base_prompt)
        + (f"{thinking}\n</think>\n" if thinking is not None else "")
        + f"\\boxed{{{answer}}}<|im_end|>"
    )


def format_prompt_only(question: str, include_thinking: bool = False,
                       base_prompt: str = BASE_PROMPT) -> str:
    """Build the prompt portion only (for inference / loss masking).

    Direct mode ends right after the assistant header, so the model generates
    \\boxed{...} straight away.
    Thinking mode ends at <think>\\n, so the model generates the reasoning first.
    """
    head = (
        f"<|im_start|>system\n{base_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return head + ("<think>\n" if include_thinking else "")


# -------------------
# LOAD CLEAN DATASET
# -------------------
def load_gsm8k_json(json_path: str) -> list:
    """Load train.json / test.json built by download_gsm8k.py.

    Expected fields: question, thinking, answer, num_gt_tokens

    Each record is a dict with:
        - question: str (problem statement)
        - thinking: str (chain of thought, calculator annotations stripped)
        - answer: str (final numeric answer)
        - num_gt_tokens: int (assistant ground-truth length, thinking + answer)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [
        {
            "question": rec["question"],
            "thinking": rec["thinking"],
            "answer": rec["answer"],
            "num_gt_tokens": int(rec["num_gt_tokens"]),
        }
        for rec in records
    ]


# -----------------
# TORCH DATASET
# -----------------
class Gsm8kDataset(Dataset):

    def __init__(self, records, tokenizer, max_seq_len=None, include_thinking=False):
        """Pre-tokenize all samples into token ID lists.

        Stores prompt_length (system + user + assistant header) so the collate
        function can mask prompt positions in targets -> loss only on the answer.

        Args:
            records: list of dicts from load_gsm8k_json
            tokenizer: Qwen3Tokenizer instance
            max_seq_len: optional max sequence length filter (skip longer samples)
            include_thinking: train on the chain of thought before the answer
        """
        self.encoded_texts = []
        self.prompt_lengths = []
        self.records = []

        for rec in records:
            thinking = rec["thinking"] if include_thinking else None

            prompt_text = format_prompt_only(
                rec["question"], include_thinking=include_thinking
            )
            full_text = format_chat_prompt(
                rec["question"], rec["answer"], thinking=thinking
            )

            full_ids = tokenizer.encode(full_text)

            if max_seq_len is not None and len(full_ids) > max_seq_len:
                continue

            prompt_ids = tokenizer.encode(prompt_text)

            self.encoded_texts.append(full_ids)
            self.prompt_lengths.append(len(prompt_ids))
            self.records.append(rec)


    def __getitem__(self, index):
        return self.encoded_texts[index], self.prompt_lengths[index]


    def __len__(self):
        return len(self.encoded_texts)


# ----------------
# COLLATE FUNCTION
# ----------------
def custom_collate_fn(
    batch,
    pad_token_id=151643,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """Collate variable-length token lists into padded tensors.

    Appends one EOS token, pads to max_length in batch,
    creates shifted targets with ignore_index for:
      - prompt positions (system + user tokens) -> train on response only
      - padding positions (except first EOS)
    """
    max_length = max(len(item[0]) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item_ids, prompt_len in batch:
        # Append EOS and pad to max_length
        seq = list(item_ids) + [pad_token_id]
        padded = seq + ([pad_token_id] * (max_length - len(seq)))

        # --- Inputs ---
        inputs = torch.tensor(padded[:-1])

        # --- Targets (shifted by 1) ---
        targets = torch.tensor(padded[1:])

        # Mask prompt tokens -> loss only on assistant response
        if prompt_len > 1:
            targets[:prompt_len - 1] = ignore_index

        # Mask padding tokens (keep the FIRST pad = real EOS, mask the rest)
        pad_positions = torch.nonzero(targets == pad_token_id).squeeze()
        if pad_positions.numel() > 1:
            targets[pad_positions[1:]] = ignore_index

        # Truncate
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_lst = torch.stack(inputs_lst).to(device)
    targets_lst = torch.stack(targets_lst).to(device)
    return inputs_lst, targets_lst


# ----------------
# MAIN - TEST
# ----------------
if __name__ == "__main__":
    import sys
    import os

    # Add project root to path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

    # Direct import to avoid __init__.py pulling in vllm
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "qwen_tokenizer", os.path.join(project_root, "collections", "qwen3", "models", "qwen_tokenizer.py")
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    Qwen3Tokenizer = _mod.Qwen3Tokenizer

    # --- Config ---
    JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.json")
    TOKENIZER_PATH = os.path.join(project_root, "collections", "qwen3", "models", "tokenizer.json")
    MAX_SEQ_LEN = 256
    NUM_SAMPLES_TO_SHOW = 2

    tokenizer = Qwen3Tokenizer(tokenizer_file_path=TOKENIZER_PATH)
    records = load_gsm8k_json(JSON_PATH)
    print(f"Records loaded: {len(records)}")

    for include_thinking in (False, True):
        dataset = Gsm8kDataset(
            records, tokenizer,
            max_seq_len=MAX_SEQ_LEN if not include_thinking else None,
            include_thinking=include_thinking,
        )
        mode = "thinking" if include_thinking else "direct"
        lengths = [len(ids) for ids in dataset.encoded_texts]
        print(f"\n[{mode}] samples: {len(dataset)} | max seq len: {max(lengths)}")

        for i in range(min(NUM_SAMPLES_TO_SHOW, len(dataset))):
            full_ids, prompt_len = dataset[i]
            print(f"  sample {i}: total {len(full_ids)} tok, prompt {prompt_len} tok, "
                  f"supervised {len(full_ids) - prompt_len + 1} tok")
            print(f"    supervised text: {tokenizer.decode(full_ids[prompt_len:])!r}")

        batch = [dataset[i] for i in range(min(4, len(dataset)))]
        inputs, targets = custom_collate_fn(
            batch, pad_token_id=tokenizer.eos_token_id, allowed_max_length=MAX_SEQ_LEN
        )
        supervised = (targets != -100).sum().item()
        print(f"  batch inputs {tuple(inputs.shape)} targets {tuple(targets.shape)} "
              f"| supervised positions: {supervised}")
