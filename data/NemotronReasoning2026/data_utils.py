import json
import torch
from torch.utils.data import Dataset


# Empty on purpose: Nemotron prompts carry their own "put your final answer
# inside \boxed{}" instruction and the traces were generated against an empty
# system turn. A system prompt here would move every sample off distribution.
BASE_PROMPT = ""


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
def load_nemotron_json(json_path: str) -> list:
    """Load train.json / test.json.

    Each record is a dict with:
        - problem_id: str (stable id from the source dataset)
        - category: str (one of 9 task families, e.g. bit_manipulation, cipher)
        - question: str (problem statement)
        - thinking: str (reasoning chain, <think> tags stripped)
        - answer: str (final answer, \\boxed{} wrapper stripped)
        - num_gt_tokens: int (assistant ground-truth length, thinking + answer)

    The last four fields match data/Gsm8k/train.json, so both datasets drive the
    same Dataset / collate / grading code.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [
        {
            "problem_id": rec["problem_id"],
            "category": rec["category"],
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
class NemotronReasoningDataset(Dataset):
    def __init__(self, records, tokenizer, max_seq_len=None, include_thinking=True):
        """Pre-tokenize all samples into token ID lists.

        Stores prompt_length (system + user + assistant header) so the collate
        function can mask prompt positions in targets -> loss only on the answer.

        Args:
            records: list of dicts from load_nemotron_json
            tokenizer: Qwen3Tokenizer or a HuggingFace Qwen3 tokenizer
            max_seq_len: optional max sequence length filter (skip longer samples)
            include_thinking: train on the reasoning chain before the answer
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
    HERE = os.path.dirname(os.path.abspath(__file__))
    TOKENIZER_PATH = os.path.join(project_root, "collections", "qwen3", "models", "tokenizer.json")
    MAX_SEQ_LEN = 8192
    NUM_SAMPLES_TO_SHOW = 2

    tokenizer = Qwen3Tokenizer(tokenizer_file_path=TOKENIZER_PATH)

    for split in ("train", "test"):
        json_path = os.path.join(HERE, f"{split}.json")
        records = load_nemotron_json(json_path)
        print(f"\n{'=' * 70}\n{split}.json — records loaded: {len(records)}\n{'=' * 70}")

        dataset = NemotronReasoningDataset(
            records, tokenizer, max_seq_len=MAX_SEQ_LEN, include_thinking=True
        )
        lengths = [len(ids) for ids in dataset.encoded_texts]
        print(f"[thinking] samples: {len(dataset)} (dropped "
              f"{len(records) - len(dataset)} over {MAX_SEQ_LEN} tok) | "
              f"max seq len: {max(lengths)}")

        for i in range(min(NUM_SAMPLES_TO_SHOW, len(dataset))):
            full_ids, prompt_len = dataset[i]
            rec = dataset.records[i]
            print(f"  sample {i}: [{rec['category']}] {len(full_ids)} tok, "
                  f"prompt {prompt_len} tok, "
                  f"supervised {len(full_ids) - prompt_len + 1} tok, "
                  f"answer {rec['answer']!r}")
            tail = tokenizer.decode(full_ids[-60:])
            print(f"    supervised tail: {tail!r}")

        batch = [dataset[i] for i in range(min(2, len(dataset)))]
        inputs, targets = custom_collate_fn(
            batch, pad_token_id=tokenizer.eos_token_id, allowed_max_length=MAX_SEQ_LEN
        )
        supervised = (targets != -100).sum().item()
        print(f"  batch inputs {tuple(inputs.shape)} targets {tuple(targets.shape)} "
              f"| supervised positions: {supervised}")
