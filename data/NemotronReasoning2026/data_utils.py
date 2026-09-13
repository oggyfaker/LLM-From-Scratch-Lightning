import csv
import torch
from torch.utils.data import Dataset, DataLoader


# --------------------
# PROMPT CHAT TEMPLATE
# --------------------
def format_chat_prompt(user_content: str, thinking: str, final_answer: str) -> str:
    """Apply Qwen3 chat template to a clean record.

    Reconstructs the EXACT original format from nemotron_decoded.csv:
        <|im_start|>system
        <|im_end|>
        <|im_start|>user
        {user_content}<|im_end|>
        <|im_start|>assistant
        <think>
        {thinking}
        </think>
        \\boxed{final_answer}<|im_end|>

    Args:
        user_content: Clean user question
        thinking: Full reasoning chain (already includes answer tail from CSV)
        final_answer: Just the answer (without \\boxed{} wrapper — will be added)

    Returns:
        Full text with chat template applied (prompt + ground_truth combined).
    """
    return (
        f"<|im_start|>system\n<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
        f"{thinking}\n</think>\n\\boxed{{{final_answer}}}<|im_end|>"
    )


def format_prompt_only(user_content: str) -> str:
    """Build the prompt portion only (for inference / generation).

    Returns:
        The input prompt that ends with <think>\\n (model generates from here).
    """
    return (
        f"<|im_start|>system\n<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )


# -------------------
# LOAD CLEAN DATASET
# -------------------
def load_nemotron_csv(csv_path: str) -> list:
    """Load nemotron_decoded_clean.csv (already cleaned by clean_csv.py).

    Expected columns: problem_id, user_content, thinking, final_answer,
                      num_tokens, num_prompt_tokens, num_gt_tokens

    Each record is a dict with:
        - problem_id: str
        - user_content: str (clean question)
        - thinking: str (reasoning chain)
        - final_answer: str (e.g. \\\\boxed{...})
        - num_tokens: int
    """
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "problem_id": row["problem_id"],
                "user_content": row["user_content"],
                "thinking": row["thinking"],
                "final_answer": row["final_answer"],
                "num_tokens": int(row["num_tokens"]),
            })
    return records


# -----------------
# TORCH DATASET
# -----------------
class NemotronReasoningDataset(Dataset):
    def __init__(self, records, tokenizer, max_seq_len=None):
        """Pre-tokenize all samples into token ID lists.

        Stores prompt_length (system + user tokens) so the collate function
        can mask prompt positions in targets -> loss only on assistant response.

        Args:
            records: list of dicts from load_nemotron_csv
            tokenizer: Qwen3Tokenizer instance
            max_seq_len: optional max sequence length filter (skip longer samples)
        """
        self.encoded_texts = []
        self.prompt_lengths = []
        self.records = []

        for rec in records:
            # Build prompt-only text (system + user + assistant start)
            prompt_text = format_prompt_only(rec["user_content"])

            # Build full text (prompt + response) with chat template applied
            full_text = format_chat_prompt(
                rec["user_content"],
                rec["thinking"],
                rec["final_answer"],
            )

            full_ids = tokenizer.encode(full_text)

            # Skip if exceeds max_seq_len
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
    CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nemotron_decoded_clean.csv")
    TOKENIZER_PATH = os.path.join(project_root, "collections", "qwen3", "models", "tokenizer.json")
    BATCH_SIZE = 2
    MAX_SEQ_LEN = 4096  # Filter out very long sequences for testing
    NUM_SAMPLES_TO_SHOW = 2

    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=TOKENIZER_PATH)

    print(f"Loading clean dataset from: {CSV_PATH}")
    records = load_nemotron_csv(CSV_PATH)
    print(f"Total records loaded: {len(records)}")

    # Show a sample with chat template applied
    print("\n" + "=" * 80)
    print("SAMPLE DATA (Record 0) - Chat template applied")
    print("=" * 80)
    rec = records[0]
    print(f"Problem ID: {rec['problem_id']}")
    print(f"\nUser Content (first 300 chars):\n{rec['user_content'][:300]}")
    print(f"\nThinking (first 300 chars):\n{rec['thinking'][:300]}")
    print(f"\nFinal Answer:\n{rec['final_answer']}")
    print(f"\n--- format_chat_prompt output (first 400 chars) ---")
    full = format_chat_prompt(rec["user_content"], rec["thinking"], rec["final_answer"])
    print(full[:400])
    print(f"\n--- format_prompt_only output ---")
    print(format_prompt_only(rec["user_content"])[:300])

    # Build dataset
    print("\n" + "=" * 80)
    print(f"Building NemotronReasoningDataset (max_seq_len={MAX_SEQ_LEN})...")
    print("=" * 80)
    dataset = NemotronReasoningDataset(records, tokenizer, max_seq_len=MAX_SEQ_LEN)
    print(f"Dataset size (after filtering): {len(dataset)}")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: custom_collate_fn(
            batch,
            pad_token_id=tokenizer.pad_token_id,
            allowed_max_length=MAX_SEQ_LEN,
        ),
    )

    # Iterate and decode
    print("\n" + "=" * 80)
    print(f"DataLoader: {len(dataloader)} batches (batch_size={BATCH_SIZE})")
    print("=" * 80)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_idx >= NUM_SAMPLES_TO_SHOW:
            break

        print(f"\n{'─' * 60}")
        print(f"Batch {batch_idx}: inputs.shape={inputs.shape}, targets.shape={targets.shape}")
        print(f"{'─' * 60}")

        for sample_idx in range(inputs.shape[0]):
            input_ids = inputs[sample_idx]
            target_ids = targets[sample_idx]

            # Decode input tokens
            # Remove padding for cleaner display
            non_pad_mask = input_ids != tokenizer.pad_token_id
            input_ids_clean = input_ids[non_pad_mask].tolist()

            # Decode target tokens (remove ignore_index and padding)
            valid_target_mask = (target_ids != -100) & (target_ids != tokenizer.pad_token_id)
            target_ids_clean = target_ids[valid_target_mask].tolist()

            decoded_input = tokenizer.decode(input_ids_clean)
            decoded_target = tokenizer.decode(target_ids_clean)

            print(f"\n  [Sample {sample_idx}]")
            print(f"  Input token count (non-pad): {len(input_ids_clean)}")
            print(f"  Target token count (valid): {len(target_ids_clean)}")
            print(f"\n  --- DECODED INPUT (first 500 chars) ---")
            print(f"  {decoded_input[:500]}")
            print(f"\n  --- DECODED TARGET / GROUND TRUTH (first 500 chars) ---")
            print(f"  {decoded_target[:500]}")
            print()

    print("\n" + "=" * 80)
    print("DONE - Dataset test complete.")
    print("=" * 80)
