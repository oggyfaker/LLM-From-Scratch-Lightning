import torch
import pandas as pd
from torch.utils.data import Dataset


# --------------------
# PROMPT CHAT TEMPLATE
# --------------------
def format_chat_prompt(body: str, rule: str, answer: str, base_prompt: str) -> str:
    """Format a single sample into Qwen3 chat template string.
    
    Format:
        <|im_start|> 
        system {base_prompt} 
        <|im_end|>
        
        <|im_start|> 
        user Comment: {body}
        rule: {rule}
        <|im_end|>

        <|im_start|>assistant
        {Yes/No}
        <|im_end|>
    """
    return (
        f"<|im_start|>system\n{base_prompt}<|im_end|>\n"
        f"<|im_start|>user\nComment: {body}\n\nrule: {rule}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )


# -------------------
# COMBINE DATA COLUMN
# -------------------
def get_dataframe_to_train(data_path: str, seed: int = 1001) -> pd.DataFrame:
    """Prepare augmented training dataframe.
    
    Strategy from Kaggle:
    1. Use train.csv labeled rows (body, rule, rule_violation)
    2. Extract positive/negative examples from test.csv as extra training rows
    3. Deduplicate
    4. Upsample test_examples rows (2x)
    5. Shuffle

    Args:
        data_path: Path to the Jigsaw2026 data folder.
        seed: Random seed for shuffling.

    Returns:
        DataFrame with columns: body, rule, rule_violation
    """
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    # Base train rows
    base = train_dataset[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)

    # Extract positive/negative examples from test.csv as extra labeled rows
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    # Combine & dedupe
    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # Upsample test_examples (add one extra copy → 2x total)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    # Shuffle
    dataframe = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return dataframe


# -----------------
# SPLIT TRAIN - VAL
# -----------------
def split_dataset(dataframe: pd.DataFrame, val_ratio=0.2, seed: int = 1001):
    """Split dataframe into train/val.

    Validation set is drawn only from original train.csv rows (source=='train').
    Test-example rows are added exclusively to the training set.
    
    Args:
        dataframe: Augmented DataFrame with 'source' column from get_dataframe_to_train.
        val_ratio: Fraction of original train rows used for validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_df, val_df)
    """
    # Separate original train rows from test-example rows
    train_only = dataframe[dataframe["source"] == "train"].copy()
    test_examples = dataframe[dataframe["source"] == "test_examples"].copy()

    # Stratified split on original train rows only
    val_frames, train_frames = [], []
    for label in train_only["rule_violation"].unique():
        subset = train_only[train_only["rule_violation"] == label]
        n_val = max(1, int(len(subset) * val_ratio))
        val_part = subset.sample(n=n_val, random_state=seed)
        train_part = subset.drop(val_part.index)
        val_frames.append(val_part)
        train_frames.append(train_part)

    # Combine 80% train rows + all test-example rows for training
    train_df = pd.concat(train_frames + [test_examples]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_frames).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Drop helper column
    train_df = train_df.drop(columns=["source"])
    val_df = val_df.drop(columns=["source"])

    print(f"Training set length: {len(train_df)} (train rows: {len(pd.concat(train_frames))}, test examples: {len(test_examples)})")
    print(f"Validation set length: {len(val_df)} (from train.csv only)")
    print(f"Train class distribution:\n{train_df['rule_violation'].value_counts().to_string()}")
    print(f"Val class distribution:\n{val_df['rule_violation'].value_counts().to_string()}")

    return train_df, val_df


# -----------------
# TORCH DATASET 
# -----------------
class JigsawDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        """Pre-tokenize all samples into token ID lists.
            - Refer: https://www.kaggle.com/code/wowfattie/1st-place-code?scriptVersionId=270106583
            - Also stores the prompt length (system + user tokens) so the collate
            function can mask prompt positions in targets → loss only on assistant
            response tokens.
        """
        self.data = dataframe
        self.encoded_texts = []
        self.prompt_lengths = []

        positive_answer = "Yes"
        negative_answer= "No"
        base_prompt = "Reddit moderation: Does the comment violate the rule? Answer 'Yes' or 'No' only."

        for _, row in dataframe.iterrows():
            answer = positive_answer if row["rule_violation"] == 1 else negative_answer
            prompt_text = (
                f"<|im_start|>system\n{base_prompt}<|im_end|>\n"
                f"<|im_start|>user\nComment: {row['body']}\n\nrule: {row['rule']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            ) # Encode prompt (system + user) separately to get boundary

            prompt_ids = tokenizer.encode(prompt_text)
            full_text = format_chat_prompt(
                row["body"], row["rule"], answer, base_prompt
            )
            full_ids = tokenizer.encode(full_text)
            
            self.encoded_texts.append(full_ids)
            self.prompt_lengths.append(len(prompt_ids))

    def __getitem__(self, index):
        return self.encoded_texts[index], self.prompt_lengths[index]

    def __len__(self):
        return len(self.data)


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
      - prompt positions (system + user tokens) → train on response only
      - padding positions (except first EOS)
    """
    # Each item is (token_ids, prompt_length)
    # Each sequence gets an EOS token appended, so +1
    max_length = max(len(item[0]) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item_ids, prompt_len in batch:
        # 0. Append EOS and pad to max_length
        seq = list(item_ids) + [pad_token_id]
        padded = seq + ([pad_token_id] * (max_length - len(seq)))

        # --- Part 1: Inputs ---
        inputs = torch.tensor(padded[:-1])

        # --- Part 2: Targets ---
        targets = torch.tensor(padded[1:])

        # Mask prompt tokens → loss only on assistant response (like train_on_responses_only)
        # In shifted targets, targets[i] = padded[i+1], so targets[prompt_len-1] is the
        # FIRST response token.  We must keep it; only mask 0..(prompt_len-2).
        if prompt_len > 1:
            targets[:prompt_len - 1] = ignore_index

        # Mask padding tokens (keep the FIRST pad = real EOS, mask the rest)
        pad_positions = torch.nonzero(targets == pad_token_id).squeeze()
        if pad_positions.numel() > 1:
            targets[pad_positions[1:]] = ignore_index

        # Truncate to max length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    inputs_lst = torch.stack(inputs_lst).to(device)
    targets_lst = torch.stack(targets_lst).to(device)
    return inputs_lst, targets_lst