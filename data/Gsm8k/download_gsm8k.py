"""
download_gsm8k.py
────────────────────────────────────────────────────────────────
Download openai/gsm8k (config "main") from HuggingFace and build
train.json / test.json for supervised fine-tuning.

Source: https://huggingface.co/datasets/openai/gsm8k
        7473 train rows / 1319 test rows, columns: question, answer

The raw `answer` column holds the chain of thought with calculator
annotations and a final answer line:

    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.
     Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
     #### 72"

Each output record splits that into:
    question      - problem statement, unusual line terminators normalized
    thinking      - chain of thought, <<...>> annotations stripped
    answer        - final answer after ####, commas removed
    num_gt_tokens - assistant ground-truth length (thinking + answer) in
                    Qwen3 tokens, matching NemotronReasoning2026 semantics

Requires: requests, pyarrow

Usage:
    python data/Gsm8k/download_gsm8k.py
"""

import io
import os
import re
import sys
import json
import importlib.util

import requests
import pyarrow.parquet as pq


PARQUET_URL = "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/{split}/0.parquet"
CALC_ANNOTATION = re.compile(r"<<[^>]*>>")

# Unicode line/paragraph separators used as line breaks upstream (train row 2381).
# Qwen3 splits U+2028 into two byte-fragment tokens, so fold them into "\n".
UNUSUAL_TERMINATORS = re.compile("[  ]")


def load_tokenizer(project_root: str):
    """Load Qwen3Tokenizer directly by path (avoids __init__.py pulling in vllm)."""
    models_dir = os.path.join(project_root, "collections", "qwen3", "models")
    spec = importlib.util.spec_from_file_location(
        "qwen_tokenizer", os.path.join(models_dir, "qwen_tokenizer.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Qwen3Tokenizer(
        tokenizer_file_path=os.path.join(models_dir, "tokenizer.json")
    )


def download_split(split: str) -> list:
    """Fetch one parquet split from the HuggingFace datasets server."""
    url = PARQUET_URL.format(split=split)
    print(f"Downloading {split}: {url}")
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    table = pq.read_table(io.BytesIO(response.content))
    print(f"  {table.num_rows} rows, columns: {table.column_names}")
    return table.to_pylist()


def normalize_text(text: str) -> str:
    """Fold unusual Unicode line terminators into plain newlines."""
    return UNUSUAL_TERMINATORS.sub("\n", text)


def parse_answer(raw_answer: str):
    """Split the raw GSM8K answer into (thinking, final_answer)."""
    chain, final = raw_answer.rsplit("####", 1)
    thinking = normalize_text(CALC_ANNOTATION.sub("", chain)).strip()
    answer = final.strip().replace(",", "")
    return thinking, answer


def build_records(rows: list, tokenizer, format_chat_prompt, format_prompt_only) -> list:
    """Convert raw parquet rows into training records with token counts."""
    records = []
    for row in rows:
        thinking, answer = parse_answer(row["answer"])
        question = normalize_text(row["question"]).strip()

        full_ids = tokenizer.encode(
            format_chat_prompt(question, answer, thinking=thinking)
        )
        prompt_ids = tokenizer.encode(
            format_prompt_only(question, include_thinking=True)
        )

        records.append({
            "question": question,
            "thinking": thinking,
            "answer": answer,
            "num_gt_tokens": len(full_ids) - len(prompt_ids),
        })
    return records


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(here))
    sys.path.insert(0, project_root)

    from data.Gsm8k.data_utils import format_chat_prompt, format_prompt_only

    tokenizer = load_tokenizer(project_root)

    for split, filename in (("train", "train.json"), ("test", "test.json")):
        rows = download_split(split)
        records = build_records(
            rows, tokenizer, format_chat_prompt, format_prompt_only
        )

        out_path = os.path.join(here, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        gt_tokens = [rec["num_gt_tokens"] for rec in records]
        print(f"  wrote {len(records)} records -> {out_path}")
        print(f"  num_gt_tokens: mean {sum(gt_tokens) / len(gt_tokens):.1f}, "
              f"max {max(gt_tokens)}")


if __name__ == "__main__":
    main()
