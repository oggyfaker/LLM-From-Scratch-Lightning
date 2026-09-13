"""
clean_csv.py - Strip chat template tokens from nemotron_decoded.csv
and save a clean version (nemotron_decoded_clean.csv).

Clean CSV columns:
    - problem_id
    - category: question type classification (e.g. bit_manipulation, cipher, gravity...)
    - user_content: raw user question (no chat template)
    - thinking: reasoning chain (no <think>/</think> tags)
    - final_answer: just the answer (extracted from \\boxed{})
    - num_tokens, num_prompt_tokens, num_gt_tokens (preserved metadata)

Categories (from TongKuiHang's nemotron-master):
    - bit_manipulation: 8-bit binary transformation (XOR, AND, OR, NOT, shifts, rotations)
    - cipher: substitution cipher (letter-to-letter mapping on words)
    - gravity: physics d = k*t^2 (find constant k from examples, compute distance)
    - unit_conversion: linear factor conversion (output = factor * input)
    - numeral: Arabic to Roman numeral conversion
    - equation_numeric_deduce: find arithmetic operation from number pairs (deduce rule)
    - equation_numeric_guess: find arithmetic operation from number pairs (guess)
    - cryptarithm_deduce: symbolic equation with letter substitution (deduce)
    - cryptarithm_guess: symbolic equation with letter substitution (guess)

Usage:
    python clean_csv.py
"""
import csv
import json
import os
import re


def clean_prompt(raw_prompt: str) -> str:
    """Extract user content from the raw prompt field.

    Raw format:
        <|im_start|>system\\n<|im_end|>\\n<|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n

    Returns:
        The user question text only.
    """
    user_start_tag = "<|im_start|>user\n"
    user_end_tag = "<|im_end|>"

    user_start = raw_prompt.find(user_start_tag)
    if user_start == -1:
        raise ValueError(f"Could not find user start tag in prompt: {raw_prompt[:100]}")

    content_start = user_start + len(user_start_tag)
    user_end = raw_prompt.find(user_end_tag, content_start)
    if user_end == -1:
        raise ValueError(f"Could not find user end tag in prompt: {raw_prompt[:100]}")

    return raw_prompt[content_start:user_end]


def clean_ground_truth(raw_gt: str) -> tuple:
    """Extract thinking and final_answer from the raw ground_truth field.

    Raw format:
        {thinking}\\n</think>\\n{final_answer}<|im_end|>

    Returns:
        (thinking, final_answer) tuple
    """
    im_end_tag = "<|im_end|>"
    think_end_tag = "\n</think>\n"

    # Remove trailing <|im_end|>
    text = raw_gt
    if text.endswith(im_end_tag):
        text = text[:-len(im_end_tag)]

    # Split on the LAST occurrence of \n</think>\n
    last_split = text.rfind(think_end_tag)
    if last_split != -1:
        thinking = text[:last_split]
        final_answer = text[last_split + len(think_end_tag):]
    else:
        # No </think> found - treat all as thinking
        thinking = text
        final_answer = ""

    # Extract content inside \boxed{...} — just the answer without wrapper
    final_answer = _extract_boxed_answer(final_answer)

    # Strip the answer placement tail from thinking:
    #   "\nI will now return the answer in \boxed{}\nThe answer in \boxed is\n\boxed{answer}"
    # This tail is added back by format_chat_prompt in data_utils.py
    thinking = _strip_answer_tail(thinking, final_answer)

    return thinking, final_answer


def _extract_boxed_answer(text: str) -> str:
    """Extract the answer from \\boxed{answer} format.

    Examples:
        '\\boxed{10010111}'       -> '10010111'
        '\\boxed{cat imagines book}' -> 'cat imagines book'
    """
    match = re.search(r"\\boxed\{(.+)\}", text)
    if match:
        return match.group(1)
    return text.strip()


def _strip_answer_tail(thinking: str, answer: str) -> str:
    """Remove the answer placement tail from the end of thinking.

    The tail pattern (consistent across ALL categories):
        \nI will now return the answer in \\boxed{}
        \nThe answer in \\boxed is
        \n\\boxed{<answer>}

    This is added back programmatically by format_chat_prompt in data_utils.py.
    """
    # Build the exact tail string to remove
    tail = (
        f"\nI will now return the answer in \\boxed{{}}\n"
        f"The answer in \\boxed is\n"
        f"\\boxed{{{answer}}}"
    )
    if thinking.endswith(tail):
        return thinking[:-len(tail)]

    # Fallback: strip just the last \boxed{answer} line
    boxed_str = f"\n\\boxed{{{answer}}}"
    if thinking.endswith(boxed_str):
        return thinking[:-len(boxed_str)]

    return thinking


# -------------------
# CLASSIFIER
# -------------------
def load_problem_categories(problems_dir: str) -> dict:
    """Load problem_id -> category mapping from nemotron-master/problems/*.jsonl.

    Each .jsonl file has a first line with {"id": ..., "category": ...}.
    """
    pid_to_cat = {}
    if not os.path.isdir(problems_dir):
        return pid_to_cat

    for fname in os.listdir(problems_dir):
        if not fname.endswith(".jsonl"):
            continue
        pid = fname.replace(".jsonl", "")
        fpath = os.path.join(problems_dir, fname)
        with open(fpath) as f:
            line = f.readline().strip()
            if line:
                rec = json.loads(line)
                pid_to_cat[pid] = rec.get("category", "unknown")
    return pid_to_cat


def classify_by_content(user_content: str) -> str:
    """Fallback classifier: determine category from the user_content text patterns.

    Used when the problem_id is not found in nemotron-master/problems/.
    """
    text = user_content.lower()

    # Bit manipulation: 8-bit binary patterns
    if "bit manipulation" in text or re.search(r"\b[01]{8}\b.*->\s*[01]{8}", user_content):
        return "bit_manipulation"

    # Cipher: substitution cipher patterns (encrypted words -> plain words)
    if "encryption rules" in text or "decrypt" in text:
        return "cipher"

    # Gravity: d = 0.5*g*t^2 or falling distance
    if "gravitational" in text or "falling distance" in text or "d = 0.5*g*t^2" in user_content:
        return "gravity"

    # Unit conversion: unit/measurement conversion with decimal numbers
    if "unit" in text and "convert" in text:
        return "unit_conversion"
    if "secretly converted" in text and "measurement" in text:
        return "unit_conversion"
    if re.search(r"[\d.]+ -> [\d.]+", user_content) and "numeral" not in text:
        # Decimal number patterns (not binary, not numeral)
        if not re.search(r"\b[01]{8}\b", user_content):
            # Check if it's gravity (has time/distance keywords)
            if "distance" in text or "t =" in user_content:
                return "gravity"
            return "unit_conversion"

    # Numeral: number system conversion (Arabic to Roman, etc.)
    if "numeral" in text or "number" in text and "system" in text:
        return "numeral"

    # Equation numeric: arithmetic operations on number pairs
    if re.search(r"\d+\s*[+\-*/^#@&!]\s*\d+\s*=\s*\d+", user_content):
        return "equation_numeric_deduce"

    # Cryptarithm: symbolic/letter equations
    if "cryptarithm" in text or re.search(r"[A-Z]{2,}\s*[+\-*/]\s*[A-Z]{2,}\s*=", user_content):
        return "cryptarithm_deduce"

    return "unknown"


def classify_problem(problem_id: str, user_content: str, pid_to_cat: dict) -> str:
    """Classify a problem by looking up the category from the problems directory,
    falling back to content-based classification.
    """
    # Strip augmentation suffixes like -p0, -p1
    base_pid = problem_id.split("-")[0] if "-" in problem_id else problem_id

    cat = pid_to_cat.get(base_pid)
    if cat:
        return cat

    # Fallback: classify from content
    return classify_by_content(user_content)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "nemotron_decoded.csv")
    output_path = os.path.join(script_dir, "nemotron_decoded_clean.csv")

    # Try to load problem categories from nemotron-master
    # Adjust this path if the repo is in a different location
    problems_dir = os.path.join(
        os.path.dirname(script_dir), "..", "..", "LLM", "data", "nemotron-master", "problems"
    )
    problems_dir = os.path.normpath(problems_dir)
    pid_to_cat = load_problem_categories(problems_dir)
    if pid_to_cat:
        print(f"Loaded {len(pid_to_cat)} problem categories from: {problems_dir}")
    else:
        print(f"WARNING: No problem categories found at {problems_dir}")
        print("  Will use content-based classification as fallback.")

    print(f"Reading: {input_path}")

    rows_out = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            user_content = clean_prompt(row["prompt"])
            thinking, final_answer = clean_ground_truth(row["ground_truth"])
            category = classify_problem(row["problem_id"], user_content, pid_to_cat)

            rows_out.append({
                "problem_id": row["problem_id"],
                "category": category,
                "user_content": user_content,
                "thinking": thinking,
                "final_answer": final_answer,
                "num_tokens": row["num_tokens"],
                "num_prompt_tokens": row["num_prompt_tokens"],
                "num_gt_tokens": row["num_gt_tokens"],
            })

    print(f"Total records cleaned: {len(rows_out)}")

    # Print category distribution
    from collections import Counter
    cat_counts = Counter(r["category"] for r in rows_out)
    print("\nCategory distribution:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")

    # Write clean CSV
    fieldnames = ["problem_id", "category", "user_content", "thinking", "final_answer",
                  "num_tokens", "num_prompt_tokens", "num_gt_tokens"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nSaved: {output_path}")

    # Verify reconstruction matches original
    print("\nVerifying reconstruction matches original...")
    with open(input_path, "r", encoding="utf-8") as f_orig:
        orig_reader = csv.DictReader(f_orig)
        mismatches = 0
        for i, (orig_row, clean_row) in enumerate(zip(orig_reader, rows_out)):
            # Reconstruct prompt
            reconstructed_prompt = (
                f"<|im_start|>system\n<|im_end|>\n"
                f"<|im_start|>user\n{clean_row['user_content']}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n"
            )
            # Reconstruct ground_truth (add answer tail + wrap answer in \boxed{})
            reconstructed_gt = (
                f"{clean_row['thinking']}\n"
                f"I will now return the answer in \\boxed{{}}\n"
                f"The answer in \\boxed is\n"
                f"\\boxed{{{clean_row['final_answer']}}}\n</think>\n"
                f"\\boxed{{{clean_row['final_answer']}}}<|im_end|>"
            )

            if reconstructed_prompt != orig_row["prompt"]:
                print(f"  MISMATCH prompt at row {i}!")
                print(f"    Original[-60:]:      {repr(orig_row['prompt'][-60:])}")
                print(f"    Reconstructed[-60:]: {repr(reconstructed_prompt[-60:])}")
                mismatches += 1
                if mismatches > 5:
                    break

            if reconstructed_gt != orig_row["ground_truth"]:
                print(f"  MISMATCH ground_truth at row {i}!")
                print(f"    Original[-80:]:      {repr(orig_row['ground_truth'][-80:])}")
                print(f"    Reconstructed[-80:]: {repr(reconstructed_gt[-80:])}")
                mismatches += 1
                if mismatches > 5:
                    break

        if mismatches == 0:
            print("  ALL rows reconstruct perfectly!")
        else:
            print(f"  {mismatches} mismatches found.")


if __name__ == "__main__":
    main()
