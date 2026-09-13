"""
merge_clean_datasets.py
────────────────────────────────────────────────────────────────────────────────
Merge nemotron_decoded_clean.csv (primary) and nemotron_decoded_clean_2.csv
(secondary) into nemotron_decoded_clean_merged.csv.

Rules
─────
1. Both files must have identical columns — abort if not.
2. All rows from both files are kept (no drops).
3. problem_id values from nemotron_decoded_clean.csv are kept as-is.
4. For every row in nemotron_decoded_clean_2.csv whose problem_id collides with
   any already-seen ID, the ID is renamed to  <original_id>_<N>  where N is a
   global counter that increments monotonically (1, 2, 3, …) across all renames.
5. Final row count must equal len(df1) + len(df2).

Usage
─────
    conda activate LLM
    python data/NemotronReasoning2026/merge_clean_datasets.py
"""

import os
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_PRIMARY   = os.path.join(_DATA_DIR, "nemotron_decoded_clean.csv")
PATH_SECONDARY = os.path.join(_DATA_DIR, "nemotron_decoded_clean_2.csv")
PATH_OUTPUT    = os.path.join(_DATA_DIR, "nemotron_decoded_clean_merged.csv")


def main() -> None:
    # ── 1. Load both files ────────────────────────────────────────────────────
    print(f"Loading primary   : {PATH_PRIMARY}")
    df1 = pd.read_csv(PATH_PRIMARY)
    print(f"  Rows: {len(df1)}  Columns: {df1.columns.tolist()}")

    print(f"Loading secondary : {PATH_SECONDARY}")
    df2 = pd.read_csv(PATH_SECONDARY)
    print(f"  Rows: {len(df2)}  Columns: {df2.columns.tolist()}")

    expected_total = len(df1) + len(df2)

    # ── 2. Column compatibility check ─────────────────────────────────────────
    cols1 = df1.columns.tolist()
    cols2 = df2.columns.tolist()
    if cols1 != cols2:
        print("\nERROR: Column mismatch — merge aborted.")
        print(f"  Primary   columns : {cols1}")
        print(f"  Secondary columns : {cols2}")
        return
    print(f"\nColumns match: {cols1}")

    # ── 3. Resolve problem_id conflicts in secondary ──────────────────────────
    seen_ids: set = set(df1["problem_id"].astype(str))
    rename_counter: int = 0
    new_ids: list = []

    for pid in df2["problem_id"].astype(str):
        if pid not in seen_ids:
            new_ids.append(pid)
            seen_ids.add(pid)
        else:
            rename_counter += 1
            new_pid = f"{pid}_{rename_counter}"
            # Guard: extremely unlikely, but ensure the renamed ID is also unique
            while new_pid in seen_ids:
                rename_counter += 1
                new_pid = f"{pid}_{rename_counter}"
            new_ids.append(new_pid)
            seen_ids.add(new_pid)

    df2 = df2.copy()
    df2["problem_id"] = new_ids

    print(f"\nDuplicate IDs renamed in secondary : {rename_counter}")
    print(f"  (each got postfix _<N> where N continues from 1 upward)")

    # ── 4. Merge ──────────────────────────────────────────────────────────────
    merged = pd.concat([df1, df2], ignore_index=True)

    # ── 5. Integrity checks ───────────────────────────────────────────────────
    actual_total = len(merged)
    dup_ids = merged["problem_id"].duplicated().sum()

    print(f"\nIntegrity checks:")
    print(f"  Expected rows : {expected_total}")
    print(f"  Actual rows   : {actual_total}  {'OK' if actual_total == expected_total else 'MISMATCH!'}")
    print(f"  Duplicate IDs : {dup_ids}  {'OK' if dup_ids == 0 else 'CONFLICT!'}")

    if actual_total != expected_total or dup_ids != 0:
        print("\nERROR: Integrity check failed — output NOT written.")
        return

    # ── 6. Write output ───────────────────────────────────────────────────────
    merged.to_csv(PATH_OUTPUT, index=False)
    print(f"\nSaved : {PATH_OUTPUT}")
    print(f"  Total rows : {actual_total}")
    print(f"  Columns    : {merged.columns.tolist()}")


if __name__ == "__main__":
    main()
