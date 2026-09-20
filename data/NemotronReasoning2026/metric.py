"""Nemotron grading and its periodic generative-evaluation callback.

Same three-label scheme and the same round machinery as data/Gsm8k/metric.py —
NemotronEvalCallback subclasses Gsm8kEvalCallback and only replaces what this
dataset actually scores differently. Everything generic (scheduling, length
sorted batching, OOM halving, saving, logging, the generate_fn / prompt_fn /
_set_mode seams) is inherited unchanged.

What IS different here:

    matching is STRING-first, not numeric. Nemotron answers are 8-bit strings
    ("00110100"), cipher phrases, Roman numerals and punctuation runs, so
    float() would score a model that dropped two leading zeros as Correct.
    Numeric is a fallback, and only where it cannot hide a formatting error.

    extract_boxed balances braces. GSM8K's regex stops at the first closing
    brace and returns "\\text{XLIV" for \\boxed{\\text{XLIV}}, which Nemotron
    models emit often enough to matter.

    rows and metrics carry a category, and a small sample is drawn per category
    rather than uniformly, so a rare family cannot vanish from a round.
"""

import random
import re

from data.Gsm8k.metric import ACCURACY_KEY, Gsm8kEvalCallback

from .data_utils import format_prompt_only

# ACCURACY_KEY is re-exported: the metric the checkpoint callbacks monitor is
# the same one for both datasets, so a Nemotron training script imports it from
# here rather than reaching into data.Gsm8k.
__all__ = ["ACCURACY_KEY", "NemotronEvalCallback", "answers_match", "build_record",
           "classify", "extract_boxed", "normalize_answer", "summarize", "to_number"]


# -----------------------------------
# ANSWER GRADING  (Correct/Wrong/Bad)
# -----------------------------------

THINK_CLOSE = "</think>"

BOXED_OPEN = "\\boxed{"

# A leading zero before another digit makes the string form significant, so
# numeric comparison is refused for it.
LEADING_ZERO_RE = re.compile(r"^[+-]?0\d")


def extract_boxed(text):
    """Return every \\boxed{...} payload in `text`, outermost braces balanced."""
    results = []
    idx = text.find(BOXED_OPEN)
    while idx != -1:
        start = idx + len(BOXED_OPEN)
        depth, pos = 1, start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:                      # a closing brace was actually found
            results.append(text[start:pos - 1])
        idx = text.find(BOXED_OPEN, start)
    return results


def normalize_answer(text):
    """Strip LaTeX wrappers, collapse whitespace, drop a trailing period.

    Case is preserved; answers_match decides whether case matters.
    """
    s = (text or "").strip()
    for wrapper in ("\\text", "\\mathrm", "\\mathbf"):
        if s.startswith(wrapper + "{") and s.endswith("}"):
            s = s[len(wrapper) + 1:-1].strip()
    if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    s = s.replace("\\", "").strip()
    s = " ".join(s.split())
    return s.rstrip(".").strip()


def to_number(text):
    """Return the numeric value of an answer string, or None if it isn't one."""
    try:
        return float(normalize_answer(text).replace(",", ""))
    except (TypeError, ValueError):
        return None


def answers_match(prediction, expected):
    """True when `prediction` is the same answer as `expected`.

      1. Case-insensitive string equality after normalization — decides
         bit_manipulation, cipher, numeral and cryptarithm.
      2. Numeric equality as a fallback, only when neither side has a
         significant leading zero, so "110100" vs "00110100" stays Wrong.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(expected)
    if pred_norm.casefold() == gold_norm.casefold():
        return True

    if LEADING_ZERO_RE.match(pred_norm) or LEADING_ZERO_RE.match(gold_norm):
        return False
    pred_num, gold_num = to_number(prediction), to_number(expected)
    return pred_num is not None and gold_num is not None and pred_num == gold_num


def classify(completion, expected, thinking_mode, finish_reason,
             think_close=THINK_CLOSE):
    """Grade one completion -> (matching, prediction, thinking_model, bad_reason).

    ``matching`` is one of three labels:
        Correct  a well-formed \\boxed{...} whose value equals the ground truth
        Wrong    a clean extracted answer, but not the right one
        Bad      no usable answer could be extracted at all

    ``bad_reason`` keeps a budget artefact from being read as a reasoning error:
        empty_output        model returned nothing
        think_not_closed    thinking mode, </think> never emitted
        truncated_length    hit max_new_tokens with no answer
        no_boxed_answer     finished cleanly but never wrote \\boxed{...}
        empty_box           wrote \\boxed{} with nothing inside
    """
    text = completion or ""

    # 1. Split the reasoning off. Only text AFTER </think> may carry the final
    #    answer — scanning the chain of thought would credit a lucky
    #    intermediate. Nemotron traces box the answer inside the reasoning too,
    #    so without this split almost everything would grade Correct.
    if thinking_mode:
        if think_close in text:
            thinking_model, _, answer_region = text.partition(think_close)
            thinking_model, answer_region = thinking_model.strip(), answer_region.strip()
        else:
            return ("Bad", None, text.strip(),
                    "empty_output" if not text.strip()
                    else ("truncated_length" if finish_reason == "length"
                          else "think_not_closed"))
    else:
        thinking_model, answer_region = "", text.strip()

    if not text.strip():
        return "Bad", None, thinking_model, "empty_output"

    # 2. The LAST box wins: reasoning often boxes an intermediate result first.
    matches = extract_boxed(answer_region)
    if not matches:
        reason = "truncated_length" if finish_reason == "length" else "no_boxed_answer"
        return "Bad", None, thinking_model, reason

    prediction = matches[-1].strip()
    if not prediction:
        return "Bad", None, thinking_model, "empty_box"

    matching = "Correct" if answers_match(prediction, expected) else "Wrong"
    return matching, prediction, thinking_model, None


def build_record(rec, completion, finish_reason, thinking_mode,
                 think_close=THINK_CLOSE):
    """One graded row, in the schema the GSM8K run writes plus the category."""
    matching, prediction, thinking_model, bad_reason = classify(
        completion, rec["answer"], thinking_mode, finish_reason, think_close
    )
    return {
        "problem_id": rec["problem_id"],
        "category": rec["category"],
        "question": rec["question"],
        "thinking": rec["thinking"],
        "answer": rec["answer"],
        "num_gt_tokens": rec["num_gt_tokens"],
        "prediction": prediction,
        "thinking-model": thinking_model,
        "matching": matching,
        # --- diagnostics ---
        "bad_reason": bad_reason,
        "finish_reason": finish_reason,
    }


def summarize(version, rows, elapsed=None):
    """Counts + boxed-answer accuracy for one round, broken down per category."""
    n = len(rows)
    counts = {k: sum(r["matching"] == k for r in rows)
              for k in ("Correct", "Wrong", "Bad")}
    bad_reasons = {}
    for r in rows:
        if r["bad_reason"]:
            bad_reasons[r["bad_reason"]] = bad_reasons.get(r["bad_reason"], 0) + 1

    per_category = {}
    for r in rows:
        bucket = per_category.setdefault(r["category"], {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += r["matching"] == "Correct"
    for bucket in per_category.values():
        bucket["accuracy"] = 100.0 * bucket["correct"] / max(1, bucket["n"])

    return {"version": version, "questions": n, **counts,
            "accuracy": 100.0 * counts["Correct"] / max(1, n),
            "bad_reasons": bad_reasons, "per_category": per_category,
            "elapsed_sec": elapsed}


# --------------------------------------------
# PERIODIC GENERATIVE EVALUATION (TEST SET)
# --------------------------------------------

class NemotronEvalCallback(Gsm8kEvalCallback):
    """Gsm8kEvalCallback with Nemotron's grading, sampling and reporting.

    Only the dataset-specific seams are replaced; scheduling, batching, OOM
    handling and metric publication are inherited. ``prompt_fn`` defaults to
    this dataset's template rather than GSM8K's, whose system turn would move
    every sample off distribution.
    """

    def __init__(self, *args, prompt_fn=None, think_close=THINK_CLOSE, **kwargs):
        super().__init__(*args, prompt_fn=prompt_fn or format_prompt_only,
                         think_close=think_close, **kwargs)

    # ------------------------------------------------- dataset-specific seams
    @staticmethod
    def _select_records(records, num_samples, seed):
        """Stratified sample of the test set, or all of it when num_samples is None.

        A uniform 32 of 404 would miss equation_numeric_guess (6 rows) entirely,
        so rounds would move with the draw rather than the model. Sampling within
        each category, at least one row each, keeps rounds comparable; the sample
        is fixed for the whole run.
        """
        records = list(records)
        if num_samples is None or num_samples >= len(records):
            return records

        by_category = {}
        for rec in records:
            by_category.setdefault(rec["category"], []).append(rec)

        rng = random.Random(seed)
        quota = num_samples / len(records)
        chosen = []
        for category in sorted(by_category):
            bucket = by_category[category]
            rng.shuffle(bucket)
            chosen.extend(bucket[:max(1, round(len(bucket) * quota))])

        # Rounding per category can overshoot or undershoot the target.
        rng.shuffle(chosen)
        if len(chosen) > num_samples:
            return chosen[:num_samples]
        if len(chosen) < num_samples:
            picked = {id(r) for r in chosen}
            spare = [r for r in records if id(r) not in picked]
            rng.shuffle(spare)
            chosen.extend(spare[:num_samples - len(chosen)])
        return chosen

    def _build_record(self, rec, completion, finish_reason):
        return build_record(rec, completion, finish_reason,
                            self.thinking_mode, self.think_close)

    def _summarize(self, version, rows, elapsed):
        return summarize(version, rows, elapsed)

    def _metrics_from_summary(self, summary):
        """The GSM8K scalars plus one accuracy per task family."""
        metrics = super()._metrics_from_summary(summary)
        for category, bucket in summary['per_category'].items():
            metrics[f'Validation/acc_{category}'] = float(bucket['accuracy'])
        return metrics

    # -------------------------------------------------------------- output
    def _report(self, summary):
        super()._report(summary)
        for category, bucket in sorted(summary["per_category"].items()):
            print(f"  [eval]   {category:<26} {bucket['correct']}/{bucket['n']} "
                  f"= {bucket['accuracy']:.0f}%", flush=True)
