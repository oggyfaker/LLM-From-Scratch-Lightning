import re
from dataclasses import dataclass, field


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Literal \boxed{...} with no nested braces. Rejects "\boxed{" (unclosed),
# "boxed{5}" (no backslash) and keeps "\boxed{}" so it can be named empty_box.
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")

# Chat turn-enders Qwen3 may emit; stripped before grading so a trailing
# <|im_end|> never lands inside the answer region.
END_TOKENS = ("<|im_end|>", "<|endoftext|>")

# correct dominates; format keeps a gradient alive in all-wrong groups, where a
# pure 0/1 reward would give every rollout the same score and zero advantage.
DEFAULT_WEIGHTS = {"correct": 1.0, "format": 0.2}


# ----------------
# ANSWER NORMALIZE
# ----------------
def normalize_answer(text):
    """GSM8K answers are plain numbers - strip formatting noise, not content."""
    if text is None:
        return ""
    return text.strip().replace(",", "").replace("$", "").rstrip(".").strip()


def to_number(text):
    """Numeric value of an answer string, or None if it is not a number."""
    try:
        return float(normalize_answer(text))
    except (TypeError, ValueError):
        return None


def strip_end_tokens(text):
    for token in END_TOKENS:
        text = text.replace(token, "")
    return text


# -----------------
# COMPLETION PARSER
# -----------------
@dataclass
class Parsed:
    """What a rollout actually produced, before any scoring."""
    thinking: str = ""
    answer_region: str = ""
    prediction: str = None
    value: float = None
    bad_reason: str = None
    n_think_close: int = 0
    n_boxed: int = 0

    @property
    def well_formed(self):
        """Exactly one </think>, then one usable numeric \\boxed{...}."""
        return self.bad_reason is None and self.value is not None


def parse_completion(completion, finish_reason="stop", thinking_mode=True):
    """Completion text -> Parsed, naming the first failure it hits.

    bad_reason vocabulary:
        empty_output         nothing was generated
        think_not_closed     thinking mode, </think> never emitted
        multiple_think_close more than one </think> (a format hack)
        truncated_length     hit max_new_tokens before producing an answer
        no_boxed_answer      finished cleanly but never wrote \\boxed{...}
        empty_box            wrote \\boxed{} with nothing inside
        non_numeric_answer   boxed something that is not a number
    """
    text = strip_end_tokens(completion or "").strip()
    if not text:
        return Parsed(bad_reason="empty_output")

    n_close = text.count(THINK_CLOSE)

    if thinking_mode:
        if n_close == 0:
            reason = "truncated_length" if finish_reason == "length" else "think_not_closed"
            return Parsed(thinking=text, bad_reason=reason, n_think_close=0)
        if n_close > 1:
            return Parsed(thinking=text, bad_reason="multiple_think_close",
                          n_think_close=n_close)
        thinking, _, answer_region = text.partition(THINK_CLOSE)
        thinking, answer_region = thinking.strip(), answer_region.strip()
    else:
        thinking, answer_region = "", text

    parsed = Parsed(thinking=thinking, answer_region=answer_region,
                    n_think_close=n_close)

    # Only text AFTER </think> may carry the answer; scanning the reasoning
    # would credit a lucky intermediate result.
    matches = BOXED_RE.findall(answer_region)
    parsed.n_boxed = len(matches)
    if not matches:
        parsed.bad_reason = ("truncated_length" if finish_reason == "length"
                             else "no_boxed_answer")
        return parsed

    # The LAST box wins: a model often restates the answer.
    parsed.prediction = matches[-1].strip()
    if not parsed.prediction:
        parsed.bad_reason = "empty_box"
        return parsed

    parsed.value = to_number(parsed.prediction)
    if parsed.value is None:
        parsed.bad_reason = "non_numeric_answer"
    return parsed


# ----------------
# REWARD COMPONENTS
# ----------------
def correct_reward(parsed, expected):
    """1.0 only when the boxed value equals the ground-truth number."""
    gold = to_number(expected)
    if parsed.value is None or gold is None:
        return 0.0
    return float(parsed.value == gold)


def format_reward(parsed):
    """1.0 when the rollout obeyed the thinking + boxed contract, right or not."""
    return float(parsed.well_formed)


REWARD_FNS = {
    "correct": correct_reward,
    "format": lambda parsed, expected: format_reward(parsed),
}


# ------------
# REWARD ENTRY
# ------------
@dataclass
class RewardOutput:
    total: float
    label: str                      # "Correct" | "Wrong" | "Bad"
    prediction: str = None
    bad_reason: str = None
    components: dict = field(default_factory=dict)

def gsm8k_reward(completion, expected, finish_reason="stop", thinking_mode=True,
                 weights=None, bad_penalty=0.0, truncation_penalty=0.0):
    """Score one rollout.

    bad_penalty / truncation_penalty are negative numbers when you want to push
    malformed or over-long rollouts below a merely wrong one. Both default to 0,
    which keeps the reward non-negative and the advantage purely comparative.
    """
    weights = DEFAULT_WEIGHTS if weights is None else weights
    parsed = parse_completion(completion, finish_reason, thinking_mode)

    components = {name: REWARD_FNS[name](parsed, expected) for name in weights}
    total = sum(weights[name] * value for name, value in components.items())

    if parsed.bad_reason == "truncated_length":
        total += truncation_penalty
    elif parsed.bad_reason is not None:
        total += bad_penalty

    if components.get("correct", 0.0) > 0:
        label = "Correct"
    elif parsed.bad_reason is None:
        label = "Wrong"
    else:
        label = "Bad"

    return RewardOutput(
        total=total, 
        label=label, 
        prediction=parsed.prediction,
        bad_reason=parsed.bad_reason, 
        components=components
    )


def reward_groups(completions, finish_reasons, answers, group_size,
                  thinking_mode=True, weights=None, bad_penalty=0.0,
                  truncation_penalty=0.0):
    """Score a whole rollout batch laid out as [p0 x G, p1 x G, ...]
    -> list[float] of N rewards, in the same order as the rollouts.

    answers holds ONE ground truth per group (N / G of them). Each group is
    graded against its own prompt's answer - a reward means nothing otherwise.
    """
    if len(completions) != len(answers) * group_size:
        raise ValueError(f"{len(completions)} rollouts != {len(answers)} groups "
                         f"x group_size {group_size}")
    rewards = []
    for g, expected in enumerate(answers):
        for i in range(g * group_size, (g + 1) * group_size):
            output = gsm8k_reward(
                completion=completions[i],
                expected=expected,
                finish_reason=finish_reasons[i],
                thinking_mode=thinking_mode,
                weights=weights,
                bad_penalty=bad_penalty,
                truncation_penalty=truncation_penalty,
            )
            rewards.append(output.total)
    return rewards


# ----------------
# MAIN - TEST
# ----------------
if __name__ == "__main__":

    # GSM8K train.json[0] - ground-truth answer is "72".
    QUESTION = ("Natalia sold clips to 48 of her friends in April, and then she "
                "sold half as many clips in May. How many clips did Natalia sell "
                "altogether in April and May?")
    ANSWER = "72"
    COT = ("Natalia sold 48/2 = 24 clips in May.\n"
           "Natalia sold 48+24 = 72 clips altogether in April and May.")

    CASES = [
        ("perfect",
         f"{COT}\n</think>\n\\boxed{{72}}<|im_end|>", "stop"),
        ("correct, decimal form",
         f"{COT}\n</think>\n\\boxed{{72.0}}<|im_end|>", "stop"),
        ("correct, dollar sign stripped",
         f"{COT}\n</think>\nThe answer is \\boxed{{$72}}<|im_end|>", "stop"),
        ("correct, restates the box twice (last wins)",
         f"{COT}\n</think>\nSo \\boxed{{24}} in May, total \\boxed{{72}}<|im_end|>", "stop"),
        ("wrong number, clean format",
         "Natalia sold 48/2 = 24 clips in May.\n</think>\n\\boxed{24}<|im_end|>", "stop"),
        ("right answer, no box at all",
         f"{COT}\nThe answer is 72.\n</think>\nShe sold 72 clips.<|im_end|>", "stop"),
        ("box is inside the reasoning only",
         f"{COT}\n\\boxed{{72}}\n</think>\nDone.<|im_end|>", "stop"),
        ("never closed the think block",
         f"{COT}\n\\boxed{{72}}", "stop"),
        ("ran out of token budget mid-reasoning",
         "Natalia sold 48/2 = 24 clips in May. Then she sold 48 +", "length"),
        ("closed think, then hit the budget",
         f"{COT}\n</think>\nThe final answer is", "length"),
        ("two </think> (format hack)",
         f"{COT}\n</think>\n</think>\n\\boxed{{72}}<|im_end|>", "stop"),
        ("empty box",
         f"{COT}\n</think>\n\\boxed{{}}<|im_end|>", "stop"),
        ("non-numeric box",
         f"{COT}\n</think>\n\\boxed{{72 clips}}<|im_end|>", "stop"),
        ("empty output",
         "", "stop"),
    ]

    print(f"Question: {QUESTION}")
    print(f"Ground truth: {ANSWER}")
    print(f"Weights: {DEFAULT_WEIGHTS}\n")

    header = f"{'case':<44} {'label':<8} {'pred':<10} {'corr':>5} {'fmt':>5} {'total':>6}  bad_reason"
    print(header)
    print("-" * len(header))

    completions, finish_reasons = [], []
    for name, completion, finish_reason in CASES:
        out = gsm8k_reward(completion, ANSWER, finish_reason, thinking_mode=True)
        completions.append(completion)
        finish_reasons.append(finish_reason)
        print(f"{name:<44} {out.label:<8} {str(out.prediction):<10} "
              f"{out.components['correct']:>5.1f} {out.components['format']:>5.1f} "
              f"{out.total:>6.2f}  {out.bad_reason or ''}")

    # Thousands separators are stripped too, shown against a matching answer.
    comma = gsm8k_reward(f"{COT}\n</think>\n\\boxed{{1,072}}<|im_end|>", "1072")
    print(f"\n{'comma separator vs answer 1072':<44} {comma.label:<8} "
          f"{str(comma.prediction):<10} total={comma.total:.2f}")

    group_rewards = reward_groups(completions, finish_reasons, answers=[ANSWER],
                                  group_size=len(CASES))
    print("\nreward_groups - all cases as ONE group:")
    print(f"  rewards {[round(r, 2) for r in group_rewards]}")
    assert group_rewards == [gsm8k_reward(c, ANSWER, fr).total for _, c, fr in CASES]

    # Batch layout [p0 x G, p1 x G]: the same 4 completions graded against two
    # different answers. Group 0 (answer 72) and group 1 (answer 24) disagree on
    # which rollouts are Correct, which is why each group needs its own answer.
    BATCH = [CASES[0], CASES[4], CASES[5], CASES[8]]      # 72, 24, no box, truncated
    G = len(BATCH)
    batch_completions = [c for _, c, _ in BATCH] * 2
    batch_finish = [fr for _, _, fr in BATCH] * 2
    rewards = reward_groups(
        batch_completions, batch_finish, answers=[ANSWER, "24"], group_size=G)
    print(f"\nreward_groups - 2 groups x G={G}, answers=['72', '24']:")
    print(f"  group 0 rewards {rewards[:G]}")
    print(f"  group 1 rewards {rewards[G:]}")
    assert rewards[:G] == [1.2, 0.2, 0.0, 0.0]
    assert rewards[G:] == [0.2, 1.2, 0.0, 0.0]

    # Penalties push Bad below Wrong when you want the policy to fix format first.
    print("\nSame table with bad_penalty=-0.5, truncation_penalty=-0.2:")
    for name, completion, finish_reason in CASES:
        out = gsm8k_reward(completion, ANSWER, finish_reason, thinking_mode=True,
                           bad_penalty=-0.5, truncation_penalty=-0.2)
        print(f"  {name:<44} {out.label:<8} {out.total:>6.2f}  {out.bad_reason or ''}")
