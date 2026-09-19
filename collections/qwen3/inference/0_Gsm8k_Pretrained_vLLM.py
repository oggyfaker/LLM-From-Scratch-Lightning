import os
import re
import sys
import json
import time
import shutil
from pathlib import Path
from dataclasses import dataclass

QWEN3_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = QWEN3_DIR.parents[1]
MODELS_DIR = QWEN3_DIR / "models"
INFER_DIR = QWEN3_DIR / "inference"

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(QWEN3_DIR))
_existing = os.environ.get("PYTHONPATH", "")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["PYTHONPATH"] = os.pathsep.join([str(QWEN3_DIR)] + ([_existing] if _existing else []))

from models.qwen3_vllm import Qwen3_14B_vLLM
from vllm import LLM, SamplingParams, ModelRegistry
from data.Gsm8k.data_utils import load_gsm8k_json, format_prompt_only
from models.qwen3_dense import QWEN_14B_CFG, Qwen3Model, from_pretrained

# ------
# CONFIG
# ------
@dataclass
class CFG:
    MODEL_CFG = QWEN_14B_CFG
    PRETRAINED_DIR = QWEN3_DIR / "checkpoint" / "Qwen3-14B"
    TOKENIZER_PATH = MODELS_DIR / "tokenizer.json"

    TEST_JSON = PROJECT_ROOT / "data" / "Gsm8k" / "test.json"
    OUT_DIR = INFER_DIR / "outputs"

    EVAL_LIMIT = None      # None: scores the full 1319-question test set
    TEMPERATURE = 0.0      # greedy, so the baseline is reproducible
    STOP_TOKEN_IDS = [151643, 151645]   # <|endoftext|>, <|im_end|>

    # Per-version budgets. A pretrained Qwen3 explains itself even when told to
    # answer directly, so v1 still needs room to reach a box; v2 needs ~2k to
    # close </think> before it may answer at all.
    MAX_TOKENS = {
        "v1_pretrained_direct": 768,
        "v2_pretrained_thinking": 2048
    }

    MAX_MODEL_LEN = 4096   # prompts are ~200 tok; the rest sizes the KV cache
    GPU_MEM_UTIL = 0.9


VERSIONS = [
    ("v1_pretrained_direct", False),
    ("v2_pretrained_thinking", True),
]


# --------------------
# CHECKPOINT AS HF DIR
# --------------------
def download_pretrained(weights_dir, model_cfg):
    """Fetch the shards through models/qwen3_dense.py if they are not on disk."""
    if list(weights_dir.glob("*.safetensors")):
        return weights_dir

    repo_id = f"Qwen/{weights_dir.name}"
    print(f"no weights in {weights_dir} -> downloading {repo_id}")

    # from_pretrained() writes to ./checkpoint/<repo name>, resolved against the
    # CURRENT directory, so call it from the one that lands on weights_dir.
    cwd = Path.cwd()
    os.chdir(weights_dir.parent.parent)
    try:
        from_pretrained(Qwen3Model(model_cfg), repo_id=repo_id)
    finally:
        os.chdir(cwd)
    return weights_dir


def write_config_json(weights_dir, model_cfg):
    """architectures MUST equal the name registered with ModelRegistry."""
    cfg = {
        "architectures": ["Qwen3vLLM"],
        "model_type": "qwen3",
        "vocab_size": model_cfg.vocab_size,
        "hidden_size": model_cfg.emb_dim,
        "intermediate_size": model_cfg.hidden_dim,
        "num_hidden_layers": model_cfg.n_blocks,
        "num_attention_heads": model_cfg.n_heads,
        "num_key_value_heads": model_cfg.n_kv_groups,
        "head_dim": model_cfg.head_dim,
        "max_position_embeddings": model_cfg.context_length,
        "rope_theta": model_cfg.rope_base,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
    (weights_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def prepare_pretrained_dir(weights_dir, tokenizer_path, model_cfg):
    download_pretrained(weights_dir, model_cfg)
    write_config_json(weights_dir, model_cfg)
    shutil.copyfile(tokenizer_path, weights_dir / "tokenizer.json")
    return weights_dir


# -------
# SCORING
# -------
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
THINK_CLOSE = "</think>"


def normalize_answer(text):
    return text.strip().replace(",", "").replace("$", "").rstrip(".").strip()


def to_number(text):
    try:
        return float(normalize_answer(text))
    except (TypeError, ValueError):
        return None


def grade(completion, expected, thinking_mode, finish_reason):
    """Grade one completion -> (matching, prediction, thinking_model, bad_reason).

    Correct : a well-formed \\boxed{...} whose value equals the ground truth
    Wrong   : a clean numeric answer that is not the right number
    Bad     : nothing usable could be extracted; bad_reason says which failure
              (empty_output | think_not_closed | truncated_length |
               no_boxed_answer | empty_box | non_numeric_answer)
    """
    text = completion or ""
    if thinking_mode:
        if THINK_CLOSE in text:
            thinking_model, _, answer_region = text.partition(THINK_CLOSE)
            thinking_model, answer_region = thinking_model.strip(), answer_region.strip()
        else:
            return ("Bad", None, text.strip(),
                    "empty_output" if not text.strip() else "think_not_closed")
    else:
        thinking_model, answer_region = "", text.strip()

    if not text.strip():
        return "Bad", None, thinking_model, "empty_output"

    # The LAST box wins: reasoning often boxes an intermediate result first.
    matches = BOXED_RE.findall(answer_region)
    if not matches:
        reason = "truncated_length" if finish_reason == "length" else "no_boxed_answer"
        return "Bad", None, thinking_model, reason

    prediction = matches[-1].strip()
    if not prediction:
        return "Bad", None, thinking_model, "empty_box"

    pred_num, gold_num = to_number(prediction), to_number(expected)
    if pred_num is None:
        return "Bad", prediction, thinking_model, "non_numeric_answer"

    matching = "Correct" if (gold_num is not None and pred_num == gold_num) else "Wrong"
    return matching, prediction, thinking_model, None


def build_record(rec, completion, finish_reason, thinking_mode):
    matching, prediction, thinking_model, bad_reason = grade(
        completion, rec["answer"], thinking_mode, finish_reason
    )
    return {
        "question": rec["question"],
        "thinking": rec["thinking"],
        "answer": rec["answer"],
        "num_gt_tokens": rec["num_gt_tokens"],
        "prediction": prediction,
        "thinking-model": thinking_model,
        "matching": matching,
        "bad_reason": bad_reason,
        "finish_reason": finish_reason,
    }


def summarize(version, rows):
    counts = {k: sum(r["matching"] == k for r in rows)
              for k in ("Correct", "Wrong", "Bad")}
    return {"version": version, "questions": len(rows), **counts,
            "accuracy": 100.0 * counts["Correct"] / max(1, len(rows))}


def print_report(summaries):
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"{'version':<26}{'accuracy':>12}{'Correct':>10}{'Wrong':>8}{'Bad':>7}")
    for s in summaries:
        print(f"{s['version']:<26}{s['accuracy']:>11.1f}%{s['Correct']:>10}"
              f"{s['Wrong']:>8}{s['Bad']:>7}")
    print(bar)


# ---------
# INFERENCE
# ---------
def load_engine(model_dir):
    return LLM(
        model=str(model_dir),
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=CFG.GPU_MEM_UTIL,
        max_model_len=CFG.MAX_MODEL_LEN,
    )


def inference(llm, version, thinking_mode, records):
    """Generate, grade and save one version. Returns its summary."""
    # Identical prompt construction to training — same template, same tags.
    prompts = [format_prompt_only(r["question"], include_thinking=thinking_mode)
               for r in records]
    params = SamplingParams(temperature=CFG.TEMPERATURE,
                            max_tokens=CFG.MAX_TOKENS[version],
                            stop_token_ids=CFG.STOP_TOKEN_IDS)

    started = time.time()
    outputs = llm.generate(prompts, params)      # vLLM preserves input order
    elapsed = (time.time() - started) / 60

    rows = [build_record(rec, out.outputs[0].text, out.outputs[0].finish_reason,
                         thinking_mode)
            for rec, out in zip(records, outputs)]
    (CFG.OUT_DIR / f"{version}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = summarize(version, rows)
    print(f"{version}: {summary['accuracy']:.1f}%  ({elapsed:.1f} min)")
    return summary


# ----
# MAIN
# ----
if __name__ == "__main__":
    CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_gsm8k_json(str(CFG.TEST_JSON))
    if CFG.EVAL_LIMIT is not None:
        records = records[:CFG.EVAL_LIMIT]

    prepare_pretrained_dir(CFG.PRETRAINED_DIR, CFG.TOKENIZER_PATH, CFG.MODEL_CFG)

    # Register vLLM
    ModelRegistry.register_model("Qwen3vLLM", Qwen3_14B_vLLM)
    llm = load_engine(CFG.PRETRAINED_DIR)

    # One engine for both versions: same weights, only the prompt differs, so
    # reloading 28 GB between them would be pure waste.
    summaries = [
        inference(llm, version, thinking_mode, records) for version, thinking_mode in VERSIONS
    ]
    print_report(summaries)
