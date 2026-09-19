"""GSM8K SFT — QLoRA adapter folded into the pretrained Qwen3-14B at load time.

The adapter stays an adapter on disk. `Qwen3_14B_SFT_vLLM.load_weights()` runs
vLLM's normal load, then rebuilds every fine-tuned projection in place, so
nothing new is written: no merged .pth, no second model directory.

The base for those projections is the checkpoint's own 4-bit tensors, NOT the
bf16 shards. QLoRA trained the adapter against dequant(NF4(W)), and the
quantization error is about as large as the LoRA delta itself, so folding the
delta into pristine bf16 weights yields a measurably different (worse) model.
Rebuilt this way the result is bit-identical to merge_lora_state_dict().
"""

import os
import re
import sys
import json
import time
import shutil
from pathlib import Path
from dataclasses import dataclass

import torch
from bitsandbytes.functional import QuantState, dequantize_4bit

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams, ModelRegistry

QWEN3_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = QWEN3_DIR.parents[1]
MODELS_DIR = QWEN3_DIR / "models"
INFER_DIR = QWEN3_DIR / "inference"

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(QWEN3_DIR))
_existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = os.pathsep.join([str(QWEN3_DIR)] + ([_existing] if _existing else []))

from data.Gsm8k.data_utils import load_gsm8k_json, format_prompt_only
from models.qwen3_dense import QWEN_14B_CFG, Qwen3Model, from_pretrained
from models.qwen3_vllm import Qwen3_14B_vLLM


# ------
# CONFIG
# ------
@dataclass
class CFG:
    MODEL_CFG = QWEN_14B_CFG
    PRETRAINED_DIR = QWEN3_DIR / "checkpoint" / "Qwen3-14B"
    TOKENIZER_PATH = MODELS_DIR / "tokenizer.json"

    # Lightning .ckpt from 1_Qwen3_Gsm8k_SFT.py — supplies both the LoRA
    # pairs and the 4-bit base they were trained against.
    ADAPTER_CKPT = Path(
        "/home/simpsonadmin/workspace/AI_X/LLM-From-Scratch-Lightning/collections/qwen3"
        "/logs/Qwen3_Gsm8k_14B_QLoRA_4bit_think/15_09_26/version_0"
        "/checkpoints/00-0.4344-85.37.ckpt")
    LORA_ALPHA = 32        # rank is read from the adapter; scaling = ALPHA / rank

    TEST_JSON = PROJECT_ROOT / "data" / "Gsm8k" / "test.json"
    OUT_DIR = INFER_DIR / "outputs"
    VERSION = "v3_finetuned_thinking"
    INCLUDE_THINKING = True

    EVAL_LIMIT = None      # None: scores the full 1319-question test set
    TEMPERATURE = 0.0      # greedy, so the run is reproducible
    STOP_TOKEN_IDS = [151643, 151645]   # <|endoftext|>, <|im_end|>

    # Same budget the fine-tune was trained and evaluated under
    # (1_Qwen3_Gsm8k_SFT.py: MAX_SEQ_LEN, INFER_MAX_NEW_TOKENS). Samples
    # longer than MAX_SEQ_LEN were dropped from training, so generating past it
    # is out of distribution and wasted compute.
    MAX_SEQ_LEN = 512
    MAX_NEW_TOKENS = MAX_SEQ_LEN - 128
    GPU_MEM_UTIL = 0.9


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


# --------------
# ADAPTER + BASE
# --------------
def fused_targets(model_cfg):
    """Leaf -> (fused vLLM parameter, row offset).

    QKVParallelLinear packs q|k|v into one qkv_proj and MergedColumnParallelLinear
    packs gate|up into one gate_up_proj, so a delta lands on a ROW SLICE of the
    fused tensor rather than on a whole parameter.
    """
    q = model_cfg.n_heads * model_cfg.head_dim
    kv = model_cfg.n_kv_groups * model_cfg.head_dim
    return {
        "self_attn.q_proj": ("self_attn.qkv_proj", 0),
        "self_attn.k_proj": ("self_attn.qkv_proj", q),
        "self_attn.v_proj": ("self_attn.qkv_proj", q + kv),
        "self_attn.o_proj": ("self_attn.o_proj", 0),
        "mlp.gate_proj":    ("mlp.gate_up_proj", 0),
        "mlp.up_proj":      ("mlp.gate_up_proj", model_cfg.hidden_dim),
        "mlp.down_proj":    ("mlp.down_proj", 0),
    }


def load_sft_modules(ckpt_path, model_cfg):
    """One entry per LoRA'd projection: (param, offset, qweight, quant_state, A, B).

    mmap keeps the checkpoint on disk: the 4-bit tensors stay unread until
    load_weights() pulls each one to the GPU, so this costs ~0.1 s here.
    """
    state = torch.load(ckpt_path, map_location="cpu", mmap=True,
                       weights_only=False)["state_dict"]
    targets = fused_targets(model_cfg)

    modules = []
    for key in state:
        if not key.endswith(".lora.A"):
            continue
        # Drop Lightning's "model." wrapper before the repo -> vLLM translation.
        repo_name = key[len("model."):].removesuffix(".lora.A")
        parts = Qwen3_14B_vLLM._translate_custom_name(f"{repo_name}.weight").split(".")
        prefix, leaf = ".".join(parts[:3]), ".".join(parts[3:-1])
        fused_leaf, offset = targets[leaf]

        qkey = f"model.{repo_name}.linear.weight"
        quant_state = {k[len(qkey) + 1:]: v
                       for k, v in state.items() if k.startswith(qkey + ".")}

        modules.append((f"{prefix}.{fused_leaf}.weight", offset,
                        state[qkey], quant_state,
                        state[key].clone(),
                        state[key.removesuffix(".A") + ".B"].clone()))

    if not modules:
        raise ValueError(f"no lora.A/lora.B tensors in {ckpt_path}")
    return modules, modules[0][4].shape[1]


class Qwen3_14B_SFT_vLLM(Qwen3_14B_vLLM):
    """Qwen3-14B rebuilt as the QLoRA fine-tune, while the weights land on GPU.

    vLLM's own load supplies the parts training never touched (embeddings, norms,
    lm_head — verified bit-identical to the checkpoint). Every fine-tuned
    projection is then replaced by dequant(NF4) + LoRA, which is what the forward
    pass computed during training. Set MODULES/SCALING before the engine is built.
    """

    MODULES = []
    SCALING = 1.0

    @torch.no_grad()
    def load_weights(self, weights):
        loaded = super().load_weights(weights)

        params = dict(self.named_parameters())
        for name, offset, qweight, quant_state, A, B in self.MODULES:
            param = params[name]
            device = param.device
            state = QuantState.from_dict(
                {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in quant_state.items()}, device=device)

            # W = dequant(NF4(W)) + (alpha/rank) * (A @ B).T — bit-identical to
            # merge_lora_state_dict() in utils/checkpoint_utils.py.
            merged = dequantize_4bit(qweight.to(device), state).float()
            merged += (self.SCALING * (A.to(device, torch.float32)
                                       @ B.to(device, torch.float32))).T
            param.data[offset:offset + merged.shape[0]] = merged.to(param.dtype)
        return loaded


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


def print_report(s):
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"{'version':<26}{'accuracy':>12}{'Correct':>10}{'Wrong':>8}{'Bad':>7}")
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
        max_model_len=CFG.MAX_SEQ_LEN,
    )


def inference(llm, records):
    """Generate, grade and save the SFT run. Returns its summary."""
    # Identical prompt construction to training — same template, same tags.
    prompts = [format_prompt_only(r["question"], include_thinking=CFG.INCLUDE_THINKING)
               for r in records]

    # Per-prompt budget, exactly as training clamped it:
    # max_new = min(MAX_NEW_TOKENS, MAX_SEQ_LEN - prompt_len).
    tokenizer = llm.get_tokenizer()
    params = [SamplingParams(
        temperature=CFG.TEMPERATURE,
        max_tokens=max(1, min(CFG.MAX_NEW_TOKENS,
                              CFG.MAX_SEQ_LEN - len(tokenizer(p).input_ids))),
        stop_token_ids=CFG.STOP_TOKEN_IDS) for p in prompts]

    started = time.time()
    outputs = llm.generate(prompts, params)      # vLLM preserves input order
    elapsed = (time.time() - started) / 60

    rows = [build_record(rec, out.outputs[0].text, out.outputs[0].finish_reason,
                         CFG.INCLUDE_THINKING)
            for rec, out in zip(records, outputs)]
    (CFG.OUT_DIR / f"{CFG.VERSION}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = summarize(CFG.VERSION, rows)
    print(f"{CFG.VERSION}: {summary['accuracy']:.1f}%  ({elapsed:.1f} min)")
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

    modules, rank = load_sft_modules(CFG.ADAPTER_CKPT, CFG.MODEL_CFG)
    Qwen3_14B_SFT_vLLM.MODULES = modules
    Qwen3_14B_SFT_vLLM.SCALING = CFG.LORA_ALPHA / rank

    # Same architecture name as config.json, but the SFT subclass, so the merge
    # happens inside vLLM's own weight-loading pass.
    ModelRegistry.register_model("Qwen3vLLM", Qwen3_14B_SFT_vLLM)
    llm = load_engine(CFG.PRETRAINED_DIR)

    print_report(inference(llm, records))
