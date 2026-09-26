"""Export a merged from-scratch Qwen3 .pth as a HuggingFace model folder.

LoRAMergeCheckpoint saves a fine-tuned policy as one state_dict in qwen3_dense.py
naming (tok_emb.weight, transformer_blocks.{l}.att.W_query.weight, ...).
Unsloth, vLLM and transformers only read the HuggingFace layout, so this renames
every tensor with the inverse of qwen3_dense.from_pretrained() and writes a
folder any of them can load:

    config.json, generation_config.json     base model config + stop tokens
    tokenizer.json, tokenizer_config.json   copied from the base model repo
    model-0000N-of-0000M.safetensors        bf16 weights + index

The merged linears are fp32 on disk: dequant(NF4(W)) + LoRA delta, the exact
policy the SFT eval scored. They are written as bf16, the dtype that forward
computed in, and are NOT re-quantized. Loading this folder with load_in_4bit
would put the sum back onto the NF4 grid, and a LoRA delta about the size of one
NF4 step can round away there.

    python collections/utils/hf_export.py <merged.pth> [--out DIR]
"""

import os
import re
import json
import shutil
import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# -----
# NAME MAPPING  (inverse of qwen3_dense.from_pretrained)
# -----

TOP_LEVEL = {
    "tok_emb.weight": "model.embed_tokens.weight",
    "final_norm.scale": "model.norm.weight",
    "out_head.weight": "lm_head.weight",
}

PER_LAYER = {
    "att.W_query.weight": "self_attn.q_proj.weight",
    "att.W_key.weight": "self_attn.k_proj.weight",
    "att.W_value.weight": "self_attn.v_proj.weight",
    "att.out_proj.weight": "self_attn.o_proj.weight",
    "att.q_norm.scale": "self_attn.q_norm.weight",
    "att.k_norm.scale": "self_attn.k_norm.weight",
    "norm1.scale": "input_layernorm.weight",
    "norm2.scale": "post_attention_layernorm.weight",
    "ff.fc1.weight": "mlp.gate_proj.weight",
    "ff.fc2.weight": "mlp.up_proj.weight",
    "ff.fc3.weight": "mlp.down_proj.weight",
}

LAYER_RE = re.compile(r"transformer_blocks\.(\d+)\.(.+)")
HF_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")

# Everything but the weights comes from the base repo the SFT run started from.
BASE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt")
MARKER = "export_info.json"


def hf_name(name):
    """qwen3_dense.py parameter name -> HuggingFace Qwen3ForCausalLM name."""
    if name in TOP_LEVEL:
        return TOP_LEVEL[name]
    m = LAYER_RE.fullmatch(name)
    if m is None or m.group(2) not in PER_LAYER:
        raise KeyError(f"no HuggingFace name for {name!r}")
    return f"model.layers.{m.group(1)}.{PER_LAYER[m.group(2)]}"


def hf_order(name):
    """Embeddings, then layer 0..N-1, then final norm and lm_head."""
    m = HF_LAYER_RE.match(name)
    if m:
        return (1, int(m.group(1)), name)
    return (0 if "embed_tokens" in name else 2, 0, name)


# -----
# BASE MODEL FILES
# -----

def fetch_base_files(repo_id, out_dir):
    from huggingface_hub import hf_hub_download
    for fname in BASE_FILES:
        shutil.copyfile(hf_hub_download(repo_id, fname), out_dir / fname)


def check_config(config, state):
    """Fail early if the base repo is not the architecture the .pth was trained as."""
    n_layers = 1 + max(int(LAYER_RE.fullmatch(k).group(1))
                       for k in state if LAYER_RE.fullmatch(k))
    q_rows = state["transformer_blocks.0.att.W_query.weight"].shape[0]
    kv_rows = state["transformer_blocks.0.att.W_key.weight"].shape[0]
    expected = {
        "vocab_size": state["tok_emb.weight"].shape[0],
        "hidden_size": state["tok_emb.weight"].shape[1],
        "intermediate_size": state["transformer_blocks.0.ff.fc1.weight"].shape[0],
        "num_hidden_layers": n_layers,
        "num_attention_heads": q_rows // config["head_dim"],
        "num_key_value_heads": kv_rows // config["head_dim"],
        "tie_word_embeddings": False,
    }
    wrong = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
    if wrong:
        raise ValueError(f"base config does not match the checkpoint "
                         f"(config, checkpoint): {wrong}")


def write_generation_config(out_dir, config):
    """Stop tokens only.

    The base repo's generation_config.json also sets temperature / top_k / top_p,
    which vLLM applies as defaults to any SamplingParams that leaves them unset.
    Both turn-enders stop: the SFT collate taught <|im_end|> then <|endoftext|>.
    """
    ids = {t["content"]: t["id"] for t in
           json.loads((out_dir / "tokenizer.json").read_text())["added_tokens"]}
    tok_cfg = json.loads((out_dir / "tokenizer_config.json").read_text())
    eos, pad = ids[tok_cfg["eos_token"]], ids[tok_cfg["pad_token"]]
    gen = {"bos_token_id": config.get("bos_token_id"),
           "eos_token_id": [eos, pad], "pad_token_id": pad}
    (out_dir / "generation_config.json").write_text(json.dumps(gen, indent=2))


# -----
# EXPORT + VERIFY
# -----

def _source_info(pth_path):
    st = pth_path.stat()
    return {"source": str(pth_path), "source_bytes": st.st_size,
            "source_mtime": st.st_mtime}


def default_out_dir(pth_path):
    """Next to the source: model_pretrained/<stem>_hf/."""
    pth_path = Path(pth_path)
    return pth_path.with_name(pth_path.stem + "_hf")


def verify_export(pth_path, out_dir, dtype=torch.bfloat16):
    """Every exported tensor equals the .pth tensor cast to dtype, bit for bit."""
    state = torch.load(pth_path, map_location="cpu", mmap=True, weights_only=True)
    weight_map = json.loads(
        (Path(out_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    if len(weight_map) != len(state):
        raise ValueError(f"{len(weight_map)} exported tensors != {len(state)} in the .pth")

    by_file = {}
    for name in state:
        by_file.setdefault(weight_map[hf_name(name)], []).append(name)
    for fname, names in by_file.items():
        with safe_open(str(Path(out_dir) / fname), framework="pt") as f:
            for name in names:
                if not torch.equal(f.get_tensor(hf_name(name)), state[name].to(dtype)):
                    raise ValueError(f"{name} -> {hf_name(name)} differs in {fname}")
    return len(state)


def export_hf(pth_path, out_dir=None, repo_id="Qwen/Qwen3-14B",
              dtype=torch.bfloat16, shard_bytes=5 * 1024 ** 3):
    """Write out_dir from pth_path. Built in <out_dir>.partial and renamed only
    after verify_export passes, so a crash never leaves a loadable half-folder."""
    pth_path = Path(pth_path)
    out_dir = Path(out_dir) if out_dir else default_out_dir(pth_path)
    tmp = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    state = torch.load(pth_path, map_location="cpu", mmap=True, weights_only=True)
    names = sorted(state, key=lambda n: hf_order(hf_name(n)))
    print(f"[export] {pth_path.name}: {len(state)} tensors -> {out_dir}")

    fetch_base_files(repo_id, tmp)
    config = json.loads((tmp / "config.json").read_text())
    check_config(config, state)
    write_generation_config(tmp, config)

    # Plan shards from shapes first: the file names carry the shard count.
    bytes_per = torch.finfo(dtype).bits // 8
    plan, current, size = [], [], 0
    for name in names:
        nbytes = state[name].numel() * bytes_per
        if current and size + nbytes > shard_bytes:
            plan.append(current)
            current, size = [], 0
        current.append(name)
        size += nbytes
    plan.append(current)

    weight_map, total = {}, 0
    for i, shard in enumerate(plan, 1):
        fname = f"model-{i:05d}-of-{len(plan):05d}.safetensors"
        tensors = {hf_name(n): state[n].to(dtype).contiguous() for n in shard}
        save_file(tensors, str(tmp / fname), metadata={"format": "pt"})
        total += sum(t.numel() * t.element_size() for t in tensors.values())
        weight_map.update(dict.fromkeys(tensors, fname))
        print(f"[export]   {fname}: {len(tensors)} tensors", flush=True)
        del tensors

    (tmp / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))

    n = verify_export(pth_path, tmp, dtype)
    print(f"[export] verified {n} tensors bit-identical to the .pth cast to {dtype}")
    (tmp / MARKER).write_text(json.dumps(
        {**_source_info(pth_path), "dtype": str(dtype), "tensors": n,
         "base_repo": repo_id}, indent=2))

    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)
    return out_dir


def ensure_hf_export(pth_path, out_dir=None, repo_id="Qwen/Qwen3-14B"):
    """out_dir if it already holds a verified export of this exact .pth, else export."""
    pth_path = Path(pth_path).resolve()
    out_dir = Path(out_dir) if out_dir else default_out_dir(pth_path)
    marker = out_dir / MARKER
    if marker.exists():
        info = json.loads(marker.read_text())
        if all(info.get(k) == v for k, v in _source_info(pth_path).items()):
            print(f"[export] reusing {out_dir}")
            return out_dir
    elif out_dir.exists():
        raise FileExistsError(f"{out_dir} exists but was not written by hf_export; "
                              f"refusing to overwrite it")
    return export_hf(pth_path, out_dir, repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pth", help="merged .pth from LoRAMergeCheckpoint")
    parser.add_argument("--out", default=None, help="default: <pth stem>_hf/ next to it")
    parser.add_argument("--repo-id", default="Qwen/Qwen3-14B",
                        help="base repo for config + tokenizer")
    args = parser.parse_args()
    ensure_hf_export(args.pth, args.out, args.repo_id)
