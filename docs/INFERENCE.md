# Inference Pipeline — vLLM Integration

This document explains how the custom Qwen3 model integrates with vLLM for high-throughput inference, implemented in [`models/Qwen3/qwen3_vllm.py`](../models/Qwen3/qwen3_vllm.py) and the inference notebook [`1_Qwen3_SFT_LoRA_4bit_Inference.ipynb`](../1_Qwen3_SFT_LoRA_4bit_Inference.ipynb).

---

## Table of Contents

- [Overview](#overview)
- [Why vLLM?](#why-vllm)
- [Architecture: Qwen3vLLM](#architecture-qwen3vllm)
- [vLLM Required Interface](#vllm-required-interface)
- [Weight Translation](#weight-translation)
- [Checkpoint Conversion](#checkpoint-conversion)
- [Inference Pipeline](#inference-pipeline)
- [Sampling Parameters](#sampling-parameters)

---

## Overview

The from-scratch Qwen3 model (`qwen3.py`) is designed for **training**. For **inference**, we wrap it in a vLLM-compatible class (`qwen3_vllm.py`) that replaces vanilla attention with PagedAttention and uses vLLM's optimized kernels.

```
Training model (qwen3.py)          →  Inference model (qwen3_vllm.py)
─────────────────────────          ─────────────────────────────────
nn.Linear                         →  QKVParallelLinear / RowParallelLinear
Manual causal mask attention       →  PagedAttention + KV cache
Manual RoPE computation            →  vLLM's fused get_rope
Sequential token generation        →  Continuous batching engine
```

---

## Why vLLM?

| Feature | Naive Generation | vLLM |
|---------|-----------------|------|
| KV cache | Recompute all tokens each step | PagedAttention — cache K/V, only compute new token |
| Memory | Full sequence × batch in GPU | Paged blocks — near-zero waste |
| Batching | One request at a time | Continuous batching — process multiple requests simultaneously |
| Throughput | ~1 token/s (14B on single GPU) | ~100+ tokens/s |

---

## Architecture: Qwen3vLLM

**File:** [`models/Qwen3/qwen3_vllm.py`](../models/Qwen3/qwen3_vllm.py)

### Class hierarchy

```python
Qwen3vLLM (base class)          # Implements vLLM interface
├── Qwen3_06B_vLLM              # MODEL_CFG = QWEN_06B_CFG
├── Qwen3_1B7_vLLM              # MODEL_CFG = QWEN_1B7_CFG
├── Qwen3_4B_vLLM               # MODEL_CFG = QWEN_4B_CFG
├── Qwen3_8B_vLLM               # MODEL_CFG = QWEN_8B_CFG
├── Qwen3_14B_vLLM              # MODEL_CFG = QWEN_14B_CFG
└── Qwen3_32B_vLLM              # MODEL_CFG = QWEN_32B_CFG
```

Each subclass is a one-liner that sets `MODEL_CFG` to the appropriate config dataclass. The base class handles everything else.

### Component mapping

| Training (qwen3.py) | vLLM (qwen3_vllm.py) | Purpose |
|---|---|---|
| `nn.Linear` (Q, K, V separate) | `QKVParallelLinear` | Fused Q/K/V projection, tensor-parallel ready |
| `nn.Linear` (output proj) | `RowParallelLinear` | Row-parallel for multi-GPU |
| `nn.Linear` (gate + up) | `MergedColumnParallelLinear` | Fused gate_proj + up_proj |
| `nn.Linear` (down) | `RowParallelLinear` | Row-parallel down projection |
| `nn.Embedding` | `VocabParallelEmbedding` | Vocab-parallel embedding |
| `nn.Linear` (lm_head) | `ParallelLMHead` | Parallel LM head |
| Manual causal mask | `VllmAttention` | PagedAttention with KV cache |
| `compute_rope_params` + `apply_rope` | `get_rope` | vLLM's fused RoPE kernel |
| `SiLU(gate) * up` | `SiluAndMul` | Fused SiLU + elementwise multiply |
| Custom `RMSNorm` | vLLM `RMSNorm` | Fused RMSNorm kernel |

---

## vLLM Required Interface

vLLM (v0.20.1) requires custom models to implement these methods:

### `__init__(self, *, vllm_config: VllmConfig, prefix: str = "")`

- Receives `vllm_config` containing `hf_text_config` (parsed from `config.json`)
- Builds the model using the config dataclass set by `MODEL_CFG`

### `embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor`

- Embeds token IDs into hidden states
- Called by vLLM's EngineCore for input processing

### `forward(self, input_ids, positions, intermediate_tensors, inputs_embeds) -> torch.Tensor`

- Main forward pass — returns hidden states (not logits)
- `positions` tensor provides absolute positions for RoPE
- `inputs_embeds` allows passing pre-computed embeddings

### `compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor`

- Projects hidden states through `lm_head` → logits
- Uses vLLM's `LogitsProcessor` for proper handling

### `load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]`

- Loads weights from safetensors files
- Handles stacked QKV and merged gate_up projections

---

## Weight Translation

The `load_weights` method handles three types of weight loading:

### 1. Stacked QKV weights

vLLM's `QKVParallelLinear` expects a single fused `qkv_proj` weight. The safetensors file has separate `q_proj`, `k_proj`, `v_proj` weights:

```python
stacked_params_mapping = [
    ("qkv_proj", "q_proj", "q"),   # shard_id="q"
    ("qkv_proj", "k_proj", "k"),   # shard_id="k"
    ("qkv_proj", "v_proj", "v"),   # shard_id="v"
]
```

Each projection is loaded into the correct shard of `qkv_proj` via vLLM's `weight_loader`.

### 2. Merged gate_up weights

vLLM's `MergedColumnParallelLinear` fuses gate and up projections:

```python
merged_params_mapping = [
    ("gate_up_proj", "gate_proj", 0),  # shard_idx=0
    ("gate_up_proj", "up_proj", 1),    # shard_idx=1
]
```

### 3. Custom name translation

If weights use the custom qwen3.py naming (from merged `.pth` files), `_translate_custom_name()` converts them to HF convention before loading. See [MODEL.md — Naming Convention](MODEL.md#naming-convention) for the full mapping table.

---

## Checkpoint Conversion

The inference notebook converts a merged `.pth` checkpoint to vLLM-compatible format.

### `convert_pth_to_vllm()`

```python
convert_pth_to_vllm(
    tokenizer=Qwen3Tokenizer("./models/Qwen3/tokenizer.json"),
    model_cfg=QWEN_14B_CFG,
    pth_path="./logs/.../model_pretrained/00-0.2356-0.8897.pth",
    output_dir="./model_vllm",
)
```

**Steps:**
1. **Check** — skips if `model.safetensors`, `config.json`, `tokenizer.json` already exist
2. **Load** `.pth` state_dict (custom naming)
3. **Translate** all keys from custom → HF naming via `translate_pretrained_to_vLLM()`
4. **Save** as `model.safetensors`
5. **Generate** `config.json` with architecture metadata:
   ```json
   {
     "architectures": ["Qwen3vLLM"],
     "model_type": "qwen3",
     "vocab_size": 151936,
     "hidden_size": 5120,
     "intermediate_size": 17408,
     "num_hidden_layers": 40,
     "num_attention_heads": 40,
     "num_key_value_heads": 8,
     "head_dim": 128,
     "max_position_embeddings": 40960,
     "rope_theta": 1000000.0,
     "tie_word_embeddings": false,
     "torch_dtype": "bfloat16"
   }
   ```
6. **Save** `tokenizer.json` via `Qwen3Tokenizer._tok.save()`

> **Important:** `"architectures": ["Qwen3vLLM"]` must match the name used in `ModelRegistry.register_model()`.

---

## Inference Pipeline

### Step 1: Register custom model

```python
from vllm import ModelRegistry
from models.Qwen3.qwen3_vllm import Qwen3_14B_vLLM

ModelRegistry.register_model("Qwen3vLLM", Qwen3_14B_vLLM)
```

### Step 2: Launch vLLM engine

```python
from vllm import LLM

llm = LLM(
    model="./model_vllm",          # Path to safetensors + config.json
    dtype="bfloat16",
    trust_remote_code=True,
    gpu_memory_utilization=0.85,
)
```

vLLM forks an `EngineCore` subprocess, loads weights, and initializes KV cache.

### Step 3: Generate

```python
from vllm import SamplingParams

prompt = (
    "<|im_start|>system\n"
    "Reddit moderation: Does the comment violate the rule? Answer 'Yes' or 'No' only.<|im_end|>\n"
    "<|im_start|>user\n"
    f"Comment: {body}\n\nrule: {rule}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8,
    stop_token_ids=[151643, 151645],  # eos_token, <|im_end|>
)

outputs = llm.generate([prompt], sampling_params)
prediction = outputs[0].outputs[0].text.strip()  # "Yes" or "No"
```

---

## Sampling Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `temperature` | 0.0 | Greedy decoding — deterministic Yes/No |
| `max_tokens` | 8 | Safety cap — answer is 1–2 tokens |
| `stop_token_ids` | [151643, 151645] | Stop at `<\|endoftext\|>` or `<\|im_end\|>` |

### Why stop tokens matter

The model is trained to generate `Yes<|im_end|>` or `No<|im_end|>`. Without `stop_token_ids`, vLLM continues generating up to `max_tokens`, causing repetition loops like `NoNoNo...` because the model's most probable continuation after `<|im_end|>` is another answer token.

---

## Batch Inference

For processing the full test dataset:

```python
prompts = []
for _, row in test_df.iterrows():
    prompt = format_prompt(row["body"], row["rule"])
    prompts.append(prompt)

outputs = llm.generate(prompts, sampling_params)  # All at once → continuous batching
predictions = [o.outputs[0].text.strip() for o in outputs]
```

vLLM handles batching, scheduling, and KV cache management automatically.
