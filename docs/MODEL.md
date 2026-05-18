# Qwen3 Model Architecture — From Scratch

This document explains the full Qwen3 dense transformer architecture implemented in [`models/Qwen3/qwen3.py`](../models/Qwen3/qwen3.py), built entirely from PyTorch primitives — no HuggingFace Transformers dependency.

---

## Table of Contents

- [Overview](#overview)
- [Tokenizer](#tokenizer)
- [RMSNorm](#rmsnorm)
- [Rotary Position Embeddings (RoPE)](#rotary-position-embeddings-rope)
- [Grouped Query Attention (GQA)](#grouped-query-attention-gqa)
- [Feed-Forward Network (SwiGLU)](#feed-forward-network-swiglu)
- [Transformer Block](#transformer-block)
- [Full Model](#full-model)
- [Model Configurations](#model-configurations)
- [Weight Loading](#weight-loading)
- [MoE Variant](#moe-variant)

---

## Overview

Qwen3 is a decoder-only transformer with the following design choices:

| Component | Implementation |
|-----------|---------------|
| Normalization | RMSNorm (pre-norm) |
| Position encoding | Rotary Position Embeddings (RoPE) |
| Attention | Grouped Query Attention (GQA) with QK-norm |
| FFN | SwiGLU (gated linear unit with SiLU activation) |
| Vocabulary | 151,936 tokens |
| Precision | bfloat16 |

---

## Tokenizer

**File:** `models/Qwen3/qwen3.py` — `Qwen3Tokenizer`

Wraps the HuggingFace `tokenizers` library (Rust-backed BPE) with custom special token handling.

```python
tokenizer = Qwen3Tokenizer("models/Qwen3/tokenizer.json")
```

**Key features:**
- **Special tokens** handled via regex splitting — `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, etc. are mapped to their IDs before BPE encoding
- **Chat template** wraps user messages in `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` format
- **Thinking mode** — when `enable_thinking=True`, omits the `<think>...</think>` block so the model generates reasoning tokens
- **EOS token** — `<|im_end|>` (151645) for chat models, `<|endoftext|>` (151643) for base models
- **Shared across all sizes** — the same `tokenizer.json` works for 0.6B through 32B

### Naming Convention

Custom parameter names differ from HuggingFace convention:

| Custom (qwen3.py) | vLLM |
|---|---|
| `tok_emb.weight` | `model.embed_tokens.weight` |
| `transformer_blocks.{id}.att.W_query.weight` | `model.layers.{id}.self_attn.q_proj.weight` |
| `transformer_blocks.{id}.att.W_key.weight` | `model.layers.{id}.self_attn.k_proj.weight` |
| `transformer_blocks.{id}.att.W_value.weight` | `model.layers.{id}.self_attn.v_proj.weight` |
| `transformer_blocks.{id}.att.out_proj.weight` | `model.layers.{id}.self_attn.o_proj.weight` |
| `transformer_blocks.{id}.att.q_norm.scale` | `model.layers.{id}.self_attn.q_norm.weight` |
| `transformer_blocks.{id}.att.k_norm.scale` | `model.layers.{id}.self_attn.k_norm.weight` |
| `transformer_blocks.{id}.norm1.scale` | `model.layers.{id}.input_layernorm.weight` |
| `transformer_blocks.{id}.norm2.scale` | `model.layers.{id}.post_attention_layernorm.weight` |
| `transformer_blocks.{id}.ff.fc1.weight` | `model.layers.{id}.mlp.gate_proj.weight` |
| `transformer_blocks.{id}.ff.fc2.weight` | `model.layers.{id}.mlp.up_proj.weight` |
| `transformer_blocks.{id}.ff.fc3.weight` | `model.layers.{id}.mlp.down_proj.weight` |
| `final_norm.scale` | `model.norm.weight` |
| `out_head.weight` | `lm_head.weight` |

---

## RMSNorm

Root Mean Square Layer Normalization — simpler than LayerNorm (no mean centering, no bias by default).

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$$

where $\gamma$ is a learned per-channel scale parameter.

**Implementation detail:** Computation is done in `float32` for numerical stability (`qwen3_compatible=True`), then cast back to the input dtype.

---

## Rotary Position Embeddings (RoPE)

RoPE encodes position information by rotating pairs of dimensions in Q and K vectors.

**Precomputation:**
$$\theta_i = \text{base}^{-2i/d}, \quad \text{angles}[t, i] = t \cdot \theta_i$$

where `base = 1,000,000` for Qwen3 and $d = 128$ (head dimension).

**Application:** Given input $x = [x_1, x_2]$ (split at midpoint):

$$\text{RoPE}(x) = [x_1 \cos\theta - x_2 \sin\theta, \; x_2 \cos\theta + x_1 \sin\theta]$$

Cosine and sine matrices are precomputed for the full context length and registered as non-persistent buffers.

---

## Grouped Query Attention (GQA)

GQA reduces KV cache memory by sharing Key/Value heads across groups of Query heads.

```
Q heads:  40 (Qwen3-14B)
KV heads:  8
Group size: 40 / 8 = 5 query heads per KV group
```

**Forward pass:**
1. Project input → Q (`n_heads × head_dim`), K (`n_kv_groups × head_dim`), V (`n_kv_groups × head_dim`)
2. Apply **QK-norm** (RMSNorm on each head independently) — stabilizes training for large models
3. Apply **RoPE** to Q and K
4. Expand K, V via `repeat_interleave` to match Q head count
5. Compute causal self-attention with upper-triangular mask
6. Project output back to `emb_dim`

**Uptraining insight** (from the GQA paper):  
> You can convert an existing MHA model to GQA by averaging the K/V weight matrices within each group, then fine-tuning with ~5% of the original pre-training compute.

---

## Feed-Forward Network (SwiGLU)

A gated linear unit with SiLU (Swish) activation:

$$\text{FFN}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}) W_{\text{down}}$$

Three linear projections, no bias:
- `fc1` (gate_proj): `emb_dim → hidden_dim`
- `fc2` (up_proj): `emb_dim → hidden_dim`
- `fc3` (down_proj): `hidden_dim → emb_dim`

---

## Transformer Block

Pre-norm residual architecture:

```
x → RMSNorm → GQA → + residual
  → RMSNorm → FFN → + residual
```

---

## Full Model

```python
class Qwen3Model(nn.Module):
    tok_emb            # nn.Embedding(vocab_size, emb_dim)
    transformer_blocks # nn.ModuleList of Block
    final_norm         # RMSNorm
    out_head           # nn.Linear(emb_dim, vocab_size)
    cos, sin           # Precomputed RoPE buffers
```

Forward: `input_ids → embedding → N blocks → final_norm → lm_head → logits`

---

## Model Configurations

All Qwen3 dense models share the same architecture, differing only in dimensions:

| Model | Layers | Emb Dim | Hidden Dim | Heads | KV Groups | Params |
|-------|--------|---------|------------|-------|-----------|--------|
| 0.6B  | 28     | 1024    | 3072       | 16    | 8         | ~0.6B  |
| 1.7B  | 28     | 2048    | 6144       | 16    | 8         | ~1.7B  |
| 4B    | 36     | 2560    | 9728       | 32    | 8         | ~4B    |
| 8B    | 36     | 4096    | 12288      | 32    | 8         | ~8B    |
| 14B   | 40     | 5120    | 17408      | 40    | 8         | ~14B   |
| 32B   | 64     | 5120    | 25600      | 64    | 8         | ~32B   |

Common across all: `vocab_size=151,936`, `head_dim=128`, `qk_norm=True`, `rope_base=1,000,000`, `context_length=40,960`

---

## Weight Loading

### `from_pretrained(model, repo_id)`

Downloads safetensors from HuggingFace (supports sharded models), translates HF key names to custom naming, and loads into the model. Handles both single-file and sharded (index.json) models automatically.

### `from_local_pth(model, pth_path)`

Loads a merged `.pth` state_dict (post-LoRA merge) directly — keys already use custom naming convention.

---

## MoE Variant

**File:** [`models/Qwen3/qwen3-moe.py`](../models/Qwen3/qwen3-moe.py)

Implements **Mixture of Experts** for Qwen3-30B-A3B:

| Parameter | Value |
|-----------|-------|
| Total experts | 128 |
| Active experts per token | 8 |
| Expert hidden dim | 768 |
| Effective hidden dim | 8 × 768 = 6144 |

Each `MoEFeedForward` layer:
1. **Router** (`gate`): linear projection → top-k expert selection
2. **Softmax** over selected expert scores
3. **Parallel expert computation**: each expert is a full SwiGLU FFN
4. **Weighted aggregation**: outputs scaled by router probabilities

The attention mechanism is identical to the dense model, using FlashAttention via PyTorch 2.x `scaled_dot_product_attention`.

---

## Text Generation

The `generate()` function supports:
- **Greedy decoding** (`temperature=0.0`)
- **Temperature sampling** with optional **top-k filtering**
- **EOS stopping** — stops at `<|im_end|>` (chat) or `<|endoftext|>` (base)

```python
output = generate(model, tokenizer, prompt, max_length=200, temperature=0.0)
```
