# Training Pipeline — LoRA + 4-bit Quantization

This document explains the full training pipeline in [`collections/qwen3/0_Qwen3_SFT_LoRA_QLoRA.py`](../collections/qwen3/0_Qwen3_SFT_LoRA_QLoRA.py), covering LoRA injection, 4-bit quantization, gradient accumulation, and checkpoint management.

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Training Configuration](#training-configuration)
- [LoRA (Low-Rank Adaptation)](#lora-low-rank-adaptation)
- [4-bit NF4 Quantization](#4-bit-nf4-quantization)
- [Data Pipeline](#data-pipeline)
- [Gradient Accumulation](#gradient-accumulation)
- [Learning Rate Schedule](#learning-rate-schedule)
- [Checkpoint Callbacks](#checkpoint-callbacks)
- [Running Training](#running-training)

---

## Pipeline Overview

```
Pretrained Qwen3-14B (bf16, ~28GB)
    → Freeze all parameters
    → Inject LoRA on Q/K/V/O/gate/up/down projections
    → Replace frozen linears with 4-bit NF4 (bitsandbytes)
    → Train on Jigsaw 2026 dataset (Yes/No classification)
    → Manual gradient accumulation (reduction='sum')
    → Cosine LR decay with linear warmup
    → LoRA merge → .pth export
```

Memory footprint: **~12GB GPU** (14B params in NF4 ≈ 7GB + LoRA params + optimizer states + activations)

---

## Training Configuration

Defined in the `CFG` dataclass:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 1 | Number of training epochs |
| `BATCH_SIZE` | 1 | Micro-batch size (per gradient accumulation step) |
| `GRAD_ACCUM` | 16 | Gradient accumulation steps → effective batch size = 16 |
| `LR` | 1.5e-4 | Peak learning rate |
| `MIN_LR` | 1.5e-5 | Minimum LR (10% of peak, for cosine decay floor) |
| `WARMUP_STEPS` | 20 | Linear warmup steps (~10% of total optimizer steps) |
| `MAX_SEQ_LEN` | 256 | Maximum sequence length (truncate longer sequences) |
| `SEED` | 1001 | Random seed |
| `WORKERS` | 2 | DataLoader workers |

---

## LoRA (Low-Rank Adaptation)

### Concept

Instead of fine-tuning all 14B parameters, LoRA freezes the base model and injects small trainable low-rank matrices alongside each target linear layer.

$$h = W_{\text{base}} x + \frac{\alpha}{r} (x A B)$$

where:
- $W_{\text{base}}$ — frozen pretrained weight
- $A \in \mathbb{R}^{d_{\text{in}} \times r}$ — trainable, Kaiming uniform init
- $B \in \mathbb{R}^{r \times d_{\text{out}}}$ — trainable, zero init (output is zero at start)
- $\alpha / r$ — scaling factor

### Implementation

```python
class LoRALayer(nn.Module):
    # A: (in_dim, rank) — Kaiming uniform init
    # B: (rank, out_dim) — zero init
    # scaling = alpha / rank

class LinearWithLoRA(nn.Module):
    # forward(x) = base_linear(x) + lora(x)
```

### Target Modules

LoRA is applied to **all** attention and FFN projections:

| Module | Parameters per layer |
|--------|---------------------|
| `W_query` | 5120 × 5120 → rank 16 |
| `W_key` | 5120 × 1024 → rank 16 |
| `W_value` | 5120 × 1024 → rank 16 |
| `out_proj` | 5120 × 5120 → rank 16 |
| `fc1` (gate) | 5120 × 17408 → rank 16 |
| `fc2` (up) | 5120 × 17408 → rank 16 |
| `fc3` (down) | 17408 × 5120 → rank 16 |

**Config:** `rank=16`, `alpha=32` → scaling = 2.0

**Result:** ~0.12% trainable parameters out of 14B total.

---

## 4-bit NF4 Quantization

After LoRA injection, frozen `nn.Linear` layers are replaced with `bitsandbytes.nn.Linear4bit` using **NF4** (Normal Float 4-bit) quantization.

### How it works

1. `apply_4bit_quantization()` walks the model tree
2. For `LinearWithLoRA` modules: replaces only the **frozen base** `module.linear` with `Linear4bit`
3. For standalone frozen `nn.Linear`: replaces the entire module
4. **Skipped modules:** `tok_emb` and `out_head` (need full precision for embedding lookup and logit computation)
5. **Actual compression happens on `.cuda()`** — weights are quantized to 4-bit on GPU transfer

### Compute dtype

Forward pass computation uses `bfloat16` (`compute_dtype=torch.bfloat16`) — weights are dequantized on-the-fly during matmul.

---

## Data Pipeline

### Dataset: Jigsaw 2026

Binary classification — does a Reddit comment violate a subreddit rule?

**Prompt format** (Qwen3 chat template):
```
<|im_start|>system
Reddit moderation: Does the comment violate the rule? Answer 'Yes' or 'No' only.<|im_end|>
<|im_start|>user
Comment: {body}

rule: {rule}<|im_end|>
<|im_start|>assistant
{Yes/No}<|im_end|>
```

### Data Augmentation

1. Training rows from `train.csv` (labeled `rule_violation`)
2. Positive/negative examples extracted from `test.csv` columns
3. Deduplication
4. 2x upsampling of test-extracted examples
5. Stratified 90/10 train/val split

### Collate Function

- Appends **EOS token** after each sequence
- Pads to max length in batch
- Creates **shifted targets** (teacher forcing)
- **Masks prompt tokens** (system + user) → loss computed only on assistant response tokens (`Yes`/`No` + `<|im_end|>`)
- Masks padding tokens (keeps first pad = real EOS)
- Truncates to `MAX_SEQ_LEN`

---

## Gradient Accumulation

Uses **manual optimization** with `reduction='sum'` for mathematically correct gradient accumulation.

### Why `reduction='sum'`?

Standard `reduction='mean'` divides by token count per micro-batch. When micro-batches have different numbers of non-masked tokens, averaging across accumulation steps gives incorrect gradients. Using `reduction='sum'` and dividing by total tokens after accumulation is equivalent to full-batch training.

### Step-by-step

```
For each micro-batch i in [1, ..., G]:
    1. loss_sum_i = CE(logits, targets, reduction='sum')
    2. backward(loss_sum_i)           # gradients accumulate
    3. accum_loss  += loss_sum_i
    4. accum_tokens += num_tokens_i

After G steps:
    5. grad /= accum_tokens           # correct normalization
    6. clip_grad_norm_(max_norm=1.0)
    7. optimizer.step()
    8. scheduler.step()
    9. train_loss = accum_loss / accum_tokens
```

Leftover micro-batches at epoch end are flushed in `on_train_epoch_end`.

---

## Learning Rate Schedule

**Warmup + Cosine Decay:**

```
Steps 0..19:        Linear warmup (0 → LR)
Steps 20..STEPS:    Cosine decay (LR → MIN_LR)
Steps > STEPS:      Constant MIN_LR
```

$$\text{LR}(t) = \text{MIN\_LR} + \frac{1}{2}(\text{LR} - \text{MIN\_LR})(1 + \cos(\pi \cdot \text{decay\_ratio}))$$

### Optimizer

AdamW with:
- **Weight decay** = 0.01 on 2D+ parameters (weight matrices)
- **No decay** on 1D parameters (RMSNorm scales)
- $\beta = (0.9, 0.95)$, $\epsilon = 10^{-8}$

---

## Checkpoint Callbacks

### 1. `ModelCheckpoint` (Lightning built-in)

Saves full `.ckpt` files for **resuming training**:
- Monitors `val_mean_token_accuracy` (mode=max)
- Keeps top 3 + last checkpoint
- Includes optimizer state, scheduler state, etc.

### 2. `LoRAMergeCheckpoint` (custom)

Saves **merged `.pth` files** for direct `Qwen3Model` loading:
- Dequantizes NF4 base weights → float
- Folds LoRA: $W_{\text{merged}} = W_{\text{base}} + \frac{\alpha}{r} (AB)^T$
- Saves plain state_dict (custom naming convention)
- Top-k management with automatic cleanup

### 3. `SafetensorCheckpoint` (custom, currently commented out)

Converts best merged `.pth` → vLLM-ready format:
- Translates custom key names to HuggingFace convention
- Saves as `model.safetensors`
- Generates `config.json` with architecture metadata
- Copies `tokenizer.json`

> **Note:** This conversion is now done in the inference notebook (`1_Qwen3_SFT_LoRA_4bit_Inference.ipynb`) via `convert_pth_to_vllm()`.

### Validation Metrics

| Metric | Description |
|--------|-------------|
| `total_val_loss` | Average CE loss over validation set |
| `val_mean_token_accuracy` | Proportion of correct top-1 token predictions (weighted by token count) |

---

## Running Training

```bash
conda activate LLM
cd LLM_From_Scratch_Lightning

# First run: auto-downloads Qwen3-14B pretrained weights (~28GB)
python collections/qwen3/0_Qwen3_SFT_LoRA_QLoRA.py
```

### Output directory structure

```
logs/9_Qwen3_14B_Jigsaw_LoRA_r16_a32/{date}/version_N/
├── checkpoints/              # Lightning .ckpt files
│   ├── epoch=00-....ckpt
│   └── last.ckpt
├── model_pretrained/         # Merged .pth files (LoRA folded in)
│   └── 00-0.2356-0.8897.pth
├── hparams.yaml
├── metrics.csv               # Training/validation metrics per step
└── loss_curve.png            # Auto-generated training visualization
```
