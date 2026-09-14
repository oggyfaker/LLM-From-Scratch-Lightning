<div align="center">

# LLM Collections

### Every SOTA LLM, rebuilt from scratch in pure PyTorch — and proven correct.

*No `transformers`. No `peft`. No `trl`. Just tensors you can read, line by line.*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Lightning](https://img.shields.io/badge/Lightning-2.6-792EE5?logo=lightning&logoColor=white)](https://lightning.ai)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![From Scratch](https://img.shields.io/badge/from--scratch-100%25-black)](.)

</div>

---

## 🎉 Latest Updates

- 2026/09:
  - **Chain-of-thought SFT for GSM8K** — [`2_Qwen3_Gsm8k_FineTune.py`](qwen3/2_Qwen3_Gsm8k_FineTune.py) now trains on the full `<think>…</think>` reasoning trace instead of the bare `\boxed{}` answer. Flip `INCLUDE_THINKING` to switch between the two recipes.
  - **Reworked training metrics** — token-weighted validation accuracy, cumulative supervised-token counts, grad-norm and VRAM per optimizer step, plus a full test-set evaluation every 20% of training.

- 2026/05:
    - **vLLM serving path** — the same from-scratch module wrapped for PagedAttention and continuous batching.
    - **Train Qwen3-MoE with LoRA-QLoRA** — Adding 3D expert-tensor LoRA for SFT training MoE (another layer have 2D LoRA as normal).
    - **Qwen3-MoE from Unsloth** — Reuse MoE model from Unsloth. I can't implement 3D tensor with MoE layers.
    - **Train Qwen3 with LoRA and QLoRA** — 4-bit NF4 and 8-bit LLM.int8() base quantization, optional LoRA+ discriminative learning rates.
    - **Qwen3 dense from scratch, 0.6B → 32B** — Implement from srcatch and verified against the official pretrained weights.

---

## ✨ Overview

**Collections** This is the workflow for each folder model in here. This will have enough these step

```
Implement            Verify              Train                      Post-Train           Evaluate
scratch backbone ─►  HF safetensors ──►  SFT w/LoRA-QLoRA...  ───►  RLVR w/GRPO-DPO ───► Compare w/original pretrained
```


---
