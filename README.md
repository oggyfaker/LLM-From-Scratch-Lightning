# Large Language Model — Scratch Everything Anywhere

Implement, train, and serve **every open-source LLM** from scratch — no HuggingFace Transformers, no PEFT library, no black boxes.

### Training Pipeline (from scratch)

| Stage | Description |
|-------|-------------|
| **SFT** | Supervised Fine-Tuning with LoRA / QLoRA (4-bit NF4) |
| **CT** | Continued Pre-Training on domain-specific corpora |
| **RLHF** | Reinforcement Learning from Human Feedback (reward model + PPO) |

Each model is trained end-to-end, evaluated, and documented with **performance benchmarks**, **tricks that worked**, and **problems encountered**.

### Serving Pipeline (from scratch)

Translate the from-scratch model implementation into production serving backends:

| Framework | Focus |
|-----------|-------|
| **vLLM** | PagedAttention, continuous batching, KV cache management |
| **MTP** | Multi-Token Prediction — speculative decoding & parallel token generation |
| **Diffusion** | Diffusion-based LLM decoding — masked diffusion, discrete flow matching |

### Models

| Model | Status |
|-------|--------|
| Qwen3 (0.6B → 32B, MoE 30B-A3B) | ✅ Implemented |
| *More models coming...* | 🚧 |

> **All core components — model architecture, tokenizer, LoRA injection, quantization, and serving integration — are implemented from scratch** to understand exactly what happens at every layer.

---

## Project Structure

```
LLM_From_Scratch_Lightning/
├── 0_Qwen3_SFT_LoRA_4bit.py              # Training script (Lightning)
├── 1_Qwen3_SFT_LoRA_4bit_Inference.ipynb  # Inference notebook (vLLM)
├── data/
│   └── Jigsaw2026/
│       ├── data_utils.py                  # Dataset, collate, prompt formatting
│       ├── train.csv
│       └── test.csv
├── models/
│   └── Qwen3/
│       ├── qwen3.py                       # From-scratch Qwen3 model & tokenizer
│       ├── qwen3_vllm.py                  # vLLM-compatible model wrapper
│       ├── qwen3-moe.py                   # MoE variant (Qwen3-30B-A3B)
│       └── tokenizer.json                 # Shared tokenizer (all Qwen3 sizes)
├── utils/
│   └── checkpoint_utils.py                # LoRA merge, safetensor export, plotting
├── docs/
│   ├── MODEL.md                           # Qwen3 architecture deep-dive
│   ├── TRAINING.md                        # LoRA, quantization & training pipeline
│   └── INFERENCE.md                       # vLLM integration & inference guide
├── checkpoint/                            # Pretrained weights (auto-downloaded)
├── logs/                                  # Training logs & checkpoints
└── README.md
```

---

## Requirements

| Dependency           | Version   |
|----------------------|-----------|
| Python               | 3.12      |
| CUDA                 | 13.0      |
| NVIDIA Driver        | ≥ 580.x   |

---

## Installation

### 1. Create Conda Environment

```bash
conda create -n LLM python=3.12 -y
conda activate LLM
```

### 2. Install PyTorch (CUDA 13.0)

```bash
pip install torch==2.11.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
```

### 3. Install vLLM (Build from Source)

Build the latest vLLM from source with GPU support:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
cd ..
```

> **Note:** Building from source compiles custom CUDA kernels and may take 10–30 minutes depending on your hardware. Ensure `nvcc` (CUDA toolkit) is available in your `PATH`.

### 4. Install Training Dependencies

```bash
pip install pytorch-lightning==2.6.1
pip install bitsandbytes==0.49.2
```

### 5. Install Remaining Libraries

```bash
pip install pandas==3.0.2 \
            safetensors==0.7.0 \
            tokenizers==0.22.2 \
            matplotlib==3.10.8 \
            tqdm==4.67.3 \
            requests==2.33.1
```

### 6. Verify Installation

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

import pytorch_lightning as pl
print(f'Lightning: {pl.__version__}')

import bitsandbytes
print(f'bitsandbytes: {bitsandbytes.__version__}')

from vllm import LLM
print('vLLM: OK')
"
```

---

## Quick Start

### Training

The training script downloads Qwen3-14B pretrained weights automatically on first run:

```bash
conda activate LLM
cd LLM_From_Scratch_Lightning
python 0_Qwen3_SFT_LoRA_4bit.py
```

### Inference

Open the notebook and run cells sequentially:

```bash
jupyter notebook 1_Qwen3_SFT_LoRA_4bit_Inference.ipynb
```

---

## Documentation

For detailed implementation guides, see:

| Document | Description |
|----------|-------------|
| [docs/MODEL.md](docs/MODEL.md) | Qwen3 architecture from scratch — RMSNorm, RoPE, GQA, SwiGLU, tokenizer, model configs (0.6B → 32B), MoE variant |
| [docs/TRAINING.md](docs/TRAINING.md) | LoRA injection, 4-bit NF4 quantization, gradient accumulation, checkpoint callbacks, training config |
| [docs/INFERENCE.md](docs/INFERENCE.md) | vLLM custom model integration, PagedAttention, weight translation, safetensor conversion, inference pipeline |

---

## License

This project is for research and educational purposes.  
Qwen3 model weights are subject to the [Qwen License](https://huggingface.co/Qwen/Qwen3-14B/blob/main/LICENSE).
