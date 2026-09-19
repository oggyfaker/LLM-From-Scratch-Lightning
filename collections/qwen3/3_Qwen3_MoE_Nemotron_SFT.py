"""
Qwen3-30B-A3B (MoE) Reasoning Fine-tuning: From-Scratch LoRA + Unsloth + Lightning
────────────────────────────────────────────────────────────────────────────────
Architecture:
- Unsloth FastLanguageModel backbone + custom LoRA + MoE tying + Lightning
- CFG.TUNING_MODE switches between LoRA (bf16 base) and QLoRA (4bit/8bit base)
- lm_head in TARGET_MODULES (LoRA on output projection)
- Dataset: NemotronReasoning2026 train.json / test.json, same record schema as
  data/Gsm8k (question / thinking / answer / num_gt_tokens + problem_id, category)
- Loss: Only on assistant response (prompt masked), using Cut Cross-Entropy
- Metrics: mirrors 1_Qwen3_Gsm8k_SFT.py, plus a generative boxed-answer
  evaluation every INFER_EVERY_PERCENTAGE of training

CCE with LoRA on lm_head:
    Since lm_head has LoRA (LinearWithLoRA), we compose the effective weight:
        effective_W = base_W + scaling * B^T @ A^T
    Then: linear_cross_entropy(hidden, effective_W, targets)
    This is mathematically equivalent to LinearWithLoRA.forward() + CE,
    but avoids materializing the full [batch, seq, vocab] logits tensor.
    Critical for long reasoning sequences (4096+ tokens × 151,936 vocab).

Memory-Efficient MoE LoRA:
    Instead of materializing full [E, out, in] delta (parametrize approach = OOM),
    we inject LoRA weights into Unsloth's native _unsloth_lora_* interface.
    Unsloth backends compute LoRA as separated small matmuls:
        output += scaling * (x @ A[e].T) @ B[e].T
    Intermediates are only [tokens, rank=16] — negligible memory.
"""
import gc
import re
import sys
import json
import math
import time
import random
from pathlib import Path
from functools import partial
from dataclasses import dataclass
# Unsloth's compiled cache defaults to a RELATIVE path, so it lands in whatever
# directory the process was launched from. Pin it next to this file instead.
# Must be set BEFORE importing unsloth — compiler.py reads it at import time.
import os
os.environ.setdefault(
    "UNSLOTH_COMPILE_LOCATION",
    str(Path(__file__).resolve().parent / "unsloth_compiled_cache"),
)

from unsloth import FastLanguageModel

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cut_cross_entropy import linear_cross_entropy
torch.set_float32_matmul_precision('high')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLLECTIONS_DIR = PROJECT_ROOT / "collections"
MODELS_DIR = Path(__file__).resolve().parent / "models"
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(COLLECTIONS_DIR))
sys.path.append(str(MODELS_DIR))

from data.NemotronReasoning2026.data_utils import (
    load_nemotron_json, NemotronReasoningDataset, custom_collate_fn,
    format_prompt_only
)
from utils.checkpoint_utils import plot_training_curves
from utils.checkpoint_moe_utils import MoELoRAMergeCheckpoint


# ──────────────
#    CONFIG
# ──────────────

@dataclass
class CFG:
    EPOCHS = 1
    WORKERS = 0  # Must be 0 to avoid fork+CUDA deadlock with Unsloth
    BATCH_SIZE = 1
    VAL_BATCH_SIZE = 1
    GRAD_ACCUM = 16
    MAX_SEQ_LEN = 8192    # Longest train sample is 7953 tok, so nothing is dropped

    INCLUDE_THINKING = True   # False: predict \boxed{answer} directly | True: train on the reasoning chain

    RANK = 16
    ALPHA = 32
    TUNING_MODE = "qlora" # "lora": lora mode | "qlora": quantized lora mode
    QUANT_BITS = "4bit"   # "4bit" or "8bit", only used with TUNING_MODE = "qlora"

    # ── MoE ──
    MOE_TIE_WEIGHTS = True  # Tie one side of MoE expert LoRA across all 128 experts

    SEED = 1001
    LR = 1.5e-4
    MIN_LR = LR * 0.1
    GRAD_NORM = 1.0
    WARMUP_STEPS = 20
    LORAPLUS_RATIO = None  # Set e.g. 4.0 to enable LoRA+ (None to disable)

    # --- Experiment tracking (Lightning AI) ---
    TEAMSPACE = "LLM-From-Scratch"   # lightning.ai teamspace holding the experiments

    # --- Terminal progress ---
    PROGRESS_EVERY_N_STEPS = 5   # one [train] line per N optimizer steps

    # --- Periodic generative evaluation on the held-out test set ---
    # One round over 32 questions costs ~3 h (completions run to ~7000 tokens),
    # so 0.50 keeps evaluation at ~6 h against ~6 h of training.
    INFER_EVERY_PERCENTAGE = 0.50     # evaluate at 50% and 100% of training

    # Chains run to ~7.6k ground-truth tokens, so scoring all 404 test questions
    # would cost more than the training. 32 carries ~+-17 points of noise: read
    # the trend across rounds, not one round's number.
    INFER_SAMPLES = 32

    # ~0.8 GB of KV cache per 8k sequence, against the ~6 GB left once the
    # training state is resident, so batch 8 sits on the cliff and 4 does not.
    INFER_BATCH_SIZE = 4

    # Full budget, so the model is never cut off before closing </think> and
    # writing \boxed{}. _generate_batch clamps per batch to MAX_SEQ_LEN - prompt.
    INFER_MAX_NEW_TOKENS = MAX_SEQ_LEN if INCLUDE_THINKING else 64

    # ── Model ──
    MODEL_PATH = "Qwen/Qwen3-30B-A3B"

    # ── Target Modules (includes lm_head for reasoning) ──
    # lm_head gets LoRA to improve next-token prediction quality for reasoning
    TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lm_head",
    ]


# ─────────────────────────
#  LoRA Layers (2D - 3D)
# ─────────────────────────

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.scaling * (x @ self.A @ self.B)

class LoRA3DLayer(nn.Module):
    def __init__(self, num_experts, in_features, out_features, rank, alpha):
        super().__init__()
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(num_experts, rank, in_features))
        self.B = nn.Parameter(torch.zeros(num_experts, out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    @property
    def weight(self):
        """Effective weight [out, in] = base + LoRA delta, returned in base dtype."""
        A = self.lora.A    # [in, rank]
        B = self.lora.B    # [rank, out]
        return (self.linear.weight + self.lora.scaling * (B.T @ A.T)).to(self.linear.weight.dtype)

    @property
    def bias(self):
        """Delegate to the base linear, so the wrapper stays a drop-in."""
        return self.linear.bias

    def forward(self, x):
        return self.linear(x) + self.lora(x)


# ───────────────────────
#  APPLY LORA FUNCTIONS
# ───────────────────────

def _replace_linear_with_lora(module, rank, alpha, target_modules):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in target_modules:
            setattr(module, name, LinearWithLoRA(child, rank, alpha))
        else:
            _replace_linear_with_lora(child, rank, alpha, target_modules)

def _expert_lora_hook(experts_module, args):
    if hasattr(experts_module, '_gate_up_lora'):
        lora = experts_module._gate_up_lora
        experts_module._unsloth_lora_gate_up_proj = (
            lora.A.transpose(1, 2), lora.B.transpose(1, 2), lora.scaling
        ) #Unsloth expects: (first[E, in, rank], second[E, rank, out], scaling)

    if hasattr(experts_module, '_down_lora'):
        lora = experts_module._down_lora
        experts_module._unsloth_lora_down_proj = (
            lora.A.transpose(1, 2), lora.B.transpose(1, 2), lora.scaling
        )

def _apply_expert_lora(model, rank, alpha):
    """Attach 3D LoRA to packed expert tensors -> number of tensors covered."""
    count = 0
    for _, module in model.named_modules():
        attached = False

        # gate_up_proj: packed gate and up expert weights [E, 2*inter, hidden]
        if hasattr(module, 'gate_up_proj'):
            p = module.gate_up_proj
            if isinstance(p, nn.Parameter) and p.dim() == 3:
                num_expert, out_dim, in_dim = p.shape
                lora = LoRA3DLayer(num_expert, in_dim, out_dim, rank, alpha).to(p.device)
                module.add_module('_gate_up_lora', lora)
                attached = True
                count += 1

        # down_proj: expert down-projection weights [E, hidden, inter]
        if hasattr(module, 'down_proj'):
            p = module.down_proj
            if isinstance(p, nn.Parameter) and p.dim() == 3:
                num_expert, out_dim, in_dim = p.shape
                lora = LoRA3DLayer(num_expert, in_dim, out_dim, rank, alpha).to(p.device)
                module.add_module('_down_lora', lora)
                attached = True
                count += 1

        if attached:
            module.register_forward_pre_hook(_expert_lora_hook)

    return count

def apply_lora(model, rank, alpha, target_modules):
    """Apply from-scratch LoRA to the model.
    1. Freeze all parameters
    2. Inject nn.Linear with LinearWithLoRA (attention, shared_expert, lm_head)
    3. Inject ExpertLoRA modules + hooks for 3D packed expert params
    """
    # 1. Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # 2. Apply LinearWithLoRA to nn.Linear modules (attention, shared_expert, lm_head)
    _replace_linear_with_lora(model, rank, alpha, target_modules)
    n_linear = sum(isinstance(m, LinearWithLoRA) for m in model.modules())

    # 3. Apply ExpertLoRA to 3D packed expert params (gate_up_proj, down_proj)
    n_expert = _apply_expert_lora(model, rank, alpha)

    print(f"LoRA injected: {n_linear} LinearWithLoRA, {n_expert} packed expert tensors")

    # Fail fast: a silent miss still trains (loss falls on the attention
    # adapters alone) and only surfaces as a checkpoint with no expert LoRA.
    if n_linear == 0:
        raise RuntimeError(f"No nn.Linear matched TARGET_MODULES={target_modules}.")
    if n_expert == 0:
        raise RuntimeError(
            "No 3D expert tensors (gate_up_proj / down_proj) found — experts would "
            "train no adapters. Needs Unsloth's fused Qwen3-MoE layout; a per-expert "
            "nn.Linear layout (experts.N.up_proj / down_proj, as in NemotronH) "
            "needs a per-expert recipe instead.")

    return model


# ────────────────────────
#     MOE WEIGHT TYING
# ────────────────────────

def setup_moe_tying(model):
    """Setup MoE weight tying for LoRA.
        - gate_up_proj (w1): Tie A → all experts share input projection, free B
        - down_proj (w2): Tie B → all experts share output projection, free A
    """
    moe_tied_params = []
    w1_param_names = ("_gate_up_lora",)
    w2_param_names = ("_down_lora",)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # w1: tie A (all experts share input projection)
        # w2: tie B (all experts share output projection)
        is_w1 = any(p in name for p in w1_param_names)
        is_w2 = any(p in name for p in w2_param_names)
        is_A = name.endswith(".A")
        is_B = name.endswith(".B")
        should_tie = (is_w1 and is_A) or (is_w2 and is_B)

        if not should_tie:
            continue
        if param.dim() < 3 or param.shape[0] <= 1:
            continue
        moe_tied_params.append(param)

    # Initialize average across expert dim
    with torch.no_grad():
        for p in moe_tied_params:
            mean = p.data.mean(dim=0, keepdim=True)
            p.data.copy_(mean.expand_as(p.data))

    def tie_grads():
        with torch.no_grad():
            for p in moe_tied_params:
                if p.grad is None:
                    continue
                grad_sum = p.grad.sum(dim=0, keepdim=True)
                p.grad.copy_(grad_sum.expand_as(p.grad))

    return tie_grads


# ──────────────────────────────────
#  ATTENTION MASK CREATION
# ──────────────────────────────────

def make_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Binary 2D attention mask: 1 for real tokens, 0 for pad positions.

    With BATCH_SIZE=1 and no padding this is all-ones, but passing it
    explicitly ensures Flash Attention / Unsloth kernels always receive
    the correct mask regardless of batch size or backend.
    """
    return (input_ids != pad_token_id).long()


# ──────────────────────────────────
#  TERMINAL PROGRESS (PIPE-SAFE)
# ──────────────────────────────────

class TrainingProgressLog(pl.Callback):
    """One compact progress line every N optimizer steps.

    Lightning's default RichProgressBar disappears as soon as stdout is a pipe
    (`Console().is_terminal` is False), so `python train.py | tee run.log` shows
    no progress at all. Plain print() behaves the same under TTY, pipe or tmux.
    """

    def __init__(self, total_steps, every_n_steps=CFG.PROGRESS_EVERY_N_STEPS):
        super().__init__()
        self.total_steps = max(1, total_steps)
        self.every_n_steps = max(1, every_n_steps)
        self._t0 = None
        self._last_logged = -1

    @staticmethod
    def _metric(trainer, key, default=float('nan')):
        value = trainer.callback_metrics.get(key)
        if value is None:
            return default
        return value.item() if hasattr(value, 'item') else float(value)

    def on_train_start(self, trainer, pl_module):
        self._t0 = time.time()
        print(f"[train] {self.total_steps} optimizer steps "
              f"({CFG.BATCH_SIZE}x{CFG.GRAD_ACCUM} samples each)", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = pl_module._opt_step
        # Fires per micro-batch; _opt_step advances once per GRAD_ACCUM of them.
        if step == 0 or step == self._last_logged:
            return
        self._last_logged = step
        if step % self.every_n_steps and step != self.total_steps:
            return

        elapsed = time.time() - (self._t0 or time.time())
        rate = step / max(1e-9, elapsed)
        eta_h = (self.total_steps - step) / max(1e-9, rate) / 3600
        print(f"[train] step {step:>4}/{self.total_steps} "
              f"{100.0 * step / self.total_steps:>3.0f}% | "
              f"loss {self._metric(trainer, 'Training/train_loss'):.4f} | "
              f"lr {self._metric(trainer, 'Training/lr'):.2e} | "
              f"grad {self._metric(trainer, 'Training/grad_norm'):.2f} | "
              f"{self._metric(trainer, 'Training/gpu_mem_gb'):.1f} GB | "
              f"{rate * 60:.1f} step/min | eta {eta_h:.1f} h", flush=True)

    def on_validation_end(self, trainer, pl_module):
        # Teacher-forced numbers; the solve rate is printed by the eval callback.
        loss = self._metric(trainer, 'Validation/loss')
        acc = self._metric(trainer, 'Validation/token_accuracy')
        if loss == loss:   # NaN -> sanity-check pass, nothing logged yet
            print(f"[valid] step {pl_module._opt_step:>4}/{self.total_steps} | "
                  f"loss {loss:.4f} | token_accuracy {acc:.4f}", flush=True)


# ─────────────────────
# LIGHTNING DATA MODULE
# ─────────────────────

class LightningDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, train_records, val_records, cfg):
        """
        Args:
            tokenizer: HuggingFace Qwen3 tokenizer returned by FastLanguageModel.
            train_records: Training records from load_nemotron_json (train.json).
            val_records: Validation records from load_nemotron_json (test.json).
            cfg: CFG dataclass with BATCH_SIZE, VAL_BATCH_SIZE, WORKERS, SEED,
                 MAX_SEQ_LEN, INCLUDE_THINKING.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.eos_token_id
        self.train_dataset = NemotronReasoningDataset(
            train_records, self.tokenizer,
            max_seq_len=cfg.MAX_SEQ_LEN, include_thinking=cfg.INCLUDE_THINKING
        )
        self.val_dataset = NemotronReasoningDataset(
            val_records, self.tokenizer,
            max_seq_len=cfg.MAX_SEQ_LEN, include_thinking=cfg.INCLUDE_THINKING
        )
        self.collate_fn = partial(
            custom_collate_fn,
            pad_token_id=self.pad_token_id,
            ignore_index=-100,
            allowed_max_length=cfg.MAX_SEQ_LEN,
            device=torch.device("cpu")
        )
        print(f"Train samples (after seq_len filter): {len(self.train_dataset)}")
        print(f"Val samples (after seq_len filter): {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=CFG.BATCH_SIZE,
            collate_fn=self.collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=CFG.WORKERS
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=CFG.VAL_BATCH_SIZE,
            collate_fn=self.collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=CFG.WORKERS
        )


# ────────────────────────
# LIGHTNING TRAIN PIPELINE
# ────────────────────────

class Qwen3_Lightning(pl.LightningModule):
    def __init__(self, base_model, pad_token_id: int):
        super().__init__()
        self.automatic_optimization = False
        self.pad_token_id = pad_token_id

        # Gradient accumulation accumulators
        self._accum_loss = 0.0
        self._accum_tokens = 0
        self._opt_step = 0              # completed optimizer steps (manual optimization)
        self._total_tokens_trained = 0  # cumulative answer tokens with loss

        # Validation tracking
        self.val_step_losses = []
        self.val_step_accuracies = []
        self.val_step_token_counts = []

        self.base_model = base_model  # Raw Unsloth model (Qwen3MoeForCausalLM)
        self.model = None             # Set in configure_model
        self.tie_grads_fn = lambda: None


    def configure_model(self):
        # 0. Enable gradient checkpointing the Unsloth way.
        #    gradient_checkpointing_enable() only sets the flag; Unsloth's backbone
        #    forward also checks an instance-level `self.training` attribute which
        #    for_training() sets explicitly.  Without it the GC condition
        #    `self.gradient_checkpointing and self.training and not use_cache` can
        #    be False → all activations are stored → OOM.
        self.base_model.for_training()   # sets gradient_checkpointing + training attrs

        # 1. Use the HF model directly — no custom forward wrapper needed.
        #    self.model is Qwen3MoeForCausalLM:
        #      .model  → Qwen3MoeModel backbone  (Unsloth-patched forward, GC active)
        #      .lm_head → nn.Linear (→ LinearWithLoRA after apply_lora)
        self.model = self.base_model
        del self.base_model

        # 2. Apply LoRA (freeze + inject LinearWithLoRA + ExpertLoRA hooks)
        self.model = apply_lora(
            self.model,
            rank=CFG.RANK,
            alpha=CFG.ALPHA,
            target_modules=CFG.TARGET_MODULES,
        )

        # 3. Cast LoRA params to fp32 for stable training
        for _, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

        # 4. MoE weight tying (Nemotron 1st-place technique)
        if CFG.MOE_TIE_WEIGHTS:
            self.tie_grads_fn = setup_moe_tying(self.model)

        # 5. Double check parameter training
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params: {total:,}")
        print(f"Trainable LoRA params: {trainable:,} ({100 * trainable / total:.2f}%) | "
              f"VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB")


    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        inp_idx, targ_idx = batch
        attention_mask = make_attention_mask(inp_idx, self.pad_token_id)

        # 1. Forward through backbone only → hidden states (no lm_head, no logits)
        #    Unsloth's gradient checkpointing runs inside model.model.forward() correctly.
        backbone_out = self.model.model(
            input_ids=inp_idx,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = backbone_out[0]

        # 2. Cut Cross-Entropy loss (avoids materializing full logits tensor)
        #    lm_head.weight handles LoRA composition via LinearWithLoRA.weight property
        lm_weight = self.model.lm_head.weight
        loss_sum = linear_cross_entropy(
            hidden, lm_weight, targ_idx,
            ignore_index=-100, reduction='sum'
        )
        num_tokens = (targ_idx != -100).sum()

        # 3. Backward
        self.manual_backward(loss_sum)

        # 4. Track
        self._accum_loss += loss_sum.detach().item()
        self._accum_tokens += num_tokens.item()

        # 5. Optimizer step after gradient accumulation
        if ((batch_idx + 1) % CFG.GRAD_ACCUM == 0):
            self._optimizer_step(opt, sch)


    def on_train_epoch_end(self):
        # Flush any leftover accumulated gradients at epoch boundary
        if self._accum_tokens > 0:
            opt = self.optimizers()
            sch = self.lr_schedulers()
            self._optimizer_step(opt, sch, from_epoch_end=True)


    def _optimizer_step(self, opt, sch, from_epoch_end=False):
        """ Gradient Accumulation Training (Unsloth fix) + MoE tying:
            - Use reduction='sum' → raw un-normalized loss per micro-batch
            - backward(loss_sum) each micro-batch → grads accumulate
            - After G steps, divide grads by M = Σ(m_i) → correct full-batch loss
        This is mathematically identical to full-batch training.
        """
        # 0. Scale gradients by 1/M → correct full-batch normalization
        M = self._accum_tokens
        self._total_tokens_trained += M
        for p in self.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.div_(M)

        # 1. MoE weight tying: sync expert gradients before clipping
        self.tie_grads_fn()

        # 2. Gradient clipping (every step, same as HF Trainer default)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=CFG.GRAD_NORM)

        # 3. Optimize + Scheduler
        opt.step()
        opt.zero_grad()
        sch.step()
        self._opt_step += 1

        # 4. Final loss = total_loss_sum / total_tokens
        train_loss = self._accum_loss / M

        # 5. Log — loss, LR, grad-norm, throughput and memory
        on_step = not from_epoch_end
        on_epoch = from_epoch_end

        # Learning rate actually applied this step (group 0 = LoRA A / base LR)
        lr_now = opt.param_groups[0]['lr']

        gpu_mem_gb = (
            torch.cuda.max_memory_allocated() / 1024 ** 3
            if torch.cuda.is_available() else 0.0
        )

        self.log('Training/train_loss', train_loss,
            sync_dist=True, prog_bar=True, on_step=on_step, on_epoch=on_epoch)

        extra = {
            'Training/lr': lr_now,
            'Training/grad_norm': grad_norm.item(),
            'Training/gpu_mem_gb': gpu_mem_gb,
            # Cumulative answer tokens the model has actually computed loss on
            'Training/number_token_trained': float(self._total_tokens_trained),
        }
        # LoRA+ runs a second LR for the B matrices — worth tracking separately
        if CFG.LORAPLUS_RATIO is not None and len(opt.param_groups) > 1:
            extra['Training/lr_lora_B'] = opt.param_groups[1]['lr']

        self.log_dict(extra, sync_dist=True, prog_bar=False,
            on_step=on_step, on_epoch=on_epoch)

        # 6. Reset accumulators
        self._accum_loss = 0.0
        self._accum_tokens = 0


    def calc_mean_tokens_accuracy(self, hidden, targets, ignore_index=-100, chunk=1024):
        """Teacher-forced next-token accuracy -> (correct, total).

        The dense equivalent (1_Qwen3_Gsm8k_SFT.py) can argmax a whole
        [B, T, vocab] logits tensor because its sequences are 512 tokens. Here a
        sequence is up to 8192 tokens against a 151,936-token vocabulary, so the
        same tensor would be 2.4 GB in bf16 on top of a 30B base that already
        fills most of the card. The sequence is therefore projected through
        lm_head in slices and only each slice's argmax is kept — the metric is
        identical, the peak allocation is ~1/8th.
        """
        with torch.no_grad():
            lm_head = self.model.lm_head
            correct, total = 0, 0
            seq_len = hidden.shape[1]
            for start in range(0, seq_len, chunk):
                targ_chunk = targets[:, start:start + chunk]
                mask = targ_chunk != ignore_index
                n_valid = int(mask.sum().item())
                if n_valid == 0:
                    continue
                logits = lm_head(hidden[:, start:start + chunk])
                predictions = logits.argmax(dim=-1)
                correct += int(((predictions == targ_chunk) & mask).sum().item())
                total += n_valid
                del logits, predictions
        return correct, total


    def validation_step(self, batch, _batch_idx):
        inp_idx, targ_idx = batch
        attention_mask = make_attention_mask(inp_idx, self.pad_token_id)

        # Forward through backbone → hidden states
        backbone_out = self.model.model(
            input_ids=inp_idx,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = backbone_out[0]

        # Same CCE kernel as training, so the two losses stay comparable.
        loss = linear_cross_entropy(
            hidden, self.model.lm_head.weight, targ_idx,
            ignore_index=-100, reduction='mean'
        )
        self.val_step_losses.append(loss.detach().float())

        correct, num_tokens = self.calc_mean_tokens_accuracy(hidden, targ_idx)
        # Weighted by token count
        self.val_step_accuracies.append(float(correct))
        self.val_step_token_counts.append(num_tokens)
        return loss


    def on_validation_epoch_end(self):
        # Loss - skip nan loss
        stacked = torch.stack(self.val_step_losses)
        valid = stacked[~torch.isnan(stacked)]
        avg_val_loss = valid.mean() if valid.numel() > 0 else torch.tensor(0.0)
        self.log('Validation/loss', avg_val_loss, sync_dist=True, prog_bar=True)

        total_weighted = sum(self.val_step_accuracies)
        total_tokens = sum(self.val_step_token_counts)
        val_token_acc = total_weighted / total_tokens if total_tokens > 0 else 0.0
        self.log('Validation/token_accuracy', val_token_acc, sync_dist=True, prog_bar=True)

        # Clear cache
        self.val_step_losses.clear()
        self.val_step_accuracies.clear()
        self.val_step_token_counts.clear()


    def loraplus_optimizer(self):
        """LoRA+ optimizer: groupA (lora_A, base LR), groupB (lora_B, higher LR)."""
        lr = CFG.LR
        lr_ratio = CFG.LORAPLUS_RATIO

        group_A = []
        group_B = []
        group_B_nodecay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # For LinearWithLoRA: .lora.B
            # For ExpertLoRA: ._gate_up_lora.B or ._down_lora.B
            if '.lora.B' in name or ('_lora.B' in name) or param.ndim == 1:
                if param.ndim >= 2:
                    group_B.append(param)
                else:
                    group_B_nodecay.append(param)
            else:
                group_A.append(param)

        optim_groups = [
            {'params': group_A, 'lr': lr, 'weight_decay': 0.0},
            {'params': group_B, 'lr': lr * lr_ratio, 'weight_decay': 0.0},
            {'params': group_B_nodecay, 'lr': lr * lr_ratio, 'weight_decay': 0.0},
        ]
        optim_groups = [g for g in optim_groups if len(g['params']) > 0]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
        return optimizer


    def standard_optimizer(self):
        """Standard AdamW — no weight decay for LoRA params (matches working 13_Qwen3 setup)."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': 0.0},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=CFG.LR, betas=(0.9, 0.95), eps=1e-8)
        return optimizer


    def configure_optimizers(self):
        if CFG.LORAPLUS_RATIO is not None:
            optimizer = self.loraplus_optimizer()
        else:
            optimizer = self.standard_optimizer()

        def lr_lambda(current_step):
            if current_step < CFG.WARMUP_STEPS:
                return (current_step + 1) / CFG.WARMUP_STEPS
            if current_step > CFG.STEPS:
                return CFG.MIN_LR / CFG.LR
            decay_ratio = (current_step - CFG.WARMUP_STEPS) / (CFG.STEPS - CFG.WARMUP_STEPS)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return (CFG.MIN_LR + coeff * (CFG.LR - CFG.MIN_LR)) / CFG.LR

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]


# --------------------
# LIGHTNING AI LOGGING
# --------------------

def lightning_ai_ready():
    """True when remote tracking can actually run: package + credentials.

    LitLogger falls back to an interactive browser login when unauthenticated,
    which blocks a headless run forever, so the credentials are checked before
    the logger is ever constructed.

    The `litlogger` backend is imported lazily inside LitLogger.experiment, i.e.
    from Trainer.fit — long after the base weights are on the GPU. A missing
    package would therefore kill the run minutes in, so it is checked up front
    and degrades to CSV exactly like missing credentials do.
    """
    import os
    from importlib.util import find_spec

    if find_spec("litlogger") is None:
        print("[logger] `litlogger` is not installed "
              "(`pip install litlogger` enables Lightning AI tracking).")
        return False
    if os.environ.get("LIGHTNING_API_KEY") and os.environ.get("LIGHTNING_USER_ID"):
        return True
    if (Path.home() / ".lightning" / "credentials.json").is_file():
        return True
    print("[logger] No Lightning AI credentials found.")
    return False


# -----------------------------------
# ANSWER GRADING  (Correct/Wrong/Bad)
# -----------------------------------
# Same three-label scheme as 1_Qwen3_Gsm8k_SFT.py, but matching is STRING-first,
# not numeric. Nemotron answers are 8-bit strings ("00110100"), cipher phrases,
# Roman numerals and punctuation runs — float() would score a model that dropped
# two leading zeros as Correct. Numeric is only a fallback, and only where it
# cannot hide a formatting error.

THINK_CLOSE = "</think>"

BOXED_OPEN = "\\boxed{"

# A leading zero before another digit makes the string form significant, so
# numeric comparison is refused for it.
LEADING_ZERO_RE = re.compile(r"^[+-]?0\d")


def extract_boxed(text):
    """Return every \\boxed{...} payload in `text`, outermost braces balanced.

    GSM8K's regex stops at the first closing brace, returning "\\text{XLIV" for
    \\boxed{\\text{XLIV}}, which Nemotron models emit often enough to matter.
    """
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


def classify(completion, expected, thinking_mode, finish_reason):
    """Grade one completion -> (matching, prediction, thinking_model, bad_reason).

    ``matching`` is one of three labels:
        Correct  a well-formed \boxed{...} whose value equals the ground truth
        Wrong    a clean extracted answer, but not the right one
        Bad      no usable answer could be extracted at all

    ``bad_reason`` keeps a budget artefact from being read as a reasoning error:
        empty_output        model returned nothing
        think_not_closed    thinking mode, </think> never emitted
        truncated_length    hit max_new_tokens with no answer
        no_boxed_answer     finished cleanly but never wrote \boxed{...}
        empty_box           wrote \boxed{} with nothing inside
    """
    text = completion or ""

    # 1. Split the reasoning off. Only text AFTER </think> may carry the final
    #    answer — scanning the chain of thought would credit a lucky intermediate.
    #    Nemotron traces box the answer inside the reasoning too, so without
    #    this split almost everything would grade Correct.
    if thinking_mode:
        if THINK_CLOSE in text:
            thinking_model, _, answer_region = text.partition(THINK_CLOSE)
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


def build_record(rec, completion, finish_reason, thinking_mode):
    """One graded row, in the schema the GSM8K run writes plus the category."""
    matching, prediction, thinking_model, bad_reason = classify(
        completion, rec["answer"], thinking_mode, finish_reason
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

class BoxedAnswerEvalCallback(pl.Callback):
    """Solve held-out questions by generation, 5 times during training.

    Fires every ``every_pct`` of the planned optimizer steps (0.20 -> 20/40/60/
    80/100%), greedily decodes a stratified sample of the test set and grades it
    with the classify() above.

    Decoding goes through model.generate() rather than the from-scratch KVCache
    1_ uses. This class owns what is not generic: swapping the attention
    implementation, switching Unsloth out of training mode, batching by prompt
    length, surviving OOM, and grading.

    Runs from on_train_batch_end, which Lightning calls just BEFORE the
    validation loop, so the metrics reach callback_metrics in time for the
    checkpoint callbacks that monitor them on_validation_end.
    """

    # <|endoftext|> and <|im_end|> both end an assistant turn.
    STOP_TOKENS = ("<|endoftext|>", "<|im_end|>")

    def __init__(self, tokenizer, records, every_pct, max_new_tokens, total_steps,
                 batch_size, output_dir, num_samples=None, seed=CFG.SEED):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        # Hard ceiling on prompt + generation, matching the training filter.
        self.max_total_len = CFG.MAX_SEQ_LEN
        self.batch_size = batch_size
        # Where each round's graded answers land: logs/<run_name>/validation/
        self.output_dir = Path(output_dir)
        self.total_steps = max(1, total_steps)
        self.every = max(1, int(round(self.total_steps * every_pct)))
        self.thinking_mode = CFG.INCLUDE_THINKING

        self.records = self._select(records, num_samples, seed)

        ids = {tokenizer.convert_tokens_to_ids(t) for t in self.STOP_TOKENS}
        self.eos_ids = sorted(i for i in ids if isinstance(i, int) and i >= 0)
        if not self.eos_ids:
            self.eos_ids = [tokenizer.eos_token_id]
        self.pad_id = (tokenizer.pad_token_id
                       if tokenizer.pad_token_id is not None
                       else tokenizer.eos_token_id)

        self._fired = set()
        self._round_idx = 0
        self._last_metrics = None   # last successful round, for _guarded_run
        self._train_attn_impl = None  # restored after each round by _set_mode

    @staticmethod
    def _select(records, num_samples, seed):
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

    # ---------------------------------------------------------- scheduling
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = pl_module._opt_step
        if step == 0 or step % self.every != 0 or step in self._fired:
            return
        self._fired.add(step)
        self._guarded_run(trainer, pl_module, step)

    def on_train_end(self, trainer, pl_module):
        # Final round, unless a boundary already fired close to the end.
        step = pl_module._opt_step
        if step <= 0 or step in self._fired:
            return
        if self._fired and (step - max(self._fired)) < self.every // 2:
            return
        self._fired.add(step)
        self._guarded_run(trainer, pl_module, step)

    def _guarded_run(self, trainer, pl_module, step):
        """Run a round; never let an eval bug destroy the training run.

        Round 1 re-raises: if it cannot run, the checkpoint callbacks monitor a
        metric that would never exist, and failing early is cheaper.
        """
        try:
            self._run(trainer, pl_module, step)
        except Exception as e:
            import traceback
            print(f"\n  [eval] round {self._round_idx} FAILED: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()
            self._set_mode(pl_module.model, inference=False)
            pl_module.model.train()

            if self._last_metrics is None:
                print("  [eval] first round failed — re-raising.", flush=True)
                raise
            # Re-publish the previous round so ModelCheckpoint still finds its
            # monitored key and training carries on to the next boundary.
            print("  [eval] carrying forward the previous round's metrics.", flush=True)
            trainer.callback_metrics.update(
                {k: torch.tensor(v) for k, v in self._last_metrics.items()})

    # ---------------------------------------------------------- generation
    @torch.no_grad()
    def _generate_batch(self, model, prompts):
        """Greedy-decode a batch of prompts -> ([(text, finish_reason), ...], steps, budget).

        Prompts are LEFT-padded so every row's next-token slot is the last
        column and the whole batch shares one position offset; HF applies the
        attention mask so the pad prefix contributes nothing.
        """
        device = next(model.parameters()).device
        # HF defaults to right padding, which would make the model continue
        # from the pad block instead of the prompt.
        previous_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                 add_special_tokens=False)
        finally:
            self.tokenizer.padding_side = previous_side

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]

        # Never generate past what training ever produced (see INFER_MAX_NEW_TOKENS).
        max_new = max(1, min(self.max_new_tokens, self.max_total_len - prompt_len))

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=False,          # greedy, so rounds differ only by the model
            use_cache=True,
            eos_token_id=self.eos_ids,
            pad_token_id=self.pad_id,
        )

        generated = out[:, prompt_len:]
        steps = generated.shape[1]

        results = []
        for row in generated:
            ids = row.tolist()
            # No stop token means the row ran out of budget, which classify()
            # reports as truncated_length rather than a reasoning error.
            stop_at = next((i for i, t in enumerate(ids) if t in self.eos_ids), None)
            finish = "stop" if stop_at is not None else "length"
            kept = ids[:stop_at] if stop_at is not None else ids
            results.append((self.tokenizer.decode(kept, skip_special_tokens=True).strip(),
                            finish))
        return results, steps, max_new

    # Attention implementation used only while generating. Training runs on
    # whatever the model loaded with (flex_attention on this build).
    INFER_ATTN_IMPL = "eager"

    def _swap_attn(self, model, impl):
        """Point the model at a different attention implementation, if it allows it."""
        if not impl or getattr(model.config, "_attn_implementation", None) == impl:
            return
        try:
            model.set_attn_implementation(impl)
        except Exception as e:
            print(f"  [eval] could not switch attention to {impl}: "
                  f"{type(e).__name__}: {e}", flush=True)

    def _set_mode(self, model, inference):
        """Flip the model between training and generation.

        ATTENTION: training needs flex_attention (68.4 GB of 79.25 at 8k tokens;
        eager OOMs there), but flex_attention cannot generate — HF's cache-aware
        mask goes through create_block_mask, which raises ValueError on it. Eager
        is cheap for decoding and is restored on the way out.

        MODE: gradient checkpointing forces use_cache off, so generating in
        training mode would recompute the prefix for every token.
        """
        if inference:
            self._train_attn_impl = getattr(model.config, "_attn_implementation", None)
            self._swap_attn(model, self.INFER_ATTN_IMPL)
        else:
            self._swap_attn(model, self._train_attn_impl)

        fn = getattr(model, "for_inference" if inference else "for_training", None)
        if callable(fn):
            try:
                fn()
                return
            except Exception as e:
                print(f"  [eval] Unsloth mode switch failed: {type(e).__name__}: {e}")
        model.eval() if inference else model.train()

    # --------------------------------------------------------------- round
    def _run(self, trainer, pl_module, step):
        model = pl_module.model
        was_training = model.training
        self._set_mode(model, inference=True)
        model.eval()

        self._round_idx += 1
        round_idx = self._round_idx
        pct = 100.0 * step / self.total_steps
        n = len(self.records)
        bar = "=" * 78
        prompts = [format_prompt_only(r["question"], include_thinking=self.thinking_mode)
                   for r in self.records]

        # Even chunks, so an odd total leaves no tiny trailing batch that still
        # costs a full decode pass.
        total_batches = max(1, math.ceil(n / max(1, self.batch_size)))
        bs = max(1, math.ceil(n / total_batches))

        # Sort by prompt length so each batch is uniform: the per-batch
        # MAX_SEQ_LEN clamp is set by the longest prompt in it, and left-padding
        # goes to the batch maximum. Unsorted back into test order before grading.
        enc_lens = [len(self.tokenizer(p, add_special_tokens=False)["input_ids"])
                    for p in prompts]
        order = sorted(range(len(prompts)), key=lambda k: enc_lens[k])
        sorted_prompts = [prompts[k] for k in order]
        lo, hi = enc_lens[order[0]], enc_lens[order[-1]]
        print(f"\n{bar}\n"
              f"GENERATIVE EVAL  round {round_idx}  |  step {step}/{self.total_steps} "
              f"({pct:.0f}%)  |  {n} questions  |  batch={bs}  |  prompts {lo}-{hi} tok  "
              f"|  max_new<={self.max_new_tokens}  |  thinking={self.thinking_mode}\n"
              f"{bar}", flush=True)

        # One line per batch: a live bar redraws several times a second and
        # buries the log under thousands of near-identical lines.
        # Release the allocator's cached blocks so the KV cache can use them.
        torch.cuda.empty_cache()

        completions, i, n_batches = [], 0, 0
        started = time.time()
        while i < len(prompts):
            chunk = sorted_prompts[i:i + bs]
            t_batch = time.time()
            try:
                out, steps, budget = self._generate_batch(model, chunk)
            except torch.cuda.OutOfMemoryError:
                # Training state is still resident; halve and retry, do not die.
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                total_batches = max(1, math.ceil(len(prompts) / bs))
                print(f"  [eval] CUDA OOM -> retrying at batch_size={bs}", flush=True)
                continue
            completions.extend(out)
            i += len(chunk)
            n_batches += 1
            dt = time.time() - t_batch
            rate = i / max(1e-9, time.time() - started)
            stopped = sum(1 for _, f in out if f == "stop")
            print(f"  [eval] batch {n_batches}/{total_batches}  {i}/{len(prompts)} q  "
                  f"|  {steps}/{budget} tok  |  {stopped}/{len(chunk)} finished  "
                  f"|  {dt:.0f}s  |  eta {(len(prompts) - i) / max(1e-9, rate) / 60:.1f} min",
                  flush=True)
        elapsed = time.time() - started
        torch.cuda.empty_cache()

        # Undo the length sort so row i lines up with self.records[i] again.
        in_order = [None] * len(prompts)
        for pos, k in enumerate(order):
            in_order[k] = completions[pos]

        rows = [build_record(rec, text, finish, self.thinking_mode)
                for rec, (text, finish) in zip(self.records, in_order)]

        summary = summarize(f"round{round_idx}", rows, elapsed)
        self._report(summary)
        self._save_rows(round_idx, rows)
        self._log_metrics(trainer, pl_module, summary)

        self._set_mode(model, inference=False)
        if was_training:
            model.train()

    # -------------------------------------------------------------- output
    def _report(self, summary):
        print(f"  [eval] {summary['Correct']} Correct / {summary['Wrong']} Wrong / "
              f"{summary['Bad']} Bad  ->  {summary['accuracy']:.1f}%  "
              f"in {summary['elapsed_sec'] / 60:.1f} min")
        if summary["bad_reasons"]:
            detail = "  ".join(f"{k}={v}" for k, v in sorted(summary["bad_reasons"].items()))
            print(f"  [eval] bad reasons: {detail}")
        for category, bucket in sorted(summary["per_category"].items()):
            print(f"  [eval]   {category:<26} {bucket['correct']}/{bucket['n']} "
                  f"= {bucket['accuracy']:.0f}%")

    def _save_rows(self, round_idx, rows):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.output_dir / f"round{round_idx}.json"
            out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            print(f"  [eval] graded answers -> {out_path}", flush=True)
        except Exception as e:
            print(f"  [eval] could not write answers: {type(e).__name__}: {e}")

    def _log_metrics(self, trainer, pl_module, summary):
        metrics = {
            'Validation/box_answer_match_accuracy': float(summary['accuracy']),
            'Validation/correct_count': float(summary['Correct']),
            'Validation/wrong_count': float(summary['Wrong']),
            'Validation/bad_count': float(summary['Bad']),
        }
        for category, bucket in summary['per_category'].items():
            metrics[f'Validation/acc_{category}'] = float(bucket['accuracy'])

        for name, value in metrics.items():
            try:
                pl_module.log(name, value, logger=False, sync_dist=True,
                              on_step=True, on_epoch=False,
                              prog_bar=name.endswith('box_answer_match_accuracy'))
            except Exception:
                pass
        trainer.callback_metrics.update(
            {k: torch.tensor(v) for k, v in metrics.items()})
        self._last_metrics = metrics
        for lg in trainer.loggers:
            try:
                lg.log_metrics(metrics, step=trainer.global_step)
            except Exception as e:
                print(f"  [eval] {type(lg).__name__} rejected metrics: {e}")


# ───────────────────
# MAIN
# ───────────────────

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    # ---- GPU Info
    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, sm_{cc[0] * 10 + cc[1]}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"torch={torch.__version__}, cuda={torch.version.cuda}")

    # ---- Data Preparation
    # train.json / test.json are a question-grouped, category-stratified 95/5
    # split, fixed on disk so every run scores the same held-out questions.
    DATA_PATH = PROJECT_ROOT / "data" / "NemotronReasoning2026"
    train_records = load_nemotron_json(str(DATA_PATH / "train.json"))
    val_records = load_nemotron_json(str(DATA_PATH / "test.json"))
    print(f"Train records: {len(train_records)}, Val records: {len(val_records)}")

    CFG.STEPS = (len(train_records) // CFG.BATCH_SIZE // CFG.GRAD_ACCUM) * CFG.EPOCHS
    print(f"Optimizer steps per epoch: {CFG.STEPS}")

    # Validate at the same marks as the generative eval rounds.
    CFG.VAL_EVERY_N_STEPS = CFG.GRAD_ACCUM * max(1, round(CFG.STEPS * CFG.INFER_EVERY_PERCENTAGE))
    print(f"Validation every {CFG.VAL_EVERY_N_STEPS} step "
          f"({CFG.INFER_EVERY_PERCENTAGE:.0%} of training)")

    # ---- Model & Tokenizer (Unsloth loading)
    # NOTE: "lora" keeps the 30B base in bf16 = ~60GB VRAM, "qlora" quantizes it.
    if CFG.TUNING_MODE == "lora":
        load_in_4bit, load_in_8bit = False, False
    elif CFG.TUNING_MODE == "qlora":
        if CFG.QUANT_BITS not in ("4bit", "8bit"):
            raise ValueError(f"Unknown QUANT_BITS: {CFG.QUANT_BITS}. Use '4bit' or '8bit'.")
        load_in_4bit = CFG.QUANT_BITS == "4bit"
        load_in_8bit = CFG.QUANT_BITS == "8bit"
    else:
        raise ValueError(f"Unknown TUNING_MODE: {CFG.TUNING_MODE}. Use 'lora' or 'qlora'.")

    custom_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG.MODEL_PATH,
        max_seq_length=CFG.MAX_SEQ_LEN,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        full_finetuning=False,
        trust_remote_code=True,
        # Deliberately unpinned: training needs flex_attention (68.4 GB of 79.25
        # at 8k tokens; eager OOMs on the longest sample, and sdpa falls back to
        # eager here). The eval callback swaps to eager per round instead.
        dtype=torch.bfloat16,
    )

    pad_token_id = tokenizer.eos_token_id

    # ---- Lightning Module (LoRA + checkpointing + tying done in configure_model)
    qwen3_module = Qwen3_Lightning(base_model=custom_model, pad_token_id=pad_token_id)

    # ---- DataModule
    datamodule = LightningDataModule(
        tokenizer=tokenizer,
        train_records=train_records,
        val_records=val_records,
        cfg=CFG
    )

    # ---- Lit-Logger
    from datetime import datetime
    date = datetime.now().strftime("%d_%m_%y")
    mode_tag = "LoRA" if CFG.TUNING_MODE == "lora" else f"QLoRA_{CFG.QUANT_BITS}"
    model_tag = CFG.MODEL_PATH.split('/')[-1].replace('Qwen3-', '')
    think_suffix = "_think" if CFG.INCLUDE_THINKING else ""

    exp_name = f'Qwen3_MoE_NemotronReasoning_{model_tag}_{mode_tag}{think_suffix}'
    csv_logger = pl.loggers.CSVLogger(save_dir=str(Path('./logs') / exp_name), name=date)
    loggers = [csv_logger]

    RUN_DIR = Path(csv_logger.log_dir)          # logs/<exp_name>/<date>/version_N
    save_dir = str(RUN_DIR)
    VALIDATION_DIR = RUN_DIR / 'validation'

    run_name = f'{exp_name}_{date}_v{csv_logger.version}'
    print(f"[logger] run directory: {RUN_DIR}")
    print(f"[logger] experiment name: {run_name}")
    print(f"[eval] per-round answers -> {VALIDATION_DIR}/round<N>.json")

    # Lightning AI remote tracking (needs `lightning login` or LIGHTNING_API_KEY
    # + LIGHTNING_USER_ID). Falls back to CSV-only so a missing login never
    # blocks a long training run.
    if not lightning_ai_ready():
        print(f"[logger] remote tracking OFF (no credentials) — CSV only: {save_dir}")
        print("[logger] run `lightning login` to enable it.")
    else:
        lit_logger = pl.loggers.LitLogger(
            name=run_name,
            teamspace=CFG.TEAMSPACE,
            root_dir=save_dir,
            # save_logs=True makes litlogger re-exec the script in a PTY; the
            # parent then blocks holding its GPU memory. Two 30B models do not fit.
            save_logs=False,
            metadata={
                'model': CFG.MODEL_PATH,
                'dataset': 'nemotron_reasoning_2026',
                'tuning_mode': CFG.TUNING_MODE,
                'quant_bits': str(CFG.QUANT_BITS),
                'rank': str(CFG.RANK),
                'alpha': str(CFG.ALPHA),
                'lr': str(CFG.LR),
                'grad_accum': str(CFG.GRAD_ACCUM),
                'max_seq_len': str(CFG.MAX_SEQ_LEN),
                'include_thinking': str(CFG.INCLUDE_THINKING),
                'moe_tie_weights': str(CFG.MOE_TIE_WEIGHTS),
            },
        )
        lit_logger._version = str(csv_logger.version)
        # Append, never insert: trainer.logger is loggers[0] and decides where
        # the checkpoint callbacks write. CSVLogger must stay first.
        loggers.append(lit_logger)
        print(f"[logger]   teamspace: {CFG.TEAMSPACE}")
        print(f"[logger]   experiment: {run_name}")

    # ---- Checkpoint callback
    CKPT_MONITOR = 'Validation/box_answer_match_accuracy'
    CKPT_NAME = '{epoch:02d}-{Validation/loss:.4f}-{Validation/box_answer_match_accuracy:.2f}'
    ckpt = pl.callbacks.ModelCheckpoint(
        monitor=CKPT_MONITOR,
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        filename=CKPT_NAME,
        # '/' in a metric name would become a directory separator when Lightning
        # auto-inserts "name=value" into the filename, so insert values only.
        auto_insert_metric_name=False,
        mode='max',
        save_on_train_epoch_end=False,
        dirpath=str(RUN_DIR / 'checkpoints'),
    )

    # ---- Merged model checkpoint (sharded safetensors for vLLM inference)
    #      Config is read from the live model at save time — no HuggingFace download.
    merge_ckpt = MoELoRAMergeCheckpoint(
        tokenizer=tokenizer,
        monitor=CKPT_MONITOR,
        mode='max',
        save_top_k=1,
        # Required: the default template interpolates {Validation/accuracy},
        # which this script does not log, and the KeyError would abort the run
        # before ModelCheckpoint (plain Callbacks run first) writes anything.
        filename_template=CKPT_NAME,
    )

    # ---- Generative evaluation on the held-out test set, every 20% of training
    boxed_eval = BoxedAnswerEvalCallback(
        tokenizer=tokenizer,
        records=val_records,
        num_samples=CFG.INFER_SAMPLES,       # None -> all 404 held-out questions
        every_pct=CFG.INFER_EVERY_PERCENTAGE,
        max_new_tokens=CFG.INFER_MAX_NEW_TOKENS,
        total_steps=CFG.STEPS,
        batch_size=CFG.INFER_BATCH_SIZE,
        output_dir=VALIDATION_DIR,
    )

    # ---- Trainer
    # ---- Terminal progress that survives being piped to a file
    progress = TrainingProgressLog(total_steps=CFG.STEPS)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        callbacks=[ckpt, merge_ckpt, boxed_eval, progress],
        logger=loggers,
        max_epochs=CFG.EPOCHS,
        precision='bf16',
        val_check_interval=CFG.VAL_EVERY_N_STEPS,  # held-out set every 20%
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=qwen3_module, datamodule=datamodule)

    # ---- Plot Results
    plot_training_curves(csv_logger)
