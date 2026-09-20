"""
Qwen3-30B-A3B (MoE) GSM8K SFT: From-Scratch LoRA + Unsloth + Lightning
────────────────────────────────────────────────────────────────────────────────
3_Qwen3_MoE_Nemotron_SFT.py's model and training recipe on 1_Qwen3_Gsm8k_SFT.py's
data:

- Model:   Unsloth FastLanguageModel backbone (no from-scratch 30B MoE here), so
           generation goes through HuggingFace generate() rather than a KVCache
- LoRA:    2D on nn.Linear + 3D on the packed expert tensors, with MoE tying
- Loss:    Cut Cross-Entropy on the assistant response only (prompt masked)
- Data:    data/Gsm8k train.json / test.json, \\boxed{} chat template
- Grading: identical to 1_, so accuracy is comparable with the 14B dense run

GSM8K is short, so MAX_SEQ_LEN is 512 (not 8192), eval scores all 1319 test
questions per round, and INFER_BATCH_SIZE is 64.
"""

# Unsloth's compiled cache defaults to a RELATIVE path, so it lands in whatever
# directory the process was launched from. Pin it next to this file instead.
# Must be set BEFORE importing unsloth — compiler.py reads it at import time.
import os as _os
from pathlib import Path as _Path
_os.environ.setdefault(
    "UNSLOTH_COMPILE_LOCATION",
    str(_Path(__file__).resolve().parent / "unsloth_compiled_cache"),
)

# Must precede torch / pytorch_lightning / cut_cross_entropy: Unsloth patches
# transformers on import, and those pull transformers in transitively.
from unsloth import FastLanguageModel

import gc
import sys
import math
import time
from pathlib import Path
from functools import partial
from dataclasses import dataclass

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

from data.Gsm8k.data_utils import (
    load_gsm8k_json, Gsm8kDataset, custom_collate_fn
)
# Grading + the periodic generative eval live with the dataset, so this run is
# scored by exactly the code that scores the 14B dense run — which is the only
# reason the two accuracy numbers are comparable at all.
from data.Gsm8k.metric import Gsm8kEvalCallback, ACCURACY_KEY
from utils.checkpoint_utils import plot_training_curves
from utils.checkpoint_moe_utils import MoELoRAMergeCheckpoint
from utils.generation_utils import hf_generate_batch, UnslothEvalMixin


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
    MAX_SEQ_LEN = 512     # CoT samples reach 476 tok (train) / 434 tok (test), so nothing is truncated

    INCLUDE_THINKING = True   # False: predict \boxed{answer} directly | True: train on the chain of thought

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
    WARMUP_STEPS = 20      # ~4% of total optimizer steps for linear warmup
    LORAPLUS_RATIO = None  # Set e.g. 4.0 to enable LoRA+ (None to disable)

    # --- Experiment tracking (Lightning AI) ---
    TEAMSPACE = "LLM-From-Scratch"   # lightning.ai teamspace holding the experiments

    # --- Terminal progress ---
    # Lightning's default progress bar here is RichProgressBar, which redraws in
    # place and reports is_terminal=False the moment stdout is a pipe. A run
    # started as `python 2_...py | tee run.log` therefore shows no progress at
    # all. TrainingProgressLog prints a plain line every N optimizer steps,
    # which survives pipes, tmux capture and nohup alike.
    PROGRESS_EVERY_N_STEPS = 5

    # --- Periodic full-test-set generative evaluation ---
    INFER_EVERY_PERCENTAGE = 0.20     # evaluate every 20% of training
    INFER_SAMPLES = None       # None scores all 1319 held-out test questions

    # ~50 MB of KV cache per 512-token sequence, so 64 is ~3.2 GB beside the
    # resident training state. Halved on CUDA OOM by the eval callback.
    INFER_BATCH_SIZE = 64

    # Bounded by the training filter; hf_generate_batch clamps again per batch.
    INFER_MAX_NEW_TOKENS = MAX_SEQ_LEN - 128 if INCLUDE_THINKING else 64

    # Turn-enders BEYOND the tokenizer's own eos_token_id, which the eval
    # callback picks up on its own. Qwen3's chat format ends an assistant turn
    # with <|im_end|> but also honours <|endoftext|>.
    INFER_STOP_TOKENS = ("<|endoftext|>", "<|im_end|>")

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
        """Effective weight [out, in] = base + LoRA delta, in base dtype.

        Read directly by the CCE loss on lm_head, keeping LoRA in the graph.
        """
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
            train_records: Training records from load_gsm8k_json (train.json).
            val_records: Validation records from load_gsm8k_json (test.json).
            cfg: CFG dataclass with BATCH_SIZE, VAL_BATCH_SIZE, WORKERS, SEED,
                 MAX_SEQ_LEN, INCLUDE_THINKING.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.eos_token_id
        self.train_dataset = Gsm8kDataset(
            train_records, self.tokenizer,
            max_seq_len=cfg.MAX_SEQ_LEN, include_thinking=cfg.INCLUDE_THINKING
        )
        self.val_dataset = Gsm8kDataset(
            val_records, self.tokenizer,
            max_seq_len=cfg.MAX_SEQ_LEN, include_thinking=cfg.INCLUDE_THINKING
        )
        print(f"Train samples (after seq_len filter): {len(self.train_dataset)}")
        print(f"Val samples (after seq_len filter): {len(self.val_dataset)}")
        self.collate_fn = partial(
            custom_collate_fn,
            pad_token_id=self.pad_token_id,
            ignore_index=-100,
            allowed_max_length=cfg.MAX_SEQ_LEN,
            device=torch.device("cpu")
        )

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
        #    for_training() sets explicitly.
        self.base_model.for_training()

        # 1. Use the HF model directly — no custom forward wrapper needed.
        #      .model   → Qwen3MoeModel backbone (Unsloth-patched, GC active)
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
        if CFG.LORAPLUS_RATIO is not None and len(opt.param_groups) > 1:
            extra['Training/lr_lora_B'] = opt.param_groups[1]['lr']

        self.log_dict(extra, sync_dist=True, prog_bar=False,
            on_step=on_step, on_epoch=on_epoch)

        # 6. Reset accumulators
        self._accum_loss = 0.0
        self._accum_tokens = 0

    def calc_mean_tokens_accuracy(self, hidden, targets, ignore_index=-100, chunk=1024):
        """Teacher-forced next-token accuracy -> (correct, total).

        Projects through lm_head in slices so [B, T, vocab] is never
        materialised whole (2.4 GB at 8k tokens against a 151,936 vocab).
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

        self.val_step_losses.clear()
        self.val_step_accuracies.clear()
        self.val_step_token_counts.clear()

    def loraplus_optimizer(self):
        """LoRA+ optimizer: groupA (lora_A, base LR), groupB (lora_B, higher LR)."""
        lr = CFG.LR
        lr_ratio = CFG.LORAPLUS_RATIO

        group_A, group_B, group_B_nodecay = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # For LinearWithLoRA: .lora.B  |  For ExpertLoRA: ._gate_up_lora.B / ._down_lora.B
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
        return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)

    def standard_optimizer(self):
        """Standard AdamW — no weight decay for LoRA params."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': 0.0},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(optim_groups, lr=CFG.LR, betas=(0.9, 0.95), eps=1e-8)

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


# ────────────────────────────────────────────
#  GENERATIVE EVALUATION (GSM8K, HF BACKBONE)
# ────────────────────────────────────────────

class MoEEvalCallback(UnslothEvalMixin, Gsm8kEvalCallback):
    """The shared GSM8K eval, driven through HuggingFace generate().

    Grading, scheduling, batching and logging are inherited unchanged from
    data/Gsm8k/metric.py — the only differences here are the backbone's, and
    both come from utils/generation_utils.py:

        UnslothEvalMixin   swaps flex_attention out for eager and flips Unsloth
                           between for_training() and for_inference()
        hf_generate_batch  passed as generate_fn below, since an HF model has
                           no generate_batch of its own

    Nothing GSM8K-specific is overridden, so this class stays empty on purpose.
    """


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
    DATA_PATH = PROJECT_ROOT / "data" / "Gsm8k"
    train_records = load_gsm8k_json(str(DATA_PATH / "train.json"))
    val_records = load_gsm8k_json(str(DATA_PATH / "test.json"))
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
        # Attention left unpinned; the eval callback swaps to eager per round.
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

    exp_name = f'Qwen3_MoE_Gsm8k_{model_tag}_{mode_tag}{think_suffix}'
    csv_logger = pl.loggers.CSVLogger(save_dir=str(Path('./logs') / exp_name), name=date)
    loggers = [csv_logger]

    RUN_DIR = Path(csv_logger.log_dir)          # logs/<exp_name>/<date>/version_N
    save_dir = str(RUN_DIR)
    VALIDATION_DIR = RUN_DIR / 'validation'

    run_name = f'{exp_name}_{date}_v{csv_logger.version}'
    print(f"[logger] run directory: {RUN_DIR}")
    print(f"[logger] experiment name: {run_name}")
    print(f"[eval] per-round answers -> {VALIDATION_DIR}/round<N>.json")

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
                'dataset': 'gsm8k',
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
    CKPT_MONITOR = ACCURACY_KEY                  # published by MoEEvalCallback
    CKPT_NAME = '{epoch:02d}-{Validation/loss:.4f}-{' + ACCURACY_KEY + ':.2f}'
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

    # ---- Generative evaluation on the FULL test set, every 20% of training
    boxed_eval = MoEEvalCallback(
        tokenizer=tokenizer,
        records=val_records,
        total_steps=CFG.STEPS,
        output_dir=VALIDATION_DIR,
        every_pct=CFG.INFER_EVERY_PERCENTAGE,
        num_samples=CFG.INFER_SAMPLES,       # None -> all 1319 questions
        batch_size=CFG.INFER_BATCH_SIZE,
        max_new_tokens=CFG.INFER_MAX_NEW_TOKENS,
        max_total_len=CFG.MAX_SEQ_LEN,
        thinking_mode=CFG.INCLUDE_THINKING,
        stop_tokens=CFG.INFER_STOP_TOKENS,
        seed=CFG.SEED,
        generate_fn=hf_generate_batch,       # HF backbone has no generate_batch
    )

    # ---- Terminal progress that survives being piped to a file
    progress = TrainingProgressLog(total_steps=CFG.STEPS)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        callbacks=[ckpt, merge_ckpt, boxed_eval, progress],
        logger=loggers,
        max_epochs=CFG.EPOCHS,
        precision='bf16',
        val_check_interval=CFG.VAL_EVERY_N_STEPS,  # full test set every 20%
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=qwen3_module, datamodule=datamodule)

    # ---- Plot Results
    plot_training_curves(csv_logger)
