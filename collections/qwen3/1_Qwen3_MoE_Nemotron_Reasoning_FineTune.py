"""
Qwen3-30B-A3B (MoE) Reasoning Fine-tuning: From-Scratch LoRA + Unsloth + Lightning
────────────────────────────────────────────────────────────────────────────────
Architecture:
- Unsloth FastLanguageModel backbone + custom LoRA + MoE tying + Lightning
- CFG.TUNING_MODE switches between LoRA (bf16 base) and QLoRA (4bit/8bit base)
- lm_head in TARGET_MODULES (LoRA on output projection)
- Dataset: NemotronReasoning2026 (math reasoning with thinking chains)
- Loss: Only on assistant response (prompt masked), using Cut Cross-Entropy

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
import sys
import math
import random
from pathlib import Path
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from cut_cross_entropy import linear_cross_entropy
torch.set_float32_matmul_precision('high')

from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLLECTIONS_DIR = PROJECT_ROOT / "collections"
MODELS_DIR = Path(__file__).resolve().parent / "models"
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(COLLECTIONS_DIR))
sys.path.append(str(MODELS_DIR))

from data.NemotronReasoning2026.data_utils import (
    load_nemotron_csv, NemotronReasoningDataset, custom_collate_fn
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
    MAX_SEQ_LEN = 8192

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

    # 3. Apply ExpertLoRA to 3D packed expert params (gate_up_proj, down_proj)
    _apply_expert_lora(model, rank, alpha)

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


# ─────────────────────
# LIGHTNING DATA MODULE
# ─────────────────────

class LightningDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, train_records, val_records, cfg):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.eos_token_id
        self.train_dataset = NemotronReasoningDataset(
            train_records, self.tokenizer, max_seq_len=cfg.MAX_SEQ_LEN
        )
        self.val_dataset = NemotronReasoningDataset(
            val_records, self.tokenizer, max_seq_len=cfg.MAX_SEQ_LEN
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
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable params (fp32): {trainable:,}")
        print(f"VRAM after setup: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


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
        if self._accum_tokens > 0:
            opt = self.optimizers()
            sch = self.lr_schedulers()
            self._optimizer_step(opt, sch, from_epoch_end=True)


    def _optimizer_step(self, opt, sch, from_epoch_end=False):
        """Gradient Accumulation + MoE Tying."""
        M = self._accum_tokens
        for p in self.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.div_(M)

        # MoE weight tying: sync expert gradients
        self.tie_grads_fn()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=CFG.GRAD_NORM)

        # Optimize + Scheduler
        opt.step()
        opt.zero_grad()
        sch.step()

        # Compute loss
        train_loss = self._accum_loss / M

        # Log
        if from_epoch_end:
            self.log('Training/train_loss', train_loss,
                     sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
            self.log('Training/grad_norm', grad_norm.item(),
                     sync_dist=True, prog_bar=False, on_step=False, on_epoch=True)
        else:
            self.log('Training/train_loss', train_loss,
                     sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)
            self.log('Training/grad_norm', grad_norm.item(),
                     sync_dist=True, prog_bar=False, on_step=True, on_epoch=False)

        # Reset accumulators
        self._accum_loss = 0.0
        self._accum_tokens = 0


    def calc_mean_tokens_accuracy(self, logits, targets, ignore_index=-100):
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            mask = targets != ignore_index
            correct = ((predictions == targets) & mask).sum()
            total = mask.sum()
            accuracy = (correct / total).item() if total > 0 else 0.0
        return accuracy


    def validation_step(self, batch, _batch_idx):
        inp_idx, targ_idx = batch
        attention_mask = make_attention_mask(inp_idx, self.pad_token_id)

        # Forward through backbone → hidden, then lm_head for full logits
        backbone_out = self.model.model(
            input_ids=inp_idx,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = backbone_out[0]
        logits = self.model.lm_head(hidden)

        loss = F.cross_entropy(logits.flatten(0, 1), targ_idx.flatten(), ignore_index=-100)
        self.val_step_losses.append(loss)

        acc = self.calc_mean_tokens_accuracy(logits, targ_idx)
        num_tokens = (targ_idx != -100).sum().item()
        self.val_step_accuracies.append(acc * num_tokens)
        self.val_step_token_counts.append(num_tokens)
        return loss


    def on_validation_epoch_end(self):
        stacked = torch.stack(self.val_step_losses)
        valid = stacked[~torch.isnan(stacked)]
        avg_val_loss = valid.mean() if valid.numel() > 0 else torch.tensor(0.0)
        self.log('Validation/loss', avg_val_loss, sync_dist=True, prog_bar=True)

        total_weighted = sum(self.val_step_accuracies)
        total_tokens = sum(self.val_step_token_counts)
        val_token_acc = total_weighted / total_tokens if total_tokens > 0 else 0.0
        self.log('Validation/accuracy', val_token_acc, sync_dist=True, prog_bar=True)

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


# ───────────────────
# MAIN
# ───────────────────

# --------------------
# LIGHTNING AI LOGGING
# --------------------

def lightning_ai_ready():
    """True when lightning.ai credentials exist.

    LitLogger falls back to an interactive browser login when unauthenticated,
    which blocks a headless run forever, so the credentials are checked before
    the logger is ever constructed.
    """
    import os
    if os.environ.get("LIGHTNING_API_KEY") and os.environ.get("LIGHTNING_USER_ID"):
        return True
    return (Path.home() / ".lightning" / "credentials.json").is_file()


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    # ---- GPU Info
    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, sm_{cc[0] * 10 + cc[1]}")
    print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Data Preparation
    records = load_nemotron_csv(str(PROJECT_ROOT / "data" / "NemotronReasoning2026" / "nemotron_decoded_clean.csv"))
    print(f"Total records loaded: {len(records)}")

    # Train/Val split (deterministic shuffle)
    random.seed(CFG.SEED)
    random.shuffle(records)
    n_val = max(1, int(len(records) * 0.05))
    val_records = records[:n_val]
    train_records = records[n_val:]
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    CFG.STEPS = (len(train_records) // CFG.BATCH_SIZE // CFG.GRAD_ACCUM) * CFG.EPOCHS
    print(f"Optimizer steps per epoch: {CFG.STEPS}")

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

    # ---- Logger
    from datetime import datetime
    date = datetime.now().strftime("%d_%m_%Y")
    mode_tag = "LoRA" if CFG.TUNING_MODE == "lora" else f"QLoRA_{CFG.QUANT_BITS}"
    model_tag = CFG.MODEL_PATH.split('/')[-1].replace('Qwen3-', '')
    save_dir = f'./logs/MoE_Qwen3_30B_A3B_Reasoning_{mode_tag}_r{CFG.RANK}_a{CFG.ALPHA}/'

    # Experiment name: Qwen3_<dataset>_<size>_<LoRA|QLoRA_bits>_<dd_mm_yyyy>
    run_name = f'Qwen3_NemotronReasoning_{model_tag}_{mode_tag}_{date}'

    # CSV stays as the local record — plot_training_curves() reads its metrics.csv
    csv_logger = pl.loggers.CSVLogger(save_dir=save_dir, name=f'{date}')
    loggers = [csv_logger]

    # Lightning AI remote tracking (needs `lightning login` or LIGHTNING_API_KEY
    # + LIGHTNING_USER_ID). Falls back to CSV-only so a missing login never
    # blocks a long training run.
    if not lightning_ai_ready():
        print("=" * 78)
        print("[logger] Lightning AI: no credentials found — remote tracking is OFF.")
        print("[logger] Logging locally to CSV instead:")
        print(f"[logger]   {save_dir}")
        print("[logger] To turn remote tracking on, run `lightning login` (or set")
        print("[logger] LIGHTNING_API_KEY + LIGHTNING_USER_ID), then restart training.")
        print("=" * 78)
    else:
        lit_logger = pl.loggers.LitLogger(
            name=run_name,
            teamspace=CFG.TEAMSPACE,
            # Lightning's wrapper defaults save_logs=True, which makes litlogger
            # re-exec this whole script inside a PTY to capture terminal output
            # (litlogger/experiment.py:112). The parent then blocks holding its
            # GPU memory while the child trains — two models do not fit, so
            # console capture stays off.
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
                'moe_tie_weights': str(CFG.MOE_TIE_WEIGHTS),
            },
        )
        # Append, never insert: trainer.logger is loggers[0] and decides where
        # the checkpoint callbacks write. CSVLogger must stay first.
        loggers.append(lit_logger)
        print(f"[logger] Lightning AI tracking enabled")
        print(f"[logger]   teamspace: {CFG.TEAMSPACE}")
        print(f"[logger]   experiment: {run_name}")

    # ---- Checkpoint callback
    ckpt = pl.callbacks.ModelCheckpoint(
        monitor='Validation/accuracy',
        save_top_k=3,
        save_last=True,
        save_weights_only=True,
        filename='{epoch:02d}-{Validation/loss:.4f}-{Validation/accuracy:.4f}',
        # '/' in a metric name would become a directory separator when Lightning
        # auto-inserts "name=value" into the filename, so insert values only.
        auto_insert_metric_name=False,
        mode='max',
        save_on_train_epoch_end=True,
    )

    # ---- Merged model checkpoint (sharded safetensors for vLLM inference)
    #      Config is read from the live model at save time — no HuggingFace download.
    merge_ckpt = MoELoRAMergeCheckpoint(
        tokenizer=tokenizer,
        monitor='Validation/accuracy',
        mode='max',
        save_top_k=1,
    )

    # ---- Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        callbacks=[ckpt, merge_ckpt],
        logger=loggers,
        max_epochs=CFG.EPOCHS,
        precision='bf16',
        check_val_every_n_epoch=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=qwen3_module, datamodule=datamodule)

    # ---- Plot Results
    plot_training_curves(csv_logger)

