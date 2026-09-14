import sys
import math
import random
import re
import bitsandbytes as bnb
from pathlib import Path
from functools import partial
from dataclasses import dataclass

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('high')  # Use TF32 on A100 for faster matmul

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLLECTIONS_DIR = PROJECT_ROOT / "collections"
MODELS_DIR = Path(__file__).resolve().parent / "models"
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(COLLECTIONS_DIR))
sys.path.append(str(MODELS_DIR))

from data.Gsm8k.data_utils import (
    load_gsm8k_json, Gsm8kDataset, custom_collate_fn, format_prompt_only
)
from utils.checkpoint_utils import (
    LoRAMergeCheckpoint, plot_training_curves,
)

from qwen3_dense import (
    Qwen3Model, from_pretrained,
    QWEN_06B_CFG, QWEN_1B7_CFG, QWEN_4B_CFG,
    QWEN_8B_CFG, QWEN_14B_CFG, QWEN_32B_CFG
)
from qwen_tokenizer import Qwen3Tokenizer


# ------
# CONFIG
# ------
QWEN3_MODELS = {
    "0.6B": (QWEN_06B_CFG, "Qwen/Qwen3-0.6B"),
    "1.7B": (QWEN_1B7_CFG, "Qwen/Qwen3-1.7B"),
    "4B":   (QWEN_4B_CFG,  "Qwen/Qwen3-4B"),
    "8B":   (QWEN_8B_CFG,  "Qwen/Qwen3-8B"),
    "14B":  (QWEN_14B_CFG, "Qwen/Qwen3-14B"),
    "32B":  (QWEN_32B_CFG, "Qwen/Qwen3-32B"),
}


@dataclass
class CFG:
    EPOCHS = 1
    WORKERS = 2
    BATCH_SIZE = 1
    VAL_BATCH_SIZE = 1
    GRAD_ACCUM = 16
    MAX_SEQ_LEN = 512     # CoT samples reach 476 tok (train) / 434 tok (test), so nothing is truncated

    MODEL_SIZE = "14B"    # "0.6B" | "1.7B" | "4B" | "8B" | "14B" | "32B"
    INCLUDE_THINKING = True   # False: predict \boxed{answer} directly | True: train on the chain of thought

    RANK = 16
    ALPHA = 32
    TUNING_MODE = "qlora" # "lora": lora mode | "qlora": quantized lora mode
    QUANT_BITS = "4bit"   # "4bit" or "8bit", only used with TUNING_MODE = "qlora"
    LORAPLUS_RATIO = None # 4.0  # LR_B = LR_A*ratio, If use LoRA+, set base lr smaller(like 7.5e-5) or decrease ratio (4.0 or 8.0) (None to disable LoRA+)

    SEED = 1001
    LR = 1.5e-4
    MIN_LR = LR * 0.1
    GRAD_NORM = 1.0
    WARMUP_STEPS = 20      # ~10% of total optimizer steps for linear warmup

    # --- Experiment tracking (Lightning AI) ---
    TEAMSPACE = "LLM-From-Scratch"   # lightning.ai teamspace holding the experiments

    # --- Periodic sample inference ---
    INFER_EVERY_PCT = 0.20    # sample the model every 20% of optimizer steps
    INFER_SAMPLES = 10        # held-out questions to print per round
    # CoT + \\boxed{} ground truths run to 346 tok (p99 = 221); generation has no
    # KV cache, so every extra token costs a full forward pass over the sequence.
    INFER_MAX_NEW_TOKENS = 320 if INCLUDE_THINKING else 48


# -----------------
# LOW-RANK ADAPTION
# -----------------

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.scaling = alpha / rank  # Standard LoRA scaling: α/r

    def forward(self, x):
        x = self.scaling * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            self.linear.in_features, self.linear.out_features, rank, alpha
        )
    def forward(self, x):
        return self.linear(x) + self.lora(x)

def apply_lora(model, rank, alpha, target_modules):
    """ Apply LoRA (Low-Rank Adaptation) to Linear Layer of Target Modules listed.
    Returns:
        model: The modified model with LoRA layers injected.
    """
    def replace_linear_with_lora(model, rank, alpha, target_modules=None):
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                if target_modules is None or name in target_modules:
                    setattr(model, name, LinearWithLoRA(module, rank, alpha))
            else:
                replace_linear_with_lora(module, rank, alpha, target_modules)

    # 1. Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # 2. Add LoRA layers to target layers
    replace_linear_with_lora(
        model,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules
    )

    # 3. Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable LoRA params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    return model


# ----------------
# 4BIT QUANTIZATION
# ----------------

def apply_4bit_quantization(model, skip_modules=None, compute_dtype=torch.bfloat16, quant_type="nf4"):
    """ 4bit Quantization LoRA + Skip layers refer: unsloth-bnb-4bit:
        - Replaces the linear-layer with Linear4bit inside and outside LoRA wrappers with
        - Actual NF4 compression happens when model is moved to GPU (.cuda())
        - unsloth-bnb-4bit recommand skip layers HF:
        SKIP_QUANTIZATION_MODULES = [
            "lm_head", "multi_modal_projector", "merger",
            "modality_projection", "router", "mlp.gate",
            "block_sparse_moe.gate", 'mamba',"audio_tower",
            "vision_tower", "score", "classifier", "qa_outputs",
        ]
        - Link refer: https://unsloth.ai/blog/dynamic-4bit
    """
    if skip_modules is None:
        skip_modules = {"out_head", "tok_emb"}

    for name, module in model.named_children():
        if name in skip_modules:
            continue

        # Replace the frozen base linear inside LoRA
        if isinstance(module, LinearWithLoRA):
            old_linear = module.linear
            new_linear = bnb.nn.Linear4bit(
                old_linear.in_features,
                old_linear.out_features,
                bias=old_linear.bias is not None,
                compute_dtype=compute_dtype,
                quant_type=quant_type,
                # compress_statistics=True,  # ← Double quantization but ~0.5GB VRAM savings, less performance
            )
            new_linear.load_state_dict(old_linear.state_dict())
            new_linear.requires_grad_(False)
            module.linear = new_linear

        # Standalone frozen linear outside LoRA
        elif isinstance(module, torch.nn.Linear):
            if not any(p.requires_grad for p in module.parameters()):
                new_linear = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=compute_dtype,
                    quant_type=quant_type,
                    # compress_statistics=True,  #~0.5GB VRAM savings but less performance
                )
                new_linear.load_state_dict(module.state_dict())
                new_linear.requires_grad_(False)
                setattr(model, name, new_linear)
        else:
            apply_4bit_quantization(module, skip_modules, compute_dtype, quant_type)

    return model


# ----------------
# 8BIT QUANTIZATION
# ----------------

def apply_8bit_quantization(model, skip_modules=None, threshold=6.0):
    """ Apply  quantization for Linear Layers (LLM.int8()):
        - Replace frozen and unfrozen nn.Linear layers with bnb.nn.Linear8bitLt.
        - Uses mixed-precision decomposition: activation columns with outliers
          (above threshold) stay in fp16, rest computed in int8.
        - threshold=6.0 → recommended by LLM.int8() paper for stable training
        - Skips out_head and tok_emb (need full precision for logits/embeddings - unsloth-bnb-4bit)
    """
    if skip_modules is None:
        skip_modules = {"out_head", "tok_emb"}

    for name, module in model.named_children():
        if name in skip_modules:
            continue

        # Replace the frozen base linear inside LoRA wrapper
        if isinstance(module, LinearWithLoRA):
            old_linear = module.linear
            new_linear = bnb.nn.Linear8bitLt(
                old_linear.in_features,
                old_linear.out_features,
                bias=old_linear.bias is not None,
                has_fp16_weights=False,
                threshold=threshold,
            )
            new_linear.load_state_dict(old_linear.state_dict())
            new_linear.requires_grad_(False)
            module.linear = new_linear

        # Standalone frozen linear (not wrapped by LoRA)
        elif isinstance(module, torch.nn.Linear):
            if not any(p.requires_grad for p in module.parameters()):
                new_linear = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False,
                    threshold=threshold,
                )
                new_linear.load_state_dict(module.state_dict())
                new_linear.requires_grad_(False)
                setattr(model, name, new_linear)
        else:
            apply_8bit_quantization(module, skip_modules, threshold)

    return model



# ---------------------
# LIGHTNING DATA MODULE
# ---------------------

class LightningDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, train_records, val_records, cfg):
        """
        Args:
            tokenizer: Qwen3Tokenizer instance.
            train_records: Training records from load_gsm8k_json (train.json).
            val_records: Validation records from load_gsm8k_json (test.json).
            cfg: CFG dataclass with BATCH_SIZE, VAL_BATCH_SIZE, WORKERS, SEED,
                 MAX_SEQ_LEN, INCLUDE_THINKING.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.eos_token_id # Padding token
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
            allowed_max_length=CFG.MAX_SEQ_LEN,
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


# ------------------------
# LIGHTNING TRAIN PIPELINE
# ------------------------

class Qwen3_Lightning(pl.LightningModule):
    def __init__(self, model=None):
        super().__init__()
        # 0. Manual optimization
        self.automatic_optimization = False

        # 1. Gradient accumulation accumulators
        self._accum_loss = 0.0
        self._accum_tokens = 0
        self._opt_step = 0   # completed optimizer steps (manual optimization)
        self._total_tokens_trained = 0  # cumulative answer tokens with loss

        # 2. Store validation step losses & accuracy
        self.val_step_losses = []
        self.val_step_accuracies = []
        self.val_step_token_counts = []

        # 3. Init the model
        self.model = model


    def configure_model(self):
        # 1. Convert Precision Type
        precision_to_dtype = {
            "16": torch.float16,
            "16-mixed": torch.float16,
            "bf16": torch.bfloat16,
            "bf16-mixed": torch.bfloat16,
            "32": torch.float32
        }
        dtype = precision_to_dtype.get(str(self.trainer.precision), torch.float32)
        self.model.dtype = dtype

        # 2. Apply LoRA with target modules
        self.model = apply_lora(
            self.model,
            rank=CFG.RANK,
            alpha=CFG.ALPHA,
            target_modules=["W_query", "W_key", "W_value", "out_proj", "fc1", "fc2", "fc3"]
        )

        # 3. Quantize the frozen base weights (QLoRA only)
        if CFG.TUNING_MODE == "lora":
            return
        elif CFG.TUNING_MODE != "qlora":
            raise ValueError(f"Unknown TUNING_MODE: {CFG.TUNING_MODE}. Use 'lora' or 'qlora'.")

        if CFG.QUANT_BITS == "4bit":
            self.model = apply_4bit_quantization(self.model)
        elif CFG.QUANT_BITS == "8bit":
            self.model = apply_8bit_quantization(self.model)
        else:
            raise ValueError(f"Unknown QUANT_BITS: {CFG.QUANT_BITS}. Use '4bit' or '8bit'.")


    def training_step(self, batch, batch_idx):
        # 0. Get optimizer & scheduler
        opt = self.optimizers()
        sch = self.lr_schedulers()

        # 1. Forward
        inp_idx, targ_idx = batch
        logits = self.model(inp_idx)

        # 2. CE-Loss (reduction='sum': no divide by token count)
        loss_sum = F.cross_entropy(
            logits.flatten(0, 1), targ_idx.flatten(),
            ignore_index=-100, reduction='sum'
        )
        num_tokens = (targ_idx != -100).sum()

        # 3. Backward (accumulates gradients)
        self.manual_backward(loss_sum)

        # 4. Track loss sum & token count
        self._accum_loss += loss_sum.detach().item()
        self._accum_tokens += num_tokens.item()

        # 5. Optimizer step after G accumulation steps
        if ((batch_idx + 1) % CFG.GRAD_ACCUM == 0):
            self._optimizer_step(opt, sch)


    def on_train_epoch_end(self):
        # Flush any leftover accumulated gradients at epoch boundary
        if self._accum_tokens > 0:
            opt = self.optimizers()
            sch = self.lr_schedulers()
            self._optimizer_step(opt, sch, from_epoch_end=True)


    def _optimizer_step(self, opt, sch, from_epoch_end=False):
        """ Gradient Accumulation Training (Unsloth fix):
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

        # 1. Gradient clipping (every step, same as HF Trainer default)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=CFG.GRAD_NORM)

        # 2. Optimize + Scheduler
        opt.step()
        opt.zero_grad()
        sch.step()
        self._opt_step += 1

        # 3. Final loss = total_loss_sum / total_tokens
        train_loss = self._accum_loss / M

        # 4. Log — loss, LR, grad-norm, throughput and memory
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

        # 5. Reset accumulators
        self._accum_loss = 0.0
        self._accum_tokens = 0


    def calc_mean_tokens_accuracy(self, logits, targets, ignore_index=-100):
        """Mean token accuracy: proportion of correct top-1 predictions on non-masked tokens."""
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            mask = targets != ignore_index
            correct = ((predictions == targets) & mask).sum()
            total = mask.sum()
            accuracy = (correct / total).item() if total > 0 else 0.0
        return accuracy


    def validation_step(self, batch, batch_idx):
        inp_idx, targ_idx = batch
        logits = self.model(inp_idx)
        loss = F.cross_entropy(logits.flatten(0, 1), targ_idx.flatten(), ignore_index=-100)
        self.val_step_losses.append(loss)

        # Compute mean_token_accuracy per batch
        acc = self.calc_mean_tokens_accuracy(logits, targ_idx)
        num_tokens = (targ_idx != -100).sum().item()

        # Weighted by token count
        self.val_step_accuracies.append(acc * num_tokens)
        self.val_step_token_counts.append(num_tokens)
        return loss


    def on_validation_epoch_end(self):
        # Loss - skip nan loss
        stacked = torch.stack(self.val_step_losses)
        valid = stacked[~torch.isnan(stacked)]
        avg_val_loss = valid.mean() if valid.numel() > 0 else torch.tensor(0.0)
        self.log('Validation/loss', avg_val_loss, sync_dist=True, prog_bar=True)

        # Mean_token_accuracy (weighted by token count)
        total_weighted = sum(self.val_step_accuracies)
        total_tokens = sum(self.val_step_token_counts)
        val_token_acc = total_weighted / total_tokens if total_tokens > 0 else 0.0
        self.log('Validation/accuracy', val_token_acc, sync_dist=True, prog_bar=True)

        # Clear cache
        self.val_step_losses.clear()
        self.val_step_accuracies.clear()
        self.val_step_token_counts.clear()


    def loraplus_optimizer(self):
        """LoRA+ optimizer (Hayou et al. 2024, ref: github.com/nikhilgsh/loraplus)
        groupA: lora_A + other 2D params → base LR (η_A) with weight_decay
        groupB: lora_B + 1D params (bias/norm) → η_A × ratio with weight_decay
        groupB_no_decay: 1D params in groupB → η_A × ratio without weight_decay
        """
        lr = CFG.LR
        lr_ratio = CFG.LORAPLUS_RATIO

        group_A = []           # lora_A + other 2D params (base LR, weight_decay)
        group_B = []           # lora_B 2D params (higher LR, weight_decay)
        group_B_nodecay = []   # 1D params: bias/norm (higher LR, no weight_decay)

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # lora_B or 1D params → groupB (higher LR)
            if '.lora.B' in name or param.ndim == 1:
                if param.ndim >= 2:
                    group_B.append(param)
                else:
                    group_B_nodecay.append(param)
            else:
                # lora_A + other 2D params → groupA (base LR)
                group_A.append(param)

        optim_groups = [
            {'params': group_A, 'lr': lr, 'weight_decay': 0.01},
            {'params': group_B, 'lr': lr * lr_ratio, 'weight_decay': 0.01},
            {'params': group_B_nodecay, 'lr': lr * lr_ratio, 'weight_decay': 0.0},
        ]
        # Filter out empty groups
        optim_groups = [g for g in optim_groups if len(g['params']) > 0]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
        return optimizer


    def standard_optimizer(self):
        """Standard AdamW with weight decay for 2D+ params only."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': 0.01},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=CFG.LR, betas=(0.9, 0.95), eps=1e-8) # Or AdamW8bit → saves ~50% optimizer VRAM
        return optimizer


    def configure_optimizers(self):
        # 1. Select optimizer strategy
        if CFG.LORAPLUS_RATIO is not None:
            optimizer = self.loraplus_optimizer()
        else:
            optimizer = self.standard_optimizer()

        # 2. Warmup + Cosine-Decay Scheduler
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


# --------------------------
# ANSWER FORMAT / CORRECTNESS
# --------------------------

# A well-formed boxed answer: literal \boxed{...} with no nested braces.
# Rejects "\boxed{" (unclosed), "boxed{5}" (no backslash) and "\boxed{}" (empty).
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def normalize_answer(text):
    """Canonical form for comparing a predicted answer to the ground truth.

    GSM8K answers are plain numbers, so thousands separators, currency symbols
    and a trailing period are formatting noise rather than a wrong answer.
    """
    return text.strip().replace(",", "").replace("$", "").rstrip(".").strip()


THINK_CLOSE = "</think>"


def split_thinking(completion):
    """Split a thinking-mode completion into (thinking, answer_part, closed).

    The prompt already ends with "<think>\n", so the model emits the chain of
    thought first and is supposed to close it with </think> before the boxed
    answer. closed is False when the tag never appeared — the reasoning either
    ran past INFER_MAX_NEW_TOKENS or the model skipped the tag altogether.
    """
    if THINK_CLOSE in completion:
        thinking, _, rest = completion.partition(THINK_CLOSE)
        return thinking.strip(), rest.strip(), True
    return completion.strip(), "", False


def parse_boxed_answer(completion):
    """Split a completion into (format_ok, answer_text).

    format_ok is True only when a well-formed \boxed{...} carrying non-empty
    content is present — independent of whether that content is correct.
    """
    match = BOXED_RE.search(completion)
    if match is None:
        return False, None
    content = match.group(1).strip()
    if not content:
        return False, None
    return True, content


# --------------------------
# PERIODIC SAMPLE INFERENCE
# --------------------------

class SampleInferenceCallback(pl.Callback):
    """Print `question:` / `answer:` pairs for held-out problems during training.

    Fires every ``every_pct`` of the planned optimizer steps (0.20 -> at
    20/40/60/80/100%) so answer quality can be watched as the run progresses.
    Greedy decoding; only the newly generated tokens are decoded, so the prompt
    is never echoed back.
    """

    def __init__(self, tokenizer, records, num_samples, every_pct,
                 max_new_tokens, total_steps, seed=CFG.SEED):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.total_steps = max(1, total_steps)
        self.every = max(1, int(round(self.total_steps * every_pct)))
        self.samples = random.Random(seed).sample(
            records, min(num_samples, len(records))
        )
        self._fired = set()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = pl_module._opt_step
        if step == 0 or step % self.every != 0 or step in self._fired:
            return
        self._fired.add(step)
        self._run(pl_module, step)

    def on_train_end(self, trainer, pl_module):
        # Final round, unless a boundary already fired close to the end
        step = pl_module._opt_step
        if step <= 0 or step in self._fired:
            return
        if self._fired and (step - max(self._fired)) < self.every // 2:
            return
        self._fired.add(step)
        self._run(pl_module, step)

    @torch.no_grad()
    def _generate(self, model, prompt):
        device = next(model.parameters()).device
        x = torch.tensor([self.tokenizer.encode(prompt)],
                         dtype=torch.long, device=device)
        eos = self.tokenizer.eos_token_id
        new_ids = []
        # The Trainer runs the forward pass under AMP autocast. The 4-bit base
        # layers compute in bfloat16 while the LoRA adapters stay fp32, so the
        # same autocast is required here or the matmuls disagree on dtype.
        use_amp = device.type == 'cuda'
        for _ in range(self.max_new_tokens):
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # greedy
            if next_id.item() == eos:
                break
            new_ids.append(next_id.item())
            x = torch.cat([x, next_id], dim=1)
        return self.tokenizer.decode(new_ids).strip()

    def _run(self, pl_module, step):
        model = pl_module.model
        was_training = model.training
        model.eval()

        pct = 100.0 * step / self.total_steps
        bar = "=" * 78
        lines = [
            bar,
            f"SAMPLE INFERENCE  |  step {step}/{self.total_steps}  ({pct:.0f}% of training)",
            bar,
        ]

        n_correct = 0
        n_format = 0
        n_think_closed = 0
        for i, rec in enumerate(self.samples, 1):
            prompt = format_prompt_only(
                rec["question"], include_thinking=CFG.INCLUDE_THINKING
            )
            try:
                completion = self._generate(model, prompt)
            except Exception as e:
                completion = f"<generation failed: {type(e).__name__}: {e}>"

            expected = rec["answer"]
            format_ok, predicted = parse_boxed_answer(completion)
            hit = format_ok and normalize_answer(predicted) == normalize_answer(expected)
            n_correct += hit
            n_format += format_ok

            status = "CORRECT" if hit else ("wrong" if format_ok else "BAD FORMAT")
            lines += [
                "",
                f"[{i}/{len(self.samples)}] {status}",
                f"question: {rec['question']}",
            ]

            if CFG.INCLUDE_THINKING:
                # The chain of thought is what the model is being trained on now,
                # so show it next to the reference reasoning rather than only the
                # final boxed answer.
                thinking, answer_part, think_closed = split_thinking(completion)
                n_think_closed += think_closed
                n_think_tok = len(self.tokenizer.encode(thinking)) if thinking else 0
                lines += [
                    f"think: ({n_think_tok} tok, "
                    f"{'closed' if think_closed else 'UNCLOSED </think>'}): "
                    f"{thinking if thinking else '<empty>'}",
                    f"answer: {answer_part if answer_part else '<none after </think>>'}",
                    f"expected_think: {rec['thinking']}",
                ]
            else:
                lines.append(f"answer: {completion}")

            lines.append(f"expected: \\boxed{{{expected}}}")

        n = max(1, len(self.samples))
        acc = 100.0 * n_correct / n
        lines += [
            "",
            "-" * 78,
            f"box_answer_match_accuracy: {n_correct}/{len(self.samples)} ({acc:.1f}%)",
            f"format_answer_matched_count: {n_format}/{len(self.samples)}",
        ]
        if CFG.INCLUDE_THINKING:
            lines.append(
                f"think_closed_count: {n_think_closed}/{len(self.samples)} "
                f"(max_new_tokens={self.max_new_tokens})"
            )
        lines.append(bar)

        report = "\n".join(lines)
        print("\n" + report + "\n", flush=True)

        try:
            pl_module.log('Validation/box_answer_match_accuracy', acc,
                          sync_dist=True, prog_bar=False)
            pl_module.log('Validation/format_answer_matched_count', float(n_format),
                          sync_dist=True, prog_bar=False)
            if CFG.INCLUDE_THINKING:
                pl_module.log('Validation/think_closed_count', float(n_think_closed),
                              sync_dist=True, prog_bar=False)
        except Exception:
            pass  # logging is not available outside a training batch

        if was_training:
            model.train()


if __name__ == '__main__':

    # ---- Data Preparation (train.json trains, test.json validates)
    DATA_PATH = PROJECT_ROOT / "data" / "Gsm8k"
    train_records = load_gsm8k_json(str(DATA_PATH / "train.json"))
    val_records = load_gsm8k_json(str(DATA_PATH / "test.json"))
    print(f"Train records: {len(train_records)}, Val records: {len(val_records)}")

    CFG.STEPS = (len(train_records) // CFG.BATCH_SIZE // CFG.GRAD_ACCUM) * CFG.EPOCHS
    print(f"Optimizer steps per epoch: {CFG.STEPS}")

    # Validate on the FULL test set at the same 20% marks as the sample rounds.
    # val_check_interval counts training batches, so convert optimizer steps back
    # to batches with GRAD_ACCUM.
    CFG.VAL_EVERY_N_BATCHES = CFG.GRAD_ACCUM * max(1, round(CFG.STEPS * CFG.INFER_EVERY_PCT))
    print(f"Validation every {CFG.VAL_EVERY_N_BATCHES} batches "
          f"({CFG.INFER_EVERY_PCT:.0%} of training)")

    # ---- Model & Tokenizer
    model_cfg, repo_id = QWEN3_MODELS[CFG.MODEL_SIZE]
    model = Qwen3Model(model_cfg)
    model = from_pretrained(model, repo_id=repo_id)
    tokenizer = Qwen3Tokenizer(str(MODELS_DIR / "tokenizer.json"))

    # ---- DataModule
    datamodule = LightningDataModule(
        tokenizer=tokenizer,
        train_records=train_records,
        val_records=val_records,
        cfg=CFG
    )

    # ---- Lightning Train
    qwen3_module = Qwen3_Lightning(model=model)

    # ---- Logger
    from datetime import datetime
    date = datetime.now().strftime("%d_%m_%Y")
    mode_tag = "LoRA" if CFG.TUNING_MODE == "lora" else f"QLoRA_{CFG.QUANT_BITS}"
    think_tag = "CoT" if CFG.INCLUDE_THINKING else "Direct"
    save_dir = f'./logs/2_Qwen3_{CFG.MODEL_SIZE}_Gsm8k_{think_tag}_{mode_tag}_r{CFG.RANK}_a{CFG.ALPHA}/'

    # Experiment name: Qwen3_Gsm8k_<size>_<LoRA|QLoRA_bits>[_think]_<dd_mm_yyyy>
    # Chain-of-thought runs carry a _think suffix so they never land on top of
    # the direct-answer experiments already tracked under the plain name.
    think_suffix = "_think" if CFG.INCLUDE_THINKING else ""
    run_name = f'Qwen3_Gsm8k_{CFG.MODEL_SIZE}_{mode_tag}{think_suffix}_{date}'

    # CSV stays as the local record — plot_training_curves() reads its metrics.csv
    csv_logger = pl.loggers.CSVLogger(save_dir=save_dir, name=f'{date}')
    loggers = [csv_logger]

    # Lightning AI remote tracking (needs `lightning login` or LIGHTNING_API_KEY
    # + LIGHTNING_USER_ID). Falls back to CSV-only so a missing login never
    # blocks a long training run.
    if not lightning_ai_ready():
        print("=" * 78)
        print("[logger] Lightning AI: remote tracking is OFF (reason above).")
        print("[logger] Logging locally to CSV instead:")
        print(f"[logger]   {save_dir}")
        print("[logger] To turn remote tracking on, install `litlogger` and run")
        print("[logger] `lightning login` (or set LIGHTNING_API_KEY +")
        print("[logger] LIGHTNING_USER_ID), then restart training.")
        print("=" * 78)
    else:
        lit_logger = pl.loggers.LitLogger(
            name=run_name,
            teamspace=CFG.TEAMSPACE,
            # Lightning's wrapper defaults save_logs=True, which makes litlogger
            # re-exec this whole script inside a PTY to capture terminal output
            # (litlogger/experiment.py:112). The parent then blocks holding its
            # GPU memory while the child trains — two 14B models do not fit, so
            # console capture stays off.
            save_logs=False,
            metadata={
                'model': f'Qwen3-{CFG.MODEL_SIZE}',
                'dataset': 'gsm8k',
                'tuning_mode': CFG.TUNING_MODE,
                'quant_bits': str(CFG.QUANT_BITS),
                'rank': str(CFG.RANK),
                'alpha': str(CFG.ALPHA),
                'lr': str(CFG.LR),
                'grad_accum': str(CFG.GRAD_ACCUM),
                'max_seq_len': str(CFG.MAX_SEQ_LEN),
                'include_thinking': str(CFG.INCLUDE_THINKING),
            },
        )
        # Append, never insert: trainer.logger is loggers[0] and decides where
        # the checkpoint callbacks write. CSVLogger must stay first.
        loggers.append(lit_logger)
        print(f"[logger] Lightning AI tracking enabled")
        print(f"[logger]   teamspace: {CFG.TEAMSPACE}")
        print(f"[logger]   experiment: {run_name}")

    # ---- Checkpoint callback (full .ckpt for resuming training)
    ckpt = pl.callbacks.ModelCheckpoint(
        monitor='Validation/accuracy',
        # Disk-bound: validation now runs 5x, and each .ckpt is ~9.5GB while each
        # merged .pth is ~52GB. Keep only the single best of each.
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        filename='{epoch:02d}-{Validation/loss:.4f}-{Validation/accuracy:.4f}',
        # '/' in a metric name would become a directory separator when Lightning
        # auto-inserts "name=value" into the filename, so insert values only.
        auto_insert_metric_name=False,
        mode='max',
        save_on_train_epoch_end=True,   # save at epoch end
    )

    # ---- Merged model callback (LoRA → base .pth for direct Qwen3Model loading)
    merged_ckpt = LoRAMergeCheckpoint(
        monitor='Validation/accuracy',
        mode='max',
        save_top_k=1,
        filename_template='{epoch:02d}-{Validation/loss:.4f}-{Validation/accuracy:.4f}',
    )

    # ---- Periodic sample inference (every 20% of training, 20 held-out questions)
    sample_infer = SampleInferenceCallback(
        tokenizer=tokenizer,
        records=val_records,
        num_samples=CFG.INFER_SAMPLES,
        every_pct=CFG.INFER_EVERY_PCT,
        max_new_tokens=CFG.INFER_MAX_NEW_TOKENS,
        total_steps=CFG.STEPS,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        callbacks=[ckpt, merged_ckpt, sample_infer],
        logger=loggers,
        max_epochs=CFG.EPOCHS,
        precision='bf16',
        val_check_interval=CFG.VAL_EVERY_N_BATCHES,  # full test set every 20%
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=qwen3_module, datamodule=datamodule)

    # ---- Plot Results ----
    plot_training_curves(csv_logger)

