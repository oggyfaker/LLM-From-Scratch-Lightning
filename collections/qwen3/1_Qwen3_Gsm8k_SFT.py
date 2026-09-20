import sys
import math
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
    load_gsm8k_json, Gsm8kDataset, custom_collate_fn
)
# Grading + the periodic generative eval live with the dataset, so every GSM8K
# run (dense, MoE, whatever comes next) scores with the exact same code.
from data.Gsm8k.metric import Gsm8kEvalCallback, ACCURACY_KEY
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

    # --- Periodic full-test-set generative evaluation ---
    INFER_EVERY_PERCENTAGE = 0.20     # evaluate every 20% process training 
    INFER_SAMPLES = None       # None is score all 1319 held-out test questions (or define exact number inference samples)

    # A decode step costs the SAME at batch 16 as at batch 256 (measured: 187 ms
    # vs ~190 ms) because bitsandbytes re-dequantises all 14B 4-bit weights every
    # step no matter how many tokens ride along. Batch size is therefore the only
    # lever on eval wall time, and a SMALLER batch is strictly slower: bs=16
    # would need 83 passes over the test set where bs=256 needs 6.
    # The callback halves it on CUDA OOM rather than dying, and rebalances so all
    # chunks are equal (1319 at bs=256 -> 6 x 220, not 5 x 256 + 1 x 39).
    INFER_BATCH_SIZE = 256

    # Generation budget derived from the training config rather than guessed.
    # Gsm8kDataset drops any sample whose prompt + CoT + \boxed{} exceeds
    # MAX_SEQ_LEN, so the model has never once produced a completion longer than
    # MAX_SEQ_LEN - len(prompt); the longest ground truth that survives the
    # filter is 346 tokens (p99 = 221). Generating 768 was both out of
    # distribution and ~2x wasted compute. Qwen3Model.generate_batch clamps this
    # again per batch to MAX_SEQ_LEN - prompt_len, so a long prompt can never
    # push generation past what training ever saw.
    INFER_MAX_NEW_TOKENS = MAX_SEQ_LEN - 128 if INCLUDE_THINKING else 64

    # Turn-enders BEYOND the tokenizer's own eos_token_id, which the eval
    # callback already picks up on its own. Qwen3's chat format ends an
    # assistant turn with <|im_end|> but also honours <|endoftext|>; this is the
    # same pair the vLLM inference scripts pass as stop_token_ids, and these
    # tokens are Qwen3's, so they are declared here rather than in the shared
    # dataset callback.
    INFER_STOP_TOKENS = ("<|endoftext|>", "<|im_end|>")


# -----------------
# LOW-RANK ADAPTION
# -----------------

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
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

        total_weighted = sum(self.val_step_accuracies)
        total_tokens = sum(self.val_step_token_counts)
        val_token_acc = total_weighted / total_tokens if total_tokens > 0 else 0.0
        self.log('Validation/token_accuracy', val_token_acc, sync_dist=True, prog_bar=True)

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


if __name__ == '__main__':

    # ---- Data Preparation
    DATA_PATH = PROJECT_ROOT / "data" / "Gsm8k"
    train_records = load_gsm8k_json(str(DATA_PATH / "train.json"))
    val_records = load_gsm8k_json(str(DATA_PATH / "test.json"))
    print(f"Train records: {len(train_records)}, Val records: {len(val_records)}")

    CFG.STEPS = (len(train_records) // CFG.BATCH_SIZE // CFG.GRAD_ACCUM) * CFG.EPOCHS
    print(f"Optimizer steps per epoch: {CFG.STEPS}")

    # Validate 20% marks as the sample rounds..
    CFG.VAL_EVERY_N_STEPS = CFG.GRAD_ACCUM * max(1, round(CFG.STEPS * CFG.INFER_EVERY_PERCENTAGE))
    print(f"Validation every {CFG.VAL_EVERY_N_STEPS} step "
          f"({CFG.INFER_EVERY_PERCENTAGE:.0%} of training)")

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

    # ---- Lit-Logger
    from datetime import datetime
    date = datetime.now().strftime("%d_%m_%y")
    mode_tag = "LoRA" if CFG.TUNING_MODE == "lora" else f"QLoRA_{CFG.QUANT_BITS}"
    think_suffix = "_think" if CFG.INCLUDE_THINKING else ""

    exp_name = f'Qwen3_Gsm8k_{CFG.MODEL_SIZE}_{mode_tag}{think_suffix}'
    csv_logger = pl.loggers.CSVLogger(save_dir=str(Path('./logs') / exp_name), name=date)
    loggers = [csv_logger]

    RUN_DIR = Path(csv_logger.log_dir)          # logs/<exp_name>/<date>/version_N
    save_dir = str(RUN_DIR)
    VALIDATION_DIR = RUN_DIR / 'validation'

    run_name = f'{exp_name}_{date}_v{csv_logger.version}'
    print(f"[logger] run directory: {RUN_DIR}")
    print(f"[logger] experiment name: {run_name}")
    print(f"[eval] per-round answers -> {VALIDATION_DIR}/round<N>.json")

    lit_logger = pl.loggers.LitLogger(
        name=run_name,
        teamspace=CFG.TEAMSPACE,
        root_dir=save_dir,
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
    lit_logger._version = str(csv_logger.version)
    loggers.append(lit_logger)
    print(f"[logger]   teamspace: {CFG.TEAMSPACE}")
    print(f"[logger]   experiment: {run_name}")

    # ---- Checkpoint callback
    CKPT_MONITOR = ACCURACY_KEY
    CKPT_NAME = '{epoch:02d}-{Validation/loss:.4f}-{' + ACCURACY_KEY + ':.2f}'
    ckpt = pl.callbacks.ModelCheckpoint(
        monitor=CKPT_MONITOR,
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        filename=CKPT_NAME,
        auto_insert_metric_name=False,
        mode='max',
        save_on_train_epoch_end=False,
        dirpath=str(RUN_DIR / 'checkpoints'),
    )

    # ---- Merged model callback (LoRA → base .pth for direct Qwen3Model loading)
    merged_ckpt = LoRAMergeCheckpoint(
        monitor=CKPT_MONITOR,
        mode='max',
        save_top_k=1,
        filename_template=CKPT_NAME,
    )

    # ---- Generative evaluation on the FULL test set, every 20% of training
    boxed_eval = Gsm8kEvalCallback(
        tokenizer=tokenizer,
        records=val_records,
        total_steps=CFG.STEPS,
        output_dir=VALIDATION_DIR,
        every_pct=CFG.INFER_EVERY_PERCENTAGE,
        num_samples=CFG.INFER_SAMPLES,
        batch_size=CFG.INFER_BATCH_SIZE,
        max_new_tokens=CFG.INFER_MAX_NEW_TOKENS,
        max_total_len=CFG.MAX_SEQ_LEN,
        thinking_mode=CFG.INCLUDE_THINKING,
        stop_tokens=CFG.INFER_STOP_TOKENS,
        seed=CFG.SEED,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        callbacks=[ckpt, merged_ckpt, boxed_eval],
        logger=loggers,
        max_epochs=CFG.EPOCHS,
        precision='bf16',
        val_check_interval=CFG.VAL_EVERY_N_STEPS,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=qwen3_module, datamodule=datamodule)

    # ---- Plot Results ----
    plot_training_curves(csv_logger)

