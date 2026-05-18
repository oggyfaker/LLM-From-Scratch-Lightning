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

from data.Jigsaw2026.data_utils import (
    get_dataframe_to_train, split_dataset,
    JigsawDataset, custom_collate_fn
)
from utils.checkpoint_utils import (
    LoRAMergeCheckpoint, plot_training_curves,
)

sys.path.append("./models/Qwen3")
from qwen3 import Qwen3Model, Qwen3Tokenizer, QWEN_14B_CFG, from_pretrained



# ------
# CONFIG
# ------
@dataclass
class CFG:
    EPOCHS = 1
    BATCH_SIZE = 1
    VAL_BATCH_SIZE = 1
    WORKERS = 2

    RANK = 16
    ALPHA = 32
    QUANT_BITS = "4bit"  # "4bit" or "8bit"

    GRAD_ACCUM = 64

    LR = 1.5e-4
    MIN_LR = LR * 0.1
    WARMUP_STEPS = 20  # ~10% of total optimizer steps for linear warmup
    SEED = 1001

    MAX_SEQ_LEN = 256


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
    def __init__(self, tokenizer, train_df, val_df, cfg):
        """
        Args:
            tokenizer: Qwen3Tokenizer instance.
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            cfg: CFG dataclass with BATCH_SIZE, VAL_BATCH_SIZE, WORKERS, SEED, 
                 MAX_SEQ_LEN, BASE_PROMPT, POSITIVE_ANSWER, NEGATIVE_ANSWER.
        """
        super().__init__()  
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.eos_token_id # Padding token
        self.train_dataset = JigsawDataset(
            train_df, self.tokenizer
        )
        self.val_dataset = JigsawDataset(
            val_df, self.tokenizer
        )
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

        # 3. Quantize based on CFG.QUANT_BITS
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
        for p in self.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.div_(M)

        # 1. Gradient clipping (every step, same as HF Trainer default)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # 2. Optimize + Scheduler
        opt.step()
        opt.zero_grad()
        sch.step()

        # 3. Final loss = total_loss_sum / total_tokens
        train_loss = self._accum_loss / M 
        
        # 4. Log
        if from_epoch_end:
            self.log('train_loss', train_loss,
                sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
            self.log('grad_norm', grad_norm.item(),
                sync_dist=True, prog_bar=False, on_step=False, on_epoch=True)
        else:
            self.log('train_loss', train_loss,
                sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)
            self.log('grad_norm', grad_norm.item(),
                sync_dist=True, prog_bar=False, on_step=True, on_epoch=False)

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
        stacked = torch.stack(self.val_step_losses)
        valid = stacked[~torch.isnan(stacked)]
        avg_val_loss = valid.mean() if valid.numel() > 0 else torch.tensor(0.0)
        self.val_step_losses.clear()
        self.log('total_val_loss', avg_val_loss, sync_dist=True, prog_bar=True)

        # Aggregate mean_token_accuracy (weighted by token count)
        total_weighted = sum(self.val_step_accuracies)
        total_tokens = sum(self.val_step_token_counts)
        val_token_acc = total_weighted / total_tokens if total_tokens > 0 else 0.0
        
        self.val_step_accuracies.clear()
        self.val_step_token_counts.clear()
        self.log('val_mean_token_accuracy', val_token_acc, sync_dist=True, prog_bar=True)


    def configure_optimizers(self):
        # 1. AdamW with weight decay for 2D+ params only
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': 0.01},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Use bnb 8-bit AdamW for quantized training (Unsloth default) → saves ~50% optimizer VRAM
        optimizer = bnb.optim.AdamW8bit(optim_groups, lr=CFG.LR, betas=(0.9, 0.95), eps=1e-8)

        # 2. Warmup + Cosine Decay scheduler
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
    DATA_PATH = "./data/Jigsaw2026"
    full_df = get_dataframe_to_train(DATA_PATH, seed=CFG.SEED)
    train_df, val_df = split_dataset(full_df, val_ratio=0.2, seed=CFG.SEED)
    CFG.STEPS = (len(train_df) // CFG.BATCH_SIZE // CFG.GRAD_ACCUM) * CFG.EPOCHS
    print(f"Optimizer steps per epoch: {CFG.STEPS}")

    # ---- Model & Tokenizer
    model = Qwen3Model(QWEN_14B_CFG)
    model = from_pretrained(model, repo_id="Qwen/Qwen3-14B")
    tokenizer = Qwen3Tokenizer("models/Qwen3/tokenizer.json")

    # ---- DataModule
    datamodule = LightningDataModule(
        tokenizer=tokenizer, 
        train_df=train_df, 
        val_df=val_df, 
        cfg=CFG
    )

    # ---- Lightning Train 
    qwen3_module = Qwen3_Lightning(model=model)

    # ---- Logger
    from datetime import datetime
    date = datetime.now().strftime("%d_%m_%Y")
    logger = pl.loggers.CSVLogger(
        save_dir=f'./logs/9_Qwen3_14B_Jigsaw_LoRA_r{CFG.RANK}_a{CFG.ALPHA}/', 
        name=f'{date}'
    )

    # ---- Checkpoint callback (full .ckpt for resuming training)
    ckpt = pl.callbacks.ModelCheckpoint(
        monitor='val_mean_token_accuracy',
        save_top_k=3,
        save_last=True,     
        save_weights_only=True,
        filename='{epoch:02d}-{total_val_loss:.4f}-{val_mean_token_accuracy:.4f}',
        mode='max',
        save_on_train_epoch_end=True,   # save at epoch end
    )

    # ---- Merged model callback (LoRA → base .pth for direct Qwen3Model loading)
    merged_ckpt = LoRAMergeCheckpoint(
        monitor='val_mean_token_accuracy',
        mode='max',
        save_top_k=3,
        filename_template='{epoch:02d}-{total_val_loss:.4f}-{val_mean_token_accuracy:.4f}',
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        callbacks=[ckpt, merged_ckpt],
        logger=logger,
        max_epochs=CFG.EPOCHS,
        precision='bf16',
        # val_check_interval=500, # validate every 100 training steps (1767 is total steps of epoch)
        check_val_every_n_epoch=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=qwen3_module, datamodule=datamodule)

    # ---- Plot Results ----
    plot_training_curves(logger)
