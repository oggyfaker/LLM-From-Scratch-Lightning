import os
import json
import shutil

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # non-interactive backend

import torch
import bitsandbytes as bnb
import pytorch_lightning as pl
from safetensors.torch import save_file


# -----
# MERGE LORA WEIGHTS INTO BASE MODEL
# -----

def merge_lora_state_dict(model):
    """Build a state_dict compatible with vanilla Qwen3Model by merging LoRA into base weights.

    For each LinearWithLoRA module, dequantizes the base weight (if 4-bit quantized),
    adds the LoRA contribution (alpha/rank * (A @ B).T), and maps back to the original key name.
    All other parameters are copied as-is.

    Returns:
        dict: A state_dict that can be loaded directly via Qwen3Model.load_state_dict().
    """
    merged = {}
    handled_prefixes = set()

    # 1. Merge LinearWithLoRA modules: dequantize base + fold in LoRA
    # Duck-type check: any module with .linear and .lora attributes is a LoRA wrapper
    for name, module in model.named_modules():
        if hasattr(module, 'linear') and hasattr(module, 'lora'):
            handled_prefixes.add(name + ".")

            base = module.linear
            if isinstance(base, bnb.nn.Linear4bit):
                w = bnb.functional.dequantize_4bit(
                    base.weight.data, base.weight.quant_state
                ).clone()
            else:
                w = base.weight.data.clone()

            # LoRA forward:  (alpha/rank) * (x @ A @ B)
            # Linear forward: x @ W.T
            # Merged W_new = W + (alpha/rank) * (A @ B).T
            A = module.lora.A.data   # (in_dim, rank)
            B = module.lora.B.data   # (rank, out_dim)
            lora_delta = module.lora.scaling * (A @ B).T  # (out_dim, in_dim)
            merged[name + ".weight"] = (w + lora_delta.to(w.dtype)).cpu()

    # 2. Copy all remaining parameters (tok_emb, out_head, RMSNorm scales, etc.)
    for pname, param in model.named_parameters():
        if not any(pname.startswith(p) for p in handled_prefixes):
            merged[pname] = param.data.cpu()

    return merged


class LoRAMergeCheckpoint(pl.Callback):
    """Save merged model (LoRA weights folded into base) as .pth after validation.

    Mirrors ModelCheckpoint behaviour (monitor, mode, save_top_k) but writes a
    plain state_dict to ``<log_dir>/model_pretrained/`` that can be loaded
    directly into a vanilla Qwen3Model.
    """

    def __init__(self, monitor='val_mean_token_accuracy', mode='max', save_top_k=3,
                 filename_template='{epoch:02d}-{total_val_loss:.4f}-{val_mean_token_accuracy:.4f}'):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename_template = filename_template
        self._saved = []  # list of (score, path)

    def on_validation_end(self, trainer, pl_module):
        # Use on_validation_end (not on_validation_epoch_end) because Lightning
        # calls Callback.on_validation_epoch_end BEFORE Module.on_validation_epoch_end,
        # so metrics aren't logged yet. on_validation_end fires after everything.
        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor)
        if current is None:
            return
        score = current.item() if hasattr(current, 'item') else float(current)

        # --- Check whether this score deserves saving (top-k) ---
        if len(self._saved) >= self.save_top_k:
            worst = self._saved[-1][0]
            better = score > worst if self.mode == 'max' else score < worst
            if not better:
                return

        # --- Build path ---
        save_dir = os.path.join(trainer.logger.log_dir, "model_pretrained")
        os.makedirs(save_dir, exist_ok=True)

        fmt = {
            'epoch': trainer.current_epoch,
            'step': trainer.global_step,
        }
        for k, v in metrics.items():
            fmt[k] = v.item() if hasattr(v, 'item') else float(v)
        filename = self.filename_template.format(**fmt) + ".pth"
        save_path = os.path.join(save_dir, filename)

        # --- Merge LoRA → base and save ---
        merged_sd = merge_lora_state_dict(pl_module.model)
        torch.save(merged_sd, save_path)
        print(f"Merged model saved to: {save_path}")

        # --- Maintain top-k list ---
        self._saved.append((score, save_path))
        self._saved.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))

        while len(self._saved) > self.save_top_k:
            _, old_path = self._saved.pop()
            if os.path.exists(old_path):
                os.remove(old_path)
                print(f"Removed old merged model: {old_path}")

    @property
    def best_model_path(self):
        """Return the path of the best saved .pth (first in sorted top-k list)."""
        if self._saved:
            return self._saved[0][1]
        return None


# -----
# TRAINING VISUALIZATION
# -----

def plot_training_curves(logger):
    """Plot training loss, validation loss, and validation token accuracy from CSV logger.

    Saves the figure to ``<log_dir>/loss_curve.png``.
    """
    metrics_path = os.path.join(logger.log_dir, "metrics.csv")
    metrics = pd.read_csv(metrics_path)

    train_df_plot = metrics[["step", "train_loss"]].dropna().reset_index(drop=True)
    val_df_plot = metrics[["step", "total_val_loss"]].dropna().reset_index(drop=True)
    val_acc_df = metrics[["step", "val_mean_token_accuracy"]].dropna().reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Loss
    ax1.plot(train_df_plot["step"], train_df_plot["train_loss"],
            linewidth=1, alpha=0.8, label="Train Loss (per step)")
    ax1.plot(val_df_plot["step"], val_df_plot["total_val_loss"],
            marker="s", markersize=5, linewidth=1.5, label="Val Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Val Mean Token Accuracy
    if not val_acc_df.empty:
        ax2.plot(val_acc_df["step"], val_acc_df["val_mean_token_accuracy"],
                marker="s", markersize=5, linewidth=1.5, label="Val Token Acc")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Val Mean Token Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    save_path = os.path.join(logger.log_dir, "loss_curve.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to: {save_path}")
