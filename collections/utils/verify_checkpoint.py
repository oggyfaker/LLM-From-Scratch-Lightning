"""
Diagnostic: Verify merged checkpoint correctness.

Compares the training model's output (with LoRA in-place) against
the merged checkpoint loaded into a fresh base model (no LoRA).
If outputs diverge, identifies WHERE the mismatch occurs (which layer).

Usage:
    python collections/utils/verify_checkpoint.py

This will:
  1. Load training model with LoRA applied (from the training script's setup)
  2. Build the merged state_dict
  3. Load a fresh base model with the merged weights (simulating vLLM loading)
  4. Compare hidden states layer-by-layer and final logits
  5. Report any discrepancies
"""

import os
import sys
import torch
import torch.nn as nn

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from unsloth import FastLanguageModel
from utils.checkpoint_moe_utils import merge_moe_lora_state_dict


# ─── Config (match training) ───
MODEL_PATH = "Qwen/Qwen3-30B-A3B"
CHECKPOINT_DIR = "./logs/MoE_Qwen3_30B_A3B_Reasoning_QLoRA_r16_a32/26_05_2026/version_1/model_pretrained/00-0.0000-1.0000"
RANK = 16
ALPHA = 32
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]


def load_training_model():
    """Load model with LoRA applied (simulates training model state)."""
    # Import training script's apply_lora and model class
    # We inline the necessary classes here to avoid import issues
    import math

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
            A = self.lora.A
            B = self.lora.B
            return (self.linear.weight + self.lora.scaling * (B.T @ A.T)).to(self.linear.weight.dtype)

        def forward(self, x):
            return self.linear(x) + self.lora(x)

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
            )
        if hasattr(experts_module, '_down_lora'):
            lora = experts_module._down_lora
            experts_module._unsloth_lora_down_proj = (
                lora.A.transpose(1, 2), lora.B.transpose(1, 2), lora.scaling
            )

    def _apply_expert_lora(model, rank, alpha):
        for _, module in model.named_modules():
            attached = False
            if hasattr(module, 'gate_up_proj'):
                p = module.gate_up_proj
                if isinstance(p, nn.Parameter) and p.dim() == 3:
                    num_expert, out_dim, in_dim = p.shape
                    lora = LoRA3DLayer(num_expert, in_dim, out_dim, rank, alpha).to(p.device)
                    module.add_module('_gate_up_lora', lora)
                    attached = True
            if hasattr(module, 'down_proj'):
                p = module.down_proj
                if isinstance(p, nn.Parameter) and p.dim() == 3:
                    num_expert, out_dim, in_dim = p.shape
                    lora = LoRA3DLayer(num_expert, in_dim, out_dim, rank, alpha).to(p.device)
                    module.add_module('_down_lora', lora)
                    attached = True
            if attached:
                module.register_forward_pre_hook(_expert_lora_hook)

    # Load base model
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_PATH, max_seq_length=256, dtype=torch.bfloat16, load_in_4bit=False,
    )

    # Wrap in custom class
    class Qwen3MoECustom(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.model = base_model.model
            self.lm_head = base_model.lm_head

        @property
        def lm_head_weight(self):
            return self.lm_head.weight

        def forward(self, input_ids):
            hidden = self.model.embed_tokens(input_ids)
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            cos, sin = self.model.rotary_emb(hidden, position_ids)
            for layer in self.model.layers:
                hidden = layer(hidden, position_embeddings=(cos, sin))
            hidden = self.model.norm(hidden)
            return hidden

    model = Qwen3MoECustom(base_model)

    # Apply LoRA
    for param in model.parameters():
        param.requires_grad = False
    _replace_linear_with_lora(model, RANK, ALPHA, TARGET_MODULES)
    _apply_expert_lora(model, RANK, ALPHA)

    return model, tokenizer


def load_checkpoint_into_fresh_model(checkpoint_dir):
    """Load merged checkpoint into a fresh base model (simulates vLLM loading)."""
    from safetensors import safe_open

    # Load base model (fresh, no LoRA)
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_PATH, max_seq_length=256, dtype=torch.bfloat16, load_in_4bit=False,
    )

    class Qwen3MoEFresh(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.model = base_model.model
            self.lm_head = base_model.lm_head

        def forward(self, input_ids):
            hidden = self.model.embed_tokens(input_ids)
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            cos, sin = self.model.rotary_emb(hidden, position_ids)
            for layer in self.model.layers:
                hidden = layer(hidden, position_embeddings=(cos, sin))
            hidden = self.model.norm(hidden)
            return hidden

    model = Qwen3MoEFresh(base_model)

    # Load merged checkpoint weights
    import json
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    # Gather all shard files
    shard_files = set(index['weight_map'].values())

    # Load all tensors
    checkpoint_sd = {}
    for shard in shard_files:
        shard_path = os.path.join(checkpoint_dir, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                checkpoint_sd[key] = f.get_tensor(key)

    # Load into model — need to handle the 3D expert packing
    # The checkpoint has individual expert keys, but the fresh model has 3D packed params
    loaded_count = 0
    missing_keys = []

    # Build a dict of model's named parameters for direct mapping
    model_params = dict(model.named_parameters())

    # First handle non-expert params
    for key, tensor in checkpoint_sd.items():
        if 'mlp.experts.' in key and any(x in key for x in ['gate_proj', 'up_proj', 'down_proj']):
            continue  # Handle experts separately
        if key in model_params:
            model_params[key].data.copy_(tensor.to(model_params[key].dtype))
            loaded_count += 1
        else:
            missing_keys.append(key)

    # Handle expert weights: pack individual keys back into 3D
    import re
    for name, module in model.named_modules():
        if not (hasattr(module, 'gate_up_proj') and isinstance(module.gate_up_proj, nn.Parameter) and module.gate_up_proj.dim() == 3):
            continue

        num_experts = module.gate_up_proj.shape[0]
        half = module.gate_up_proj.shape[1] // 2

        for e in range(num_experts):
            gate_key = f"{name}.{e}.gate_proj.weight"
            up_key = f"{name}.{e}.up_proj.weight"
            down_key = f"{name}.{e}.down_proj.weight"

            if gate_key in checkpoint_sd and up_key in checkpoint_sd:
                module.gate_up_proj.data[e, :half, :] = checkpoint_sd[gate_key].to(module.gate_up_proj.dtype)
                module.gate_up_proj.data[e, half:, :] = checkpoint_sd[up_key].to(module.gate_up_proj.dtype)
                loaded_count += 2

            if down_key in checkpoint_sd:
                module.down_proj.data[e] = checkpoint_sd[down_key].to(module.down_proj.dtype)
                loaded_count += 1

    print(f"Loaded {loaded_count} weight tensors into fresh model")
    if missing_keys:
        print(f"Missing keys (not loaded): {missing_keys[:10]}...")

    return model, tokenizer


def compare_models():
    """Main diagnostic: compare training model vs checkpoint-loaded model."""
    print("=" * 70)
    print("CHECKPOINT VERIFICATION DIAGNOSTIC")
    print("=" * 70)

    # Step 1: Load training model (with LoRA, random init — we'll load checkpoint weights)
    print("\n[1/4] Loading training model structure...")
    train_model, tokenizer = load_training_model()

    # Step 2: Build merged state dict from training model
    # NOTE: Since LoRA is freshly initialized (random A, zero B), the merged weights
    # equal base weights (B=0 → delta=0). We need to verify against the SAVED checkpoint.
    # Instead, let's directly compare: checkpoint values vs base model values.
    # If they differ, the LoRA was trained and merged. If they don't differ, LoRA had no effect.
    print("\n[2/4] Loading checkpoint...")
    import json
    from safetensors import safe_open

    index_path = os.path.join(CHECKPOINT_DIR, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    shard_files = set(index['weight_map'].values())
    checkpoint_sd = {}
    for shard in shard_files:
        shard_path = os.path.join(CHECKPOINT_DIR, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                checkpoint_sd[key] = f.get_tensor(key)

    # Step 3: Compare checkpoint weights against base model weights
    print("\n[3/4] Comparing checkpoint vs base model weights...")
    print("-" * 50)

    # Get base model weights (before any LoRA)
    base_model_fresh, _ = FastLanguageModel.from_pretrained(
        MODEL_PATH, max_seq_length=256, dtype=torch.bfloat16, load_in_4bit=False,
    )

    # Check lm_head
    base_lm_head = base_model_fresh.lm_head.weight.data.cpu()
    ckpt_lm_head = checkpoint_sd['lm_head.weight']
    lm_head_diff = (ckpt_lm_head.float() - base_lm_head.float()).abs()
    print(f"lm_head.weight:")
    print(f"  Max diff from base: {lm_head_diff.max().item():.6f}")
    print(f"  Mean diff from base: {lm_head_diff.mean().item():.6f}")
    print(f"  Changed: {'YES ✓' if lm_head_diff.max() > 1e-6 else 'NO ✗ (LoRA had no effect!)'}")

    # Check attention projections (layer 0)
    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        ckpt_key = f"model.layers.0.self_attn.{proj}.weight"
        if ckpt_key in checkpoint_sd:
            # Get base weight from the fresh model
            base_w = None
            for name, param in base_model_fresh.named_parameters():
                if f"layers.0.self_attn.{proj}" in name and 'weight' in name:
                    base_w = param.data.cpu()
                    break
            if base_w is not None:
                diff = (checkpoint_sd[ckpt_key].float() - base_w.float()).abs()
                changed = diff.max() > 1e-6
                print(f"Layer 0 {proj}: Max diff={diff.max().item():.6f} {'✓' if changed else '✗'}")

    # Check a few expert weights
    print(f"\nExpert weights (layer 0):")
    for name, module in base_model_fresh.named_modules():
        if 'layers.0.mlp.experts' in name and hasattr(module, 'gate_up_proj'):
            p = module.gate_up_proj
            if isinstance(p, nn.Parameter) and p.dim() == 3:
                half = p.shape[1] // 2
                # Expert 0 gate_proj
                base_gate = p.data[0, :half, :].cpu()
                ckpt_gate = checkpoint_sd.get("model.layers.0.mlp.experts.0.gate_proj.weight")
                if ckpt_gate is not None:
                    diff = (ckpt_gate.float() - base_gate.float()).abs()
                    print(f"  Expert 0 gate_proj: Max diff={diff.max().item():.6f} {'✓' if diff.max() > 1e-6 else '✗'}")

                # Expert 0 down_proj
                base_down = module.down_proj.data[0].cpu()
                ckpt_down = checkpoint_sd.get("model.layers.0.mlp.experts.0.down_proj.weight")
                if ckpt_down is not None:
                    diff = (ckpt_down.float() - base_down.float()).abs()
                    print(f"  Expert 0 down_proj: Max diff={diff.max().item():.6f} {'✓' if diff.max() > 1e-6 else '✗'}")
                break

    # Step 4: Forward pass comparison
    print("\n[4/4] Forward pass comparison (checkpoint-loaded vs base)...")
    print("-" * 50)

    # Load checkpoint into a fresh model and run forward
    test_text = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n"
    input_ids = tokenizer.encode(test_text, return_tensors='pt').to(base_model_fresh.device)
    print(f"Test input: {input_ids.shape[1]} tokens")

    # Forward through base model (no LoRA merged)
    with torch.no_grad():
        base_output = base_model_fresh(input_ids)
        if hasattr(base_output, 'logits'):
            base_logits = base_output.logits
        else:
            # Manual: get hidden states and apply lm_head
            base_hidden = base_output[0] if isinstance(base_output, tuple) else base_output.last_hidden_state
            base_logits = base_model_fresh.lm_head(base_hidden)

    # Load checkpoint into fresh model and forward
    ckpt_model, _ = load_checkpoint_into_fresh_model(CHECKPOINT_DIR)
    ckpt_model.eval()
    with torch.no_grad():
        ckpt_hidden = ckpt_model(input_ids.to(next(ckpt_model.parameters()).device))
        ckpt_logits = ckpt_model.lm_head(ckpt_hidden)

    # Compare logits
    # Move to same device for comparison
    base_logits_cpu = base_logits[0, -1, :].float().cpu()
    ckpt_logits_cpu = ckpt_logits[0, -1, :].float().cpu()

    logit_diff = (ckpt_logits_cpu - base_logits_cpu).abs()
    print(f"\nLast-token logit comparison (checkpoint vs base):")
    print(f"  Max diff: {logit_diff.max().item():.6f}")
    print(f"  Mean diff: {logit_diff.mean().item():.6f}")

    if logit_diff.max() < 1e-4:
        print("\n⚠️  WARNING: Checkpoint logits are IDENTICAL to base model!")
        print("    This means LoRA training had NO EFFECT on the saved checkpoint.")
        print("    Possible causes:")
        print("    - The checkpoint was saved before training started")
        print("    - The merge function didn't actually merge LoRA deltas")
        print("    - The LoRA params were all zeros (B initialized to zeros, never trained)")
    else:
        print(f"\n✓ Checkpoint differs from base model (LoRA was merged)")

    # Show top predictions from both
    base_top5 = base_logits_cpu.topk(5)
    ckpt_top5 = ckpt_logits_cpu.topk(5)
    print(f"\nBase model top-5 next tokens:")
    for i in range(5):
        token = tokenizer.decode([base_top5.indices[i].item()])
        print(f"  {base_top5.values[i].item():.4f}: '{token}' (id={base_top5.indices[i].item()})")
    print(f"\nCheckpoint model top-5 next tokens:")
    for i in range(5):
        token = tokenizer.decode([ckpt_top5.indices[i].item()])
        print(f"  {ckpt_top5.values[i].item():.4f}: '{token}' (id={ckpt_top5.indices[i].item()})")


if __name__ == "__main__":
    compare_models()
