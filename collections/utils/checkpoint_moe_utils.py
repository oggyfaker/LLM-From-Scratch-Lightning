import os
import json
import shutil

import torch
import bitsandbytes as bnb
from safetensors.torch import save_file

try:
    import pytorch_lightning as pl
    _PL_CALLBACK_BASE = pl.Callback
except ImportError:
    pl = None
    class _PL_CALLBACK_BASE:  # no-op shim so class definitions parse in non-training envs
        pass


# -----
# MERGE MOE LORA WEIGHTS INTO BASE MODEL (Unsloth / HuggingFace format)
# -----

def merge_moe_lora_state_dict(model):
    """Build a merged state_dict for Qwen3-MoE model with custom LoRA.

    Handles two types of LoRA:
      1. LinearWithLoRA (2D): attention q/k/v/o_proj, shared_expert gate/up/down
         Merge: W_new = W + scaling * (A @ B).T
      2. ExpertLoRA (3D): packed expert gate_up_proj / down_proj via hooks
         Merge: W_new = W + scaling * B @ A
         Then unpack 3D → individual expert keys for HuggingFace format.

    Returns:
        dict: HuggingFace-compatible state_dict with individual expert keys.
              Keys like: model.layers.{i}.mlp.experts.{e}.gate_proj.weight
    """
    merged = {}
    handled_prefixes = set()
    expert_3d_params = set()  # Track 3D params handled by ExpertLoRA

    # 1. Merge LinearWithLoRA modules (2D: attention + shared_expert)
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

            # LoRA forward: scaling * (x @ A @ B)
            # Linear: x @ W.T → Merged: W + scaling * (A @ B).T
            A = module.lora.A.data   # (in_dim, rank)
            B = module.lora.B.data   # (rank, out_dim)
            lora_delta = module.lora.scaling * (A @ B).T  # (out_dim, in_dim)
            merged[name + ".weight"] = (w + lora_delta.to(w.dtype)).cpu()

    # 2. Merge ExpertLoRA (3D) and unpack to individual expert keys
    for name, module in model.named_modules():
        has_expert_lora = hasattr(module, '_gate_up_lora') or hasattr(module, '_down_lora')
        if not has_expert_lora:
            continue

        # gate_up_proj: packed [E, 2*moe_inter, hidden]
        if hasattr(module, 'gate_up_proj') and hasattr(module, '_gate_up_lora'):
            base_param = module.gate_up_proj
            expert_3d_params.add(id(base_param))

            lora = module._gate_up_lora
            # A: (E, rank, in_dim), B: (E, out_dim, rank)
            # Merge: W_new = W + scaling * B @ A → (E, out_dim, in_dim)
            with torch.no_grad():
                delta = lora.scaling * torch.bmm(lora.B.data, lora.A.data)
            w = (base_param.data + delta.to(base_param.dtype)).cpu()

            # Unpack: gate_up_proj[e] = [gate_proj[e]; up_proj[e]] along dim 0
            num_experts = w.shape[0]
            half = w.shape[1] // 2
            for e in range(num_experts):
                gate_w = w[e, :half, :]  # (moe_inter, hidden)
                up_w = w[e, half:, :]    # (moe_inter, hidden)
                expert_key = f"{name}.{e}.gate_proj.weight"
                merged[expert_key] = gate_w
                expert_key = f"{name}.{e}.up_proj.weight"
                merged[expert_key] = up_w

        # down_proj: packed [E, hidden, moe_inter]
        if hasattr(module, 'down_proj') and hasattr(module, '_down_lora'):
            base_param = module.down_proj
            expert_3d_params.add(id(base_param))

            lora = module._down_lora
            with torch.no_grad():
                delta = lora.scaling * torch.bmm(lora.B.data, lora.A.data)
            w = (base_param.data + delta.to(base_param.dtype)).cpu()

            num_experts = w.shape[0]
            for e in range(num_experts):
                expert_key = f"{name}.{e}.down_proj.weight"
                merged[expert_key] = w[e]  # (hidden, moe_inter)

    # 3. Copy all remaining parameters (embeddings, norms, router, etc.)
    for pname, param in model.named_parameters():
        if any(pname.startswith(p) for p in handled_prefixes):
            continue
        if id(param) in expert_3d_params:
            continue
        # Skip LoRA-only parameters (they're merged into base)
        if '_gate_up_lora' in pname or '_down_lora' in pname:
            continue
        merged[pname] = param.data.cpu()

    return merged


def save_sharded_safetensors(state_dict, output_dir, max_shard_size_gb=5.0):
    """Save a state_dict as sharded safetensors files.

    Args:
        state_dict: dict of tensor name → tensor
        output_dir: directory to save shard files + index
        max_shard_size_gb: max size per shard in GB
    """
    os.makedirs(output_dir, exist_ok=True)
    max_shard_bytes = int(max_shard_size_gb * 1e9)

    # Sort keys for deterministic sharding
    sorted_keys = sorted(state_dict.keys())

    # Build shards
    shards = []
    current_shard = {}
    current_size = 0

    for key in sorted_keys:
        tensor = state_dict[key]
        tensor_bytes = tensor.numel() * tensor.element_size()

        if current_size + tensor_bytes > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_bytes

    if current_shard:
        shards.append(current_shard)

    # Save each shard
    total_size = 0
    weight_map = {}

    for i, shard in enumerate(shards, 1):
        num_shards = len(shards)
        shard_name = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        shard_path = os.path.join(output_dir, shard_name)
        save_file(shard, shard_path)

        for key, tensor in shard.items():
            weight_map[key] = shard_name
            total_size += tensor.numel() * tensor.element_size()

    # Save index file
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Saved {len(shards)} shards to {output_dir} ({total_size / 1e9:.2f} GB)")


class MoELoRAMergeCheckpoint(_PL_CALLBACK_BASE):
    """Save merged MoE model as sharded safetensors after validation.

    Merges LoRA weights into base, unpacks 3D expert params to individual expert
    keys, and saves in HuggingFace-compatible sharded safetensors format.
    Also saves config.json and tokenizer files for direct vLLM loading.
    """

    def __init__(self, tokenizer,
                 monitor='Validation/accuracy', mode='max', save_top_k=3,
                 filename_template='{epoch:02d}-{Validation/loss:.4f}-{Validation/accuracy:.4f}'):
        super().__init__()
        self.tokenizer = tokenizer
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename_template = filename_template
        self._saved = []  # list of (score, path)

    def on_validation_end(self, trainer, pl_module):
        """Export the merged model; never take the training run down with it.

        This runs mid-training, and Lightning invokes plain Callbacks before
        ModelCheckpoint, so an exception here also pre-empts the ordinary .ckpt
        write and the run dies having saved nothing. A 60 GB merge-and-shard has
        plenty of ways to fail late (disk, a renamed metric in the filename
        template, a layout change in the expert tensors); none of them are worth
        the hours of training already done. The failure is printed in full and
        training continues to the next validation, which gets another attempt.
        """
        try:
            self._save(trainer, pl_module)
        except Exception as e:
            import traceback
            print(f"[merge-ckpt] export FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    def _save(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor)
        if current is None:
            return
        score = current.item() if hasattr(current, 'item') else float(current)

        # Check whether this score deserves saving (top-k)
        if len(self._saved) >= self.save_top_k:
            worst = self._saved[-1][0]
            better = score > worst if self.mode == 'max' else score < worst
            if not better:
                return

        # Build path
        save_dir = os.path.join(trainer.logger.log_dir, "model_pretrained")
        fmt = {
            'epoch': trainer.current_epoch,
            'step': trainer.global_step,
        }
        for k, v in metrics.items():
            fmt[k] = v.item() if hasattr(v, 'item') else float(v)
        dirname = self.filename_template.format(**fmt)
        output_dir = os.path.join(save_dir, dirname)

        # Merge LoRA → base and unpack experts
        # pl_module.model is Qwen3MoeForCausalLM with apply_lora injected directly
        merged_sd = merge_moe_lora_state_dict(pl_module.model)

        # Save sharded safetensors
        save_sharded_safetensors(merged_sd, output_dir)

        # Save config.json from the live model (no HuggingFace download needed)
        config_dst = os.path.join(output_dir, "config.json")
        config = pl_module.model.config.to_dict()
        config["architectures"] = ["Qwen3MoeForCausalLM"]
        with open(config_dst, "w") as f:
            json.dump(config, f, indent=2)

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)

        print(f"Merged MoE model saved to: {output_dir}")

        # Maintain top-k list
        self._saved.append((score, output_dir))
        self._saved.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))

        while len(self._saved) > self.save_top_k:
            _, old_path = self._saved.pop()
            if os.path.isdir(old_path):
                shutil.rmtree(old_path)
                print(f"Removed old merged model: {old_path}")

    @property
    def best_model_path(self):
        if self._saved:
            return self._saved[0][1]
        return None


# -----
# NEMOTRON: MERGE 2D LORA WEIGHTS INTO BASE MODEL
# -----

def merge_nemotron_lora_state_dict(model):
    """Build a merged state_dict for Nemotron model with custom 2D LinearWithLoRA.

    All LoRA in Nemotron is standard 2D LinearWithLoRA (no 3D packed expert tensors):
      - Mamba layers   : in_proj, out_proj
      - Attention      : q_proj, k_proj, v_proj, o_proj
      - MLP / MoE      : up_proj, down_proj (per-expert nn.Linear, handled recursively)
      - Shared expert  : up_proj, down_proj
      - Output         : lm_head

    Returns:
        HuggingFace-compatible state_dict.
    """
    merged = {}
    handled_prefixes = set()

    # Merge all LinearWithLoRA modules (2D: attention, Mamba, MLP, per-expert, lm_head)
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

            A = module.lora.A.data   # [in_dim, rank]
            B = module.lora.B.data   # [rank, out_dim]
            lora_delta = module.lora.scaling * (A @ B).T   # [out_dim, in_dim]
            merged[name + ".weight"] = (w + lora_delta.to(w.dtype)).cpu()

            if hasattr(base, 'bias') and base.bias is not None:
                merged[name + ".bias"] = base.bias.data.cpu()

    # Copy all remaining parameters (norms, embeddings, Mamba state params, router, etc.)
    for pname, param in model.named_parameters():
        if any(pname.startswith(p) for p in handled_prefixes):
            continue
        merged[pname] = param.data.cpu()

    return merged


class NemotronLoRAMergeCheckpoint(_PL_CALLBACK_BASE):
    """Save merged Nemotron model as sharded safetensors after validation.

    Merges all 2D LinearWithLoRA weights into base, and saves in
    HuggingFace-compatible sharded safetensors format with config.json
    and tokenizer files for direct vLLM loading.
    """

    def __init__(self, tokenizer,
                 monitor='Validation/accuracy', mode='max', save_top_k=1,
                 filename_template='{epoch:02d}-{Validation/loss:.4f}-{Validation/accuracy:.4f}'):
        super().__init__()
        self.tokenizer = tokenizer
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename_template = filename_template
        self._saved = []

    def on_validation_end(self, trainer, pl_module):
        """Export the merged model; never take the training run down with it.

        This runs mid-training, and Lightning invokes plain Callbacks before
        ModelCheckpoint, so an exception here also pre-empts the ordinary .ckpt
        write and the run dies having saved nothing. A 60 GB merge-and-shard has
        plenty of ways to fail late (disk, a renamed metric in the filename
        template, a layout change in the expert tensors); none of them are worth
        the hours of training already done. The failure is printed in full and
        training continues to the next validation, which gets another attempt.
        """
        try:
            self._save(trainer, pl_module)
        except Exception as e:
            import traceback
            print(f"[merge-ckpt] export FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    def _save(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor)
        if current is None:
            return
        score = current.item() if hasattr(current, 'item') else float(current)

        if len(self._saved) >= self.save_top_k:
            worst = self._saved[-1][0]
            better = score > worst if self.mode == 'max' else score < worst
            if not better:
                return

        save_dir = os.path.join(trainer.logger.log_dir, "model_pretrained")
        fmt = {
            'epoch': trainer.current_epoch,
            'step': trainer.global_step,
        }
        for k, v in metrics.items():
            fmt[k] = v.item() if hasattr(v, 'item') else float(v)
        dirname = self.filename_template.format(**fmt)
        output_dir = os.path.join(save_dir, dirname)

        merged_sd = merge_nemotron_lora_state_dict(pl_module.model)
        save_sharded_safetensors(merged_sd, output_dir)

        config_dst = os.path.join(output_dir, "config.json")
        config = pl_module.model.config.to_dict()
        config["architectures"] = ["NemotronHForCausalLM"]
        with open(config_dst, "w") as f:
            json.dump(config, f, indent=2)

        self.tokenizer.save_pretrained(output_dir)

        print(f"Merged Nemotron model saved to: {output_dir}")

        self._saved.append((score, output_dir))
        self._saved.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))

        while len(self._saved) > self.save_top_k:
            _, old_path = self._saved.pop()
            if os.path.isdir(old_path):
                shutil.rmtree(old_path)
                print(f"Removed old merged Nemotron model: {old_path}")

    @property
    def best_model_path(self):
        if self._saved:
            return self._saved[0][1]
        return None
