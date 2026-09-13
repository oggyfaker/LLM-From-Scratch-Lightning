"""
qwen3-vllm.py

Qwen3 dense model integrated with vLLM for high-performance inference.

- Attention uses vLLM's PagedAttention + KV cache
- RoPE uses vLLM's optimized `get_rope`
- Linear layers use vLLM's parallel variants
- Forward accepts flattened tensors (no batch dim during inference)
- Implements `load_weights` for vLLM's weight loading pipeline
- Registered via ModelRegistry for out-of-tree integration

Usage:
    # Direct creation
    model = Qwen3vLLM(QWEN_14B_CFG())

    # vLLM inference
    python qwen3-vllm.py
"""

from dataclasses import dataclass
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention as VllmAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors



# ============================================================
# Grouped Query Attention (vLLM PagedAttention + KV cache)
# ============================================================
class PagedAttention(nn.Module):

    def __init__(self, cfg, layer_idx: int, prefix: str = ""):
        super().__init__()
        self.num_heads = cfg.n_heads
        self.num_kv_heads = cfg.n_kv_groups
        self.head_dim = cfg.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=cfg.emb_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            cfg.emb_dim,
            bias=False,
            prefix=f"{prefix}.o_proj",
        )

        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=cfg.context_length,
            rope_parameters={
                "rope_type": "default",
                "factor": 1.0,
                "base": cfg.rope_base,
            },
        )

        self.attn = VllmAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.head_dim ** -0.5,
            num_kv_heads=self.num_kv_heads,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.q_norm is not None:
            q = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
            q = self.q_norm(q)
            q = q.view(*q.shape[:-2], -1)

        if self.k_norm is not None:
            k = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
            k = self.k_norm(k)
            k = k.view(*k.shape[:-2], -1)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output



# ============================================================
# Feed-Forward Network (SwiGLU)
# ============================================================
class FeedForward(nn.Module):

    def __init__(self, cfg, prefix: str = ""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            cfg.emb_dim,
            [cfg.hidden_dim, cfg.hidden_dim],
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            cfg.hidden_dim,
            cfg.emb_dim,
            bias=False,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        x = self.act_fn(gate_up)
        output, _ = self.down_proj(x)
        return output



# ============================================================
# Transformer Block
# ============================================================
class Block(nn.Module):

    def __init__(self, cfg, layer_idx: int, prefix: str = ""):
        super().__init__()
        self.self_attn = PagedAttention(cfg, layer_idx=layer_idx, prefix=f"{prefix}.self_attn")
        self.mlp = FeedForward(cfg, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(cfg.emb_dim, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim, eps=1e-6)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual



# ============================================================
# Full Model (transformer backbone)
# ============================================================
class Qwen3Model(nn.Module):

    def __init__(self, cfg, prefix: str = ""):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = VocabParallelEmbedding(
            cfg.vocab_size,
            cfg.emb_dim,
            prefix=f"{prefix}.embed_tokens",
        )

        self.layers = nn.ModuleList([
            Block(cfg, layer_idx=i, prefix=f"{prefix}.layers.{i}")
            for i in range(cfg.n_blocks)
        ])

        self.norm = RMSNorm(cfg.emb_dim, eps=1e-6)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states



# ============================================================
# Top-level CausalLM (registered with vLLM) — Base Class
# ============================================================
class Qwen3vLLM(nn.Module):
    """Base vLLM model class for Qwen3 dense models.

    Subclass and set ``_default_cfg_cls`` to the desired config dataclass.
    vLLM reads hf_config from config.json and overrides defaults accordingly.

    Example:
        class Qwen3_14B_vLLM(Qwen3vLLM):
            _default_cfg_cls = QWEN_14B_CFG

        ModelRegistry.register_model("Qwen3_14B_vLLM", Qwen3_14B_vLLM)
    """

    MODEL_CFG = None  # Subclasses must set this (dataclass class, e.g. QWEN_14B_CFG)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # --- Use MODEL_CFG directly (set by subclass) ---
        self.cfg = self.MODEL_CFG()
        self.config = vllm_config.model_config.hf_text_config

        # --- Build model ---
        self.model = Qwen3Model(self.cfg, prefix="model")

        self.lm_head = ParallelLMHead(
            self.cfg.vocab_size,
            self.cfg.emb_dim,
            prefix="lm_head",
        )

        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

        self.logits_processor = LogitsProcessor(self.cfg.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        merged_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            name = self._translate_custom_name(name)

            # Handle stacked QKV weights
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                stacked_name = name.replace(weight_name, param_name)
                if stacked_name not in params_dict:
                    continue
                param = params_dict[stacked_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                is_stacked = True
                loaded_params.add(stacked_name)
                break
            if is_stacked:
                continue

            # Handle merged gate_up weights
            is_merged = False
            for param_name, weight_name, shard_idx in merged_params_mapping:
                if weight_name not in name:
                    continue
                merged_name = name.replace(weight_name, param_name)
                if merged_name not in params_dict:
                    continue
                param = params_dict[merged_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_idx)
                is_merged = True
                loaded_params.add(merged_name)
                break
            if is_merged:
                continue

            # Direct weight
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params

    @staticmethod
    def _translate_custom_name(name: str) -> str:
        if name.startswith("model.") or name.startswith("lm_head."):
            return name

        if name == "tok_emb.weight":
            return "model.embed_tokens.weight"
        if name == "final_norm.scale":
            return "model.norm.weight"
        if name == "out_head.weight":
            return "lm_head.weight"

        if name.startswith("transformer_blocks."):
            parts = name.split(".")
            layer_idx = parts[1]
            rest = ".".join(parts[2:])

            rest = rest.replace("att.W_query.weight", "self_attn.q_proj.weight")
            rest = rest.replace("att.W_key.weight", "self_attn.k_proj.weight")
            rest = rest.replace("att.W_value.weight", "self_attn.v_proj.weight")
            rest = rest.replace("att.out_proj.weight", "self_attn.o_proj.weight")
            rest = rest.replace("att.q_norm.scale", "self_attn.q_norm.weight")
            rest = rest.replace("att.k_norm.scale", "self_attn.k_norm.weight")
            rest = rest.replace("norm1.scale", "input_layernorm.weight")
            rest = rest.replace("norm2.scale", "post_attention_layernorm.weight")
            rest = rest.replace("ff.fc1.weight", "mlp.gate_proj.weight")
            rest = rest.replace("ff.fc2.weight", "mlp.up_proj.weight")
            rest = rest.replace("ff.fc3.weight", "mlp.down_proj.weight")

            return f"model.layers.{layer_idx}.{rest}"

        return name


# ============================================================
# Model Configuration
# ============================================================
from .qwen3_dense import (
    QWEN_06B_CFG, QWEN_1B7_CFG, QWEN_4B_CFG, 
    QWEN_8B_CFG, QWEN_14B_CFG, QWEN_32B_CFG
)

class Qwen3_06B_vLLM(Qwen3vLLM):
    MODEL_CFG = QWEN_06B_CFG

class Qwen3_1B7_vLLM(Qwen3vLLM):
    MODEL_CFG = QWEN_1B7_CFG

class Qwen3_4B_vLLM(Qwen3vLLM):
    MODEL_CFG = QWEN_4B_CFG

class Qwen3_8B_vLLM(Qwen3vLLM):
    MODEL_CFG = QWEN_8B_CFG

class Qwen3_14B_vLLM(Qwen3vLLM):
    MODEL_CFG = QWEN_14B_CFG

class Qwen3_32B_vLLM(Qwen3vLLM):
    MODEL_CFG = QWEN_32B_CFG
