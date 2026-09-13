import torch 
import torch.nn as nn

import json
import requests
from pathlib import Path
from dataclasses import dataclass
from safetensors.torch import load_file


# ------- RMS Normalization -------
class RMSNorm(nn.Module):
    def __init__(
        self,
        emb_dim,
        eps=1e-6,
        bias=False,
        qwen3_compatible=True,
    ):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


# ------- Rotation Position Embeddings (RoPE) -------
def compute_rope_params(
    head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32
):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    inv_freq = 1.0 / (theta_base ** (
        torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float()
        / head_dim
    ))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin, offset=0):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2:]  # Second half

    # Adjust sin and cos shapes, shape: (1, 1, seq_len, head_dim)
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)  
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


# ------- Gated Linear Units (GLU) -------
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(
            cfg.emb_dim, cfg.hidden_dim, dtype=cfg.dtype,
            bias=False
        )
        self.fc2 = nn.Linear(
            cfg.emb_dim, cfg.hidden_dim, dtype=cfg.dtype,
            bias=False
        )
        self.fc3 = nn.Linear(
            cfg.hidden_dim, cfg.emb_dim, dtype=cfg.dtype,
            bias=False
        )

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


# ------- Group Query Attention (GQA) -------
class GroupedQueryAttention(nn.Module):
    '''The facts 
    The big question: Do we always need to train a GQA model from scratch? 
    The answer is no.
    
    The original GQA paper showed that we can take an existing Multi-Head Attention model and convert it into a GQA model very cheaply. 
    This is called uptraining.

    Step 1: Take an existing MHA model that has already been trained.
    
    Step 2: For each group, take all the Key weight matrices of the heads in that group and average them. 
    Do the same for the Value weight matrices. 
    This gives us one shared Key and one shared Value per group.
    
    Step 3: Fine-tune the model for a short time. The original paper showed that uptraining with just around 5% of the original pre-training compute is enough to recover quality close to full MHA.
    This is one of the main reasons GQA was adopted so quickly. 
    Labs did not have to throw away their existing MHA models or spend huge compute to train new ones. 
    They could just uptrain them into GQA.
    '''
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, cos, sin):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # --- Causal mask computed here inside GQA ---
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        mask = mask[None, None, :, :]  # (1, 1, num_tokens, num_tokens)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)
    

# ------- Qwen3 BLock -------
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg.emb_dim,
            num_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            num_kv_groups=cfg.n_kv_groups,
            qk_norm=cfg.qk_norm,
            dtype=cfg.dtype
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg.emb_dim, eps=1e-6)
        self.norm2 = RMSNorm(cfg.emb_dim, eps=1e-6)

    def forward(self, x, cos, sin):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, cos, sin)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x


# ------- Qwen3 Model -------
class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=cfg.dtype)
        self.transformer_blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_blocks)])
        self.final_norm = RMSNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)

        if cfg.head_dim is None:
            head_dim = cfg.emb_dim // cfg.n_heads
        else:
            head_dim = cfg.head_dim

        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg.rope_base,
            context_length=cfg.context_length
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.dtype = cfg.dtype

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)

        for block in self.transformer_blocks:
            x = block(x, self.cos, self.sin)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.dtype))
        
        return logits


# ------- Loading from local merged .pth (post LoRA-merge) -------
def from_local_pth(model, pth_path):
    """Load a merged .pth state_dict (from MergedModelCheckpoint) into a Qwen3Model.

    The .pth file uses the custom qwen3_dense.py naming convention:
        tok_emb.weight, transformer_blocks.{l}.att.W_query.weight, etc.

    Args:
        model: Qwen3Model instance (already initialized with correct config).
        pth_path: Path to the merged .pth file.

    Returns:
        model
    """
    print(f"Loading merged model from: {pth_path}")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    print(f"✓ Loaded {len(state_dict)} parameters from {pth_path}")

    return model


# ------- Loading pretrained huggingface -------
def from_pretrained(
    model=None,
    repo_id="Qwen/Qwen3-0.6B",
    model_ckpt="model.safetensors",
):

    # Download files from HuggingFace
    out_dir = Path("checkpoint") / repo_id.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True) # Get folder name local saving (e.g., "Qwen/Qwen3-0.6B" -> "checkpoint/Qwen3-0.6B")

    base_url = f"https://huggingface.co/{repo_id}/resolve/main"

    def _download_file(fname):
        """Download a single file from HuggingFace repo to out_dir."""
        dest = out_dir / fname
        url = f"{base_url}/{fname}"

        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            size_remote = int(r.headers.get("Content-Length", 0))
            if dest.exists() and size_remote and dest.stat().st_size == size_remote:
                print(f"✓ {dest} already up-to-date")
                return

            # Download with progress
            block = 1024 * 1024
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=block):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if size_remote:
                        pct = downloaded * 100 // size_remote
                        print(f"\r{fname}: {pct:3d}% ({downloaded // (1024*1024)} MiB / {size_remote // (1024*1024)} MiB)", end="", flush=True)
            if size_remote:
                print()
        print(f"✓ Downloaded {fname}")

    # Determine if model is sharded (multiple safetensors files) or single file
    index_fname = "model.safetensors.index.json"
    index_url = f"{base_url}/{index_fname}"
    r = requests.head(index_url, timeout=30, allow_redirects=True)

    if r.status_code == 200:
        # Sharded model: download index, then all shard files
        _download_file(index_fname)
        with open(out_dir / index_fname) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        print(f"Sharded model: {len(shard_files)} files to download")
        for shard in shard_files:
            _download_file(shard)
    else:
        # Single file model
        _download_file(model_ckpt)
        shard_files = [model_ckpt]

    # Load safetensors weights into model
    weights = {}
    for shard in shard_files:
        weights.update(load_file(str(out_dir / shard)))

    with torch.no_grad():
        model.tok_emb.weight.copy_(weights["model.embed_tokens.weight"])

        for l in range(len(model.transformer_blocks)):
            block = model.transformer_blocks[l]        
            att = block.att
            att.W_query.weight.copy_(weights[f"model.layers.{l}.self_attn.q_proj.weight"])
            att.W_key.weight.copy_(weights[f"model.layers.{l}.self_attn.k_proj.weight"])
            att.W_value.weight.copy_(weights[f"model.layers.{l}.self_attn.v_proj.weight"])
            att.out_proj.weight.copy_(weights[f"model.layers.{l}.self_attn.o_proj.weight"])

            if att.q_norm is not None:
                att.q_norm.scale.copy_(weights[f"model.layers.{l}.self_attn.q_norm.weight"])
            if att.k_norm is not None:
                att.k_norm.scale.copy_(weights[f"model.layers.{l}.self_attn.k_norm.weight"])

            block.norm1.scale.copy_(weights[f"model.layers.{l}.input_layernorm.weight"])
            block.ff.fc1.weight.copy_(weights[f"model.layers.{l}.mlp.gate_proj.weight"])
            block.ff.fc2.weight.copy_(weights[f"model.layers.{l}.mlp.up_proj.weight"])
            block.ff.fc3.weight.copy_(weights[f"model.layers.{l}.mlp.down_proj.weight"])
            block.norm2.scale.copy_(weights[f"model.layers.{l}.post_attention_layernorm.weight"])

        model.final_norm.scale.copy_(weights["model.norm.weight"])
        if "lm_head.weight" in weights:
            model.out_head.weight.copy_(weights["lm_head.weight"])
        else:
            model.out_head.weight.copy_(model.tok_emb.weight)
            print("Model uses weight tying.")

    print(f"Model loaded from: {out_dir}/")
    return model


# ------- Inference model -------
def generate(
    model,
    tokenizer,
    prompt,
    max_length=256,
    num_sequences=1,
    temperature=1.0,
    top_k=0,
    eos_token_id=None,
):
    device = next(model.parameters()).device
    model.eval()

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    results = []
    for _ in range(num_sequences):
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(max_length):
                logits = model(input_ids)
                next_logits = logits[:, -1, :]  # (1, vocab_size)

                # Greedy decoding
                if temperature == 0.0:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    # Apply temperature
                    if temperature != 1.0:
                        next_logits = next_logits / temperature

                    # Apply top-k filtering
                    if top_k > 0:
                        top_values, _ = torch.topk(next_logits, top_k, dim=-1)
                        min_top = top_values[:, -1].unsqueeze(-1)
                        next_logits = next_logits.masked_fill(next_logits < min_top, -torch.inf)

                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=1)

                if next_token.item() == eos_token_id:
                    break

        output_ids = input_ids[0].tolist()
        results.append(tokenizer.decode(output_ids))

    return results if num_sequences > 1 else results[0]


@dataclass 
class QWEN_06B_CFG:
    vocab_size: int = 151_936    # Vocabulary size
    context_length: int = 40_960 # Context length that was used to train the model

    n_blocks: int = 28           # Number of blocks transformer
    emb_dim: int = 1024          # Embedding dimension (In-feat FFN)
    hidden_dim: int= 3072        # Size of the intermediate dimension in FeedForward (Out-feat FFN)
    
    n_heads: int = 16            # Number of attention heads
    head_dim: int = 128          # Size of the heads in GQA
    qk_norm: bool = True         # Whether to normalize queries and keys in GQA
    n_kv_groups: int = 8         # Key-Value groups for grouped-query attention
    rope_base: int = 1_000_000.0 # The base in RoPE's "theta"
    
    dtype: torch.dtype = torch.bfloat16 # Lower-precision dtype to reduce memory usage

@dataclass 
class QWEN_1B7_CFG:
    vocab_size: int = 151_936    # Vocabulary size
    context_length: int = 40_960 # Context length that was used to train the model

    n_blocks: int = 28           # Number of blocks transformer
    emb_dim: int = 2048          # Embedding dimension (In-feat FFN)
    hidden_dim: int= 6144        # Size of the intermediate dimension in FeedForward (Out-feat FFN)
    
    n_heads: int = 16            # Number of attention heads
    head_dim: int = 128          # Size of the heads in GQA
    qk_norm: bool = True         # Whether to normalize queries and keys in GQA
    n_kv_groups: int = 8         # Key-Value groups for grouped-query attention
    rope_base: int = 1_000_000.0 # The base in RoPE's "theta"
    
    dtype: torch.dtype = torch.bfloat16 # Lower-precision dtype to reduce memory usage


@dataclass 
class QWEN_4B_CFG:
    vocab_size: int = 151_936    # Vocabulary size
    context_length: int = 40_960 # Context length that was used to train the model

    n_blocks: int = 36           # Number of blocks transformer
    emb_dim: int = 2560          # Embedding dimension (In-feat FFN)
    hidden_dim: int= 9728        # Size of the intermediate dimension in FeedForward (Out-feat FFN)
    
    n_heads: int = 32            # Number of attention heads
    head_dim: int = 128          # Size of the heads in GQA
    qk_norm: bool = True         # Whether to normalize queries and keys in GQA
    n_kv_groups: int = 8         # Key-Value groups for grouped-query attention
    rope_base: int = 1_000_000.0 # The base in RoPE's "theta"
    
    dtype: torch.dtype = torch.bfloat16 # Lower-precision dtype to reduce memory usage


class QWEN_8B_CFG:
    vocab_size: int = 151_936    # Vocabulary size
    context_length: int = 40_960 # Context length that was used to train the model

    n_blocks: int = 36           # Number of blocks transformer
    emb_dim: int = 4096          # Embedding dimension (In-feat FFN)
    hidden_dim: int= 12288        # Size of the intermediate dimension in FeedForward (Out-feat FFN)
    
    n_heads: int = 32            # Number of attention heads
    head_dim: int = 128          # Size of the heads in GQA
    qk_norm: bool = True         # Whether to normalize queries and keys in GQA
    n_kv_groups: int = 8         # Key-Value groups for grouped-query attention
    rope_base: int = 1_000_000.0 # The base in RoPE's "theta"
    
    dtype: torch.dtype = torch.bfloat16 # Lower-precision dtype to reduce memory usage


class QWEN_14B_CFG:
    vocab_size: int = 151_936    # Vocabulary size
    context_length: int = 40_960 # Context length that was used to train the model

    n_blocks: int = 40           # Number of blocks transformer
    emb_dim: int = 5120          # Embedding dimension (In-feat FFN)
    hidden_dim: int= 17408        # Size of the intermediate dimension in FeedForward (Out-feat FFN)
    
    n_heads: int = 40            # Number of attention heads
    head_dim: int = 128          # Size of the heads in GQA
    qk_norm: bool = True         # Whether to normalize queries and keys in GQA
    n_kv_groups: int = 8         # Key-Value groups for grouped-query attention
    rope_base: int = 1_000_000.0 # The base in RoPE's "theta"
    
    dtype: torch.dtype = torch.bfloat16 # Lower-precision dtype to reduce memory usage


class QWEN_32B_CFG:
    vocab_size: int = 151_936    # Vocabulary size
    context_length: int = 40_960 # Context length that was used to train the model

    n_blocks: int = 64           # Number of blocks transformer
    emb_dim: int = 5120          # Embedding dimension (In-feat FFN)
    hidden_dim: int= 25600        # Size of the intermediate dimension in FeedForward (Out-feat FFN)
    
    n_heads: int = 64            # Number of attention heads
    head_dim: int = 128          # Size of the heads in GQA
    qk_norm: bool = True         # Whether to normalize queries and keys in GQA
    n_kv_groups: int = 8         # Key-Value groups for grouped-query attention
    rope_base: int = 1_000_000.0 # The base in RoPE's "theta"
    
    dtype: torch.dtype = torch.bfloat16 # Lower-precision dtype to reduce memory usage


if __name__ == '__main__':
    from qwen_tokenizer import Qwen3Tokenizer

    TOKENIZER_PATH = "collections/qwen3/models/tokenizer.json"

    # Qwen3 - 0.6B 
    model = Qwen3Model(QWEN_06B_CFG)
    model = from_pretrained(model, repo_id="Qwen/Qwen3-0.6B-Base") #Pretraining
    # model = from_pretrained(model, repo_id="Qwen/Qwen3-0.6B") #SFT/RLHF 

    # Qwen3 - 1.7B 
    # model = Qwen3Model(QWEN_1B7_CFG)
    # model = from_pretrained(model, repo_id="Qwen/Qwen3-1.7B-Base") #Pretraining
    # model = from_pretrained(model, repo_id="Qwen/Qwen3-1.7B") #SFT/RLHF 
    
    # Qwen3 - 4B
    # model = Qwen3Model(QWEN_4B_CFG)
    # model = from_pretrained(model, repo_id="Qwen/Qwen3-4B-Base") #Pretraining
    # model = from_pretrained(model, repo_id="Qwen/Qwen3-4B") #SFT/RLHF 

    # Qwen3 - 8B
    # model = Qwen3Model(QWEN_8B_CFG)
    # model = from_pretrained(model, repo_id="Qwen/Qwen3-8B-Base") #Pretraining 
    # model = from_pretrained(model, repo_id="Qwen/Qwen3-8B") #SFT/RLHF 

    # Qwen3 - 14B
    # model = Qwen3Model(QWEN_14B_CFG)
    # model = from_pretrained(model, repo_id="Qwen/Qwen3-14B-Base") #Pretraining
    # model = from_pretrained(model, repo_id="Qwen/Qwen3-14B") #SFT/RLHF 

    # Qwen3 - 32B
    # model = Qwen3Model(QWEN_32B_CFG)
    # model = from_pretrained(model, repo_id="Qwen/Qwen3-32B") #SFT/RLHF 


    # 2. Load tokenizer (same for all Qwen3 model sizes)
    tokenizer = Qwen3Tokenizer(TOKENIZER_PATH)
    model.cuda()

    # 3. Demo 
    prompt = "Explain large language models in a single sentence."
    output = generate(
        model, tokenizer, prompt, 
        max_length=200, temperature=0.0, top_k=0.0
    )