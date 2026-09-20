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



# ------- KV Cache (inference only) -------
class KVCache:
    def __init__(self, n_layers):
        self.keys = [None] * n_layers
        self.values = [None] * n_layers

    def get(self, layer_idx):
        return self.keys[layer_idx], self.values[layer_idx]

    def update(self, layer_idx, keys, values):
        self.keys[layer_idx] = keys
        self.values[layer_idx] = values

    @property
    def seq_len(self):
        return 0 if self.keys[0] is None else self.keys[0].shape[2]

    def reset(self):
        self.keys = [None] * len(self.keys)
        self.values = [None] * len(self.values)


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
    
    Step 3: Fine-tune the model for a short time. The original paper showed that uptraining with just around 5% of the original pre-training 
    compute is enough to recover quality close to full MHA.
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

    def forward(self, x, cos, sin, cache=None, layer_idx=0, start_pos=0, pad_mask=None):
        """
        cache/start_pos/pad_mask are inference-only and default to the training
        behaviour. With all three at their defaults this is the original
        full-sequence causal attention, unchanged.

        start_pos: absolute position of x[:, 0]. RoPE is relative, so a whole
            left-padded row may be shifted by its pad count without changing any
            score -- only differences of positions ever reach the softmax.
        pad_mask: (b, total_keys) bool, True where a slot is left-padding.
        """
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

        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys = apply_rope(keys, cos, sin, offset=start_pos)

        # --- Prepend the positions already decoded (inference only) ---
        if cache is not None:
            past_k, past_v = cache.get(layer_idx)
            if past_k is not None:
                keys = torch.cat([past_k, keys], dim=2)
                values = torch.cat([past_v, values], dim=2)
            cache.update(layer_idx, keys, values)

        if cache is None and pad_mask is None:
            # ---------------- Training path (unchanged) ----------------
            # Expand K and V to match number of heads
            keys = keys.repeat_interleave(self.group_size, dim=1)
            values = values.repeat_interleave(self.group_size, dim=1)

            attn_scores = queries @ keys.transpose(2, 3)

            # --- Causal mask computed here inside GQA ---
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            mask = mask[None, None, :, :]  # (1, 1, num_tokens, num_tokens)

            attn_scores = attn_scores.masked_fill(mask, -torch.inf)
            attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

            context = (attn_weights @ values).transpose(1, 2)
            context = context.reshape(b, num_tokens, self.d_out)
            return self.out_proj(context)

        # ---------------- Generation path ----------------
        # repeat_interleave would copy K and V out to n_heads on EVERY layer of
        # EVERY decode step. At batch 128 / 985 cached positions that is ~124 GB
        # of memory traffic per step -- more than the 4-bit weight dequant it is
        # supposed to be hiding behind. Broadcasting over the group axis instead
        # reads each cached K/V exactly once and allocates nothing.
        #   queries (b, n_heads, Tq, hd) -> (b, n_kv, group_size, Tq, hd)
        #   keys    (b, n_kv,    Tk, hd) -> (b, n_kv, 1,          Tk, hd)
        # Head h of group g is h = g * group_size + s, which is exactly the
        # layout repeat_interleave produces, so the two paths agree.
        total_keys = keys.shape[2]
        q = queries.view(b, self.num_kv_groups, self.group_size, num_tokens, self.head_dim)
        attn_scores = q @ keys.unsqueeze(2).transpose(-2, -1)   # (b, g, gs, Tq, Tk)

        # Keys run 0..total_keys-1 while queries start at start_pos, so the
        # triangle has to be built from absolute positions.
        q_pos = torch.arange(start_pos, start_pos + num_tokens, device=x.device)
        k_pos = torch.arange(total_keys, device=x.device)
        mask = (k_pos[None, :] > q_pos[:, None])[None, None, None, :, :]

        if pad_mask is not None:
            # Real queries must not see padding. Padding queries are left
            # UNMASKED on purpose: blocking every key would leave a row of all
            # -inf, whose softmax is NaN, and 0 * NaN in the value matmul would
            # then poison the real positions too.
            pad_k = pad_mask[:, None, None, None, :]                       # (b,1,1,1,Tk)
            pad_q = pad_mask[:, start_pos:start_pos + num_tokens]           # (b,Tq)
            mask = mask | (pad_k & ~pad_q[:, None, None, :, None])

        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = attn_weights @ values.unsqueeze(2)            # (b, g, gs, Tq, hd)
        context = context.reshape(b, self.num_heads, num_tokens, self.head_dim)
        context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out)
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

    def forward(self, x, cos, sin, cache=None, layer_idx=0, start_pos=0, pad_mask=None):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, cos, sin, cache=cache, layer_idx=layer_idx,
                     start_pos=start_pos, pad_mask=pad_mask)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x


# ------- Top-p (nucleus) sampling -------
def top_p_filter(probas, top_p):
    """Keep the smallest set of tokens whose cumulative mass reaches top_p."""
    if top_p is None or top_p >= 1.0:
        return probas

    sorted_probas, sorted_idx = torch.sort(probas, dim=-1, descending=True)
    prefix = torch.cumsum(sorted_probas, dim=-1) - sorted_probas
    keep = prefix < top_p
    keep[:, 0] = True                       # always keep the most likely token

    kept = torch.where(keep, sorted_probas, torch.zeros_like(sorted_probas))
    filtered = torch.zeros_like(probas).scatter(-1, sorted_idx, kept)
    return filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)


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

    def forward(self, in_idx, cache=None, start_pos=0, pad_mask=None,
                logits_last_only=False):
        """cache/start_pos/pad_mask/logits_last_only are inference-only.

        Training calls model(in_idx) and gets the identical full-sequence
        forward it always did.

        logits_last_only: project only the final position through out_head.
            Greedy decoding reads logits[:, -1] and throws the rest away, but
            out_head is 5120 x 151936 -- at batch 128 with a 217-token prompt the
            discarded part is an 8.4 GB allocation, which is what caps the
            evaluation batch size.
        """
        x = self.tok_emb(in_idx)

        for layer_idx, block in enumerate(self.transformer_blocks):
            x = block(x, self.cos, self.sin, cache=cache, layer_idx=layer_idx,
                      start_pos=start_pos, pad_mask=pad_mask)

        x = self.final_norm(x)
        if logits_last_only:
            x = x[:, -1:, :]
        logits = self.out_head(x.to(self.dtype))

        return logits

    # ------- Batched generation -------
    @torch.no_grad()
    def generate_batch(self, tokenizer, prompts, max_new_tokens=256,
                       max_total_len=None, eos_ids=None, pad_id=None,
                       temperature=0.0, top_p=None):

        # 
        device = next(self.parameters()).device
        eos_ids = {tokenizer.eos_token_id} if eos_ids is None else set(eos_ids)
        enc = [tokenizer.encode(p) for p in prompts]
        B, T = len(enc), max(len(e) for e in enc)

        # Pad - inputs - mask  
        pad_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id
        input_ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
        pad_mask = torch.ones((B, T), dtype=torch.bool, device=device)

        # Left-pad the prompts into one (B, T) block
        for i, ids in enumerate(enc):
            input_ids[i, T - len(ids):] = torch.tensor(ids, dtype=torch.long, device=device)
            pad_mask[i, T - len(ids):] = False

        # Never generate past what training ever saw
        budget = max_new_tokens if max_total_len is None else min(max_new_tokens, max_total_len - T)
        budget = max(1, budget)

        # Autocast mixed precision matching the training 
        amp = torch.autocast(
            device_type=device.type, 
            dtype=torch.bfloat16,
            enabled=device.type == "cuda"
        )

        # KVcache setup  
        cache = KVCache(len(self.transformer_blocks))
        out_ids = [[] for _ in range(B)]
        finished = [False] * B

        # Prefill: one pass over the whole prompt, filling the cache
        with amp:
            logits = self(
                input_ids, cache=cache, start_pos=0, pad_mask=pad_mask, logits_last_only=True
            )[:, -1]

        for steps in range(1, budget + 1):
            if temperature:
                probas = torch.softmax(logits.float() / temperature, dim=-1)
                probas = top_p_filter(probas, top_p)
                next_token = torch.multinomial(probas, num_samples=1).squeeze(-1)
            else:
                next_token = logits.argmax(dim=-1)

            for i, token_id in enumerate(next_token.tolist()):
                if finished[i]:
                    continue
                if token_id in eos_ids:
                    finished[i] = True
                else:
                    out_ids[i].append(token_id)
            if all(finished):
                break

            # Decode: one token per pass, reading the cache
            pad_mask = torch.cat(
                [pad_mask, torch.zeros(B, 1, dtype=torch.bool, device=device)], dim=1)

            with amp:
                logits = self(next_token[:, None], cache=cache,
                              start_pos=pad_mask.shape[1] - 1, pad_mask=pad_mask,
                              logits_last_only=True)[:, -1]

        cache.reset()
        results = [(tokenizer.decode(ids).strip(), "stop" if finished[i] else "length")
                   for i, ids in enumerate(out_ids)]
        return results, steps, budget


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
    top_p=None,
    eos_token_id=None,
):
    """Single-prompt wrapper over Qwen3Model.generate_batch.

    Keeps the old call signature while reusing the one KV-cached decode loop,
    so there is no second sampling implementation to keep in step.

    num_sequences decodes that many copies as ONE batch, which is faster than
    looping and only differs from repeated calls when temperature > 0.

    Returns prompt + completion (a list when num_sequences > 1, else the string).
    """
    model.eval()
    results, _, _ = model.generate_batch(
        tokenizer,
        [prompt] * num_sequences,
        max_new_tokens=max_length,
        eos_ids=None if eos_token_id is None else {eos_token_id},
        temperature=temperature,
        top_p=top_p,
    )
    texts = [prompt + text for text, _ in results]
    return texts if num_sequences > 1 else texts[0]


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


# if __name__ == '__main__':
#     from qwen_tokenizer import Qwen3Tokenizer

#     TOKENIZER_PATH = "collections/qwen3/models/tokenizer.json"

#     # Qwen3 - 0.6B 
#     model = Qwen3Model(QWEN_06B_CFG)
#     model = from_pretrained(model, repo_id="Qwen/Qwen3-0.6B-Base") #Pretraining
#     # model = from_pretrained(model, repo_id="Qwen/Qwen3-0.6B") #SFT/RLHF 

#     # Qwen3 - 1.7B 
#     # model = Qwen3Model(QWEN_1B7_CFG)
#     # model = from_pretrained(model, repo_id="Qwen/Qwen3-1.7B-Base") #Pretraining
#     # model = from_pretrained(model, repo_id="Qwen/Qwen3-1.7B") #SFT/RLHF 
    
#     # Qwen3 - 4B
#     # model = Qwen3Model(QWEN_4B_CFG)
#     # model = from_pretrained(model, repo_id="Qwen/Qwen3-4B-Base") #Pretraining
#     # model = from_pretrained(model, repo_id="Qwen/Qwen3-4B") #SFT/RLHF 

#     # Qwen3 - 8B
#     # model = Qwen3Model(QWEN_8B_CFG)
#     # model = from_pretrained(model, repo_id="Qwen/Qwen3-8B-Base") #Pretraining 
#     # model = from_pretrained(model, repo_id="Qwen/Qwen3-8B") #SFT/RLHF 

#     # Qwen3 - 14B
#     # model = Qwen3Model(QWEN_14B_CFG)
#     # model = from_pretrained(model, repo_id="Qwen/Qwen3-14B-Base") #Pretraining
#     # model = from_pretrained(model, repo_id="Qwen/Qwen3-14B") #SFT/RLHF 

#     # Qwen3 - 32B
#     # model = Qwen3Model(QWEN_32B_CFG)
#     # model = from_pretrained(model, repo_id="Qwen/Qwen3-32B") #SFT/RLHF 


#     # 2. Load tokenizer (same for all Qwen3 model sizes)
#     tokenizer = Qwen3Tokenizer(TOKENIZER_PATH)
#     model.cuda()

#     # 3. Demo 
#     prompt = "Explain large language models in a single sentence."
#     output = generate(
#         model, tokenizer, prompt, 
#         max_length=200, temperature=0.0, top_p=None
#     )