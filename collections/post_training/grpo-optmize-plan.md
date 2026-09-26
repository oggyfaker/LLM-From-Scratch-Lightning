# GRPO speed-up plan (from-scratch file 4)

Optimization plan and the related conversation for
`collections/qwen3/4_Qwen3_Gsm8k_GRPO.py`. Nothing here is implemented yet.
Pick items by ID (S1a, S2c, …) when asking for implementation.

- Created: 2026-09-26
- Hardware: 1× A100 80GB PCIe, conda env `LLM` (torch 2.11, pytorch_lightning 2.6.1,
  vLLM 0.20.1, unsloth 2026.5.2, trl 0.24.0)

---

## 1. Where the time goes today (measured)

Numbers come from the file 4 run launched 2026-09-26 (`logs/Qwen3_Gsm8k_GRPO_14B_LoRA_think/26_09_26/version_0/`):
Qwen3-14B bf16 + LoRA r32, 4 prompts × 4 answers = 16 rollouts per step, 384 max new tokens.

| What | From-scratch file 4 | Unsloth + vLLM (file 6) |
|---|---|---|
| Mean optimizer step | **17.6 s** (median 16.7 s) | ~18 s |
| Step with no backward (every group had zero advantage) = sampling only | **14.7 s** | – |
| Step with backward | 18.4 s, so the backward adds ~3.7 s | – |
| Generation + scoring inside a step | ~14–15 s (~80% of the step) | 15.7 s (includes wake/sleep, LoRA load, old/ref log-probs) |
| One eval round (1319 test questions, greedy) | **11.4 min** | **1.7 min** |
| Peak GPU memory per step | median 35.6 GB, max 49.5 GB | ~30.7 GB used (vLLM in standby) |
| Mean completion length | 97.7 tokens | ~97 tokens |

Conclusions:
- **Sampling is the bottleneck.** The training forward/backward is only ~3.7 s per step.
- **Rollouts decode slowly.** At batch 16 a decode step costs ~60 ms per token (≈14.7 s ÷ ~250 steps to the longest rollout). Reading the bf16 weights once takes only ~20 ms (29.5 GB at ~1.5 TB/s), so ~2/3 of each token is overhead.
- **Eval is ~6.7× slower than vLLM:** ~0.38 s per token at batch 256.
- **vLLM would not win much per training step:** Unsloth's measured step on this exact workload was ~18 s. Its clear win is the eval round.

### Suspected overheads (verify with the profiler first)

1. `torch.cat` in the KV cache re-copies the whole K/V history of every layer on every token (O(T²) memory traffic).
2. The hand-written GQA attention broadcasts K/V over the 5 query heads per KV head. `torch.matmul` on the broadcast likely makes a 5× copy of K/V per step.
3. `next_token.tolist()` plus a Python loop over rows forces a GPU→CPU sync on every token.
4. There are no CUDA graphs: ~3,000 small kernel launches per token (40 layers × attention/MLP/RMSNorm/RoPE ops plus 7 LoRA adapters × 2 matmuls plus fp32→bf16 casts).
5. The `token_logprobs` forward (training) goes through the hand-written masked attention. It stores `(B, 40, T, T)` fp32 attention probabilities, which is why `LOGPROB_CHUNK` is only 2.

**Profile before changing code.** Wrap ~20 decode steps of `sample_rollouts` (batch 16) and ~20 steps of an eval batch (batch 256) in `torch.profiler.profile(activities=[CPU, CUDA])`. Sort by CUDA time and by CPU time to see whether copies, attention or launch gaps dominate.

---

## 2. Step 1: fix the from-scratch decoder (cheap, large gain, stays from scratch)

Files to change:

| File | Change |
|---|---|
| `collections/qwen3/models/qwen3_dense.py` | `KVCache` → pre-allocated `StaticKVCache` (S1a). Replace both hand-written attention paths in `GroupedQueryAttention.forward` with SDPA (S1b). Point `Qwen3Model.generate_batch` at the new decode loop. |
| `collections/post_training/GRPO/rollout.py` | Keep the decode loop on the GPU (S1c). Delete the private `KVCache` copy. Optionally CUDA-graph the decode step (S1d). |
| `collections/post_training/GRPO/grpo_logprobs.py` | No change: it benefits from S1b automatically. |
| `collections/qwen3/4_Qwen3_Gsm8k_GRPO.py` | Config only: `LOGPROB_CHUNK` 2 → 4 once S1b frees the attention memory. |

### S1a: Pre-allocated KV cache (`qwen3_dense.py`)

```python
class StaticKVCache:
    """One pre-allocated (B, n_kv, max_len, head_dim) buffer per layer. Decode writes
    one column in place instead of torch.cat-ing the whole history every token."""
    def __init__(self, n_layers, batch, n_kv, head_dim, max_len, device, dtype=torch.bfloat16):
        shape = (batch, n_kv, max_len, head_dim)
        self.keys = [torch.empty(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.values = [torch.empty(shape, device=device, dtype=dtype) for _ in range(n_layers)]

    def write(self, layer_idx, start_pos, k, v):
        end = start_pos + k.shape[2]
        self.keys[layer_idx][:, :, start_pos:end] = k
        self.values[layer_idx][:, :, start_pos:end] = v
        return self.keys[layer_idx][:, :, :end], self.values[layer_idx][:, :, :end]
```

Memory is the same as today's peak, e.g. eval batch 256 × 601 positions × 160 KB/token ≈ 25 GB. Qwen3-14B KV per token is 40 layers × 2 × 8 heads × 128 × 2 bytes = 160 KB.

### S1b: SDPA attention with native GQA (`GroupedQueryAttention.forward`)

This replaces both the training path and the generation path after RoPE:

```python
import torch.nn.functional as F

q = apply_rope(q, cos, sin, offset=start_pos)
k = apply_rope(k, cos, sin, offset=start_pos)
if cache is not None:
    k, v = cache.write(layer_idx, start_pos, k, v)            # in place, no cat
Tq, Tk = q.shape[2], k.shape[2]

if pad_mask is None and cache is None:                         # SFT path: flash kernel
    ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
else:                                                          # rollouts, eval, token_logprobs
    q_pos = torch.arange(start_pos, start_pos + Tq, device=x.device)
    allowed = (torch.arange(Tk, device=x.device)[None, :] <= q_pos[:, None])[None, None]
    pad_k = pad_mask[:, None, None, :Tk]
    pad_q = pad_mask[:, None, start_pos:start_pos + Tq, None]
    allowed = allowed & (~pad_k | pad_q)   # same rule as today: real queries never see
                                           # padding, pad queries keep a non-empty row (no NaN)
    ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, enable_gqa=True)
ctx = ctx.transpose(1, 2).reshape(b, Tq, self.d_out)
return self.out_proj(ctx)
```

Notes:
- SDPA's default scale is `1/sqrt(head_dim)`, the same as today.
- The mask rule is equivalent to today's `blocked = causal | (pad_k & ~pad_q)`.
- The masked path no longer stores `(B, 40, T, T)` fp32 attention probabilities in training, so `LOGPROB_CHUNK` can go to 4 (one GRPO group per forward).
- With a boolean mask, SDPA uses the memory-efficient kernel, not flash. Check with the profiler that `enable_gqa` does not fall back to a `repeat_interleave` copy. If it does, use the fold trick for decode: reshape q to `(b, n_kv, group_size*Tq, hd)` so no GQA is needed.
- Switching the SFT path (`pad_mask is None and cache is None`) to flash changes SFT numerics slightly. That part is optional.

### S1c: Keep the decode loop on the GPU (`rollout.py`, `sample_rollouts` steps 6–7)

```python
N, L = len(flat_prompts), T + max_new_tokens
att = model.transformer_blocks[0].att
cache = StaticKVCache(len(model.transformer_blocks), N, att.num_kv_groups, att.head_dim, L, device)
pad_mask_full = torch.zeros((N, L), dtype=torch.bool, device=device)
pad_mask_full[:, :T] = pad_mask                                   # the prompt's left padding
eos = torch.tensor(sorted(eos_ids), device=device)

with amp:
    logits = model(input_ids, cache=cache, start_pos=0, pad_mask=pad_mask_full,
                   logits_last_only=True)[:, -1]
out = torch.full((N, max_new_tokens), pad_id, device=device)
finished = torch.zeros(N, dtype=torch.bool, device=device)
gen_lens = torch.zeros(N, dtype=torch.long, device=device)
for t in range(max_new_tokens):
    if temperature > 0:
        probs = top_p_filter(torch.softmax(logits.float() / temperature, -1), top_p)
        nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    else:
        nxt = logits.argmax(-1)
    nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
    out[:, t] = nxt
    gen_lens += (~finished).long()                      # EOS still counts, as today
    finished |= torch.isin(nxt, eos)
    if (t + 1) % 16 == 0 and bool(finished.all()):      # the only host sync: every 16 tokens
        break
    with amp:
        logits = model(nxt[:, None], cache=cache, start_pos=T + t,
                       pad_mask=pad_mask_full, logits_last_only=True)[:, -1]
# steps 8-11 unchanged, but read out[:, :gen_lens.max()], gen_lens.tolist(), finished.tolist()
```

Optional: prefill the P unique prompts once and copy their K/V into the G rollouts with `repeat_interleave(G, dim=0)`. It saves little here because prompts are short (p50 = 85 tokens).

### S1d (optional, last): CUDA-graph the one-token decode step

A graph needs fixed shapes:
- attend over the full `L` buffer with a mask computed from a GPU `pos` tensor;
- gather RoPE `cos`/`sin` with `index_select(pos)`;
- write the cache with `index_copy_(2, pos, k)`;
- use no `.item()` and no Python branching on tensor values.

```python
class DecodeGraph:
    def __init__(self, model, cache, pad_mask, N, device):
        self.tok = torch.zeros(N, 1, dtype=torch.long, device=device)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                                # warm-up outside the graph
            for _ in range(2): model.decode_step(self.tok, self.pos, cache, pad_mask)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = model.decode_step(self.tok, self.pos, cache, pad_mask)

    def __call__(self, tok, pos):
        self.tok.copy_(tok[:, None]); self.pos.fill_(pos)
        self.graph.replay()
        return self.logits
```

- Capture one graph for training rollouts (N=16) and one for eval (N=256).
- LoRA weights are read by pointer, so the optimizer's in-place updates reach the graph with no re-capture.
- Lower-effort alternative: `torch.compile(model.decode_step, mode="reduce-overhead")`.
- This matters most at batch 16, where launch overhead dominates.

### Step 1 acceptance checks

- The same seed gives the same round-0 greedy accuracy (87.49%, 1154/164/1), or within a couple of questions (bf16 kernel differences).
- `token_logprobs` before and after S1b: max |diff| ~1e-2 in bf16 on a real batch; exact equality is not expected.
- Tiny-model check (scratchpad `unit_check.py` pattern): trimmed vs full log-probs, LoRA-off == base, chunked == whole-batch loss.
- Log `Training/time_step_s` and the eval-round minutes before and after each of S1a–S1d.

---

## 3. Step 2: vLLM as the rollout and eval engine (the Unsloth approach)

Files:

| File | Change |
|---|---|
| **new** `collections/post_training/GRPO/vllm_engine.py` | Engine creation, `push_lora`, `share_base_weights`, standby patch, `vllm_rollouts` |
| `collections/post_training/GRPO/rollout.py` | Factor steps 8–11 of `sample_rollouts` into `build_rollout_batch(tokenizer, prompts, records, group_size, completion_ids, finish_reasons)` so both samplers return the same `RolloutBatch` |
| `collections/qwen3/4_Qwen3_Gsm8k_GRPO.py` | Create the engine before the model reaches the GPU; `training_step` calls `push_lora` + `vllm_rollouts`; `validation_step` uses greedy through the engine |
| `collections/utils/hf_export.py` | Already exists (used by file 6). It gives vLLM the HF folder `<pth stem>_hf/` next to the SFT `.pth` |

Facts checked against the installed code:
- vLLM's built-in `Qwen3ForCausalLM` implements `SupportsLoRA` with `packed_modules_mapping`, so use it through the HF export.
- `models/qwen3_vllm.py` (the from-scratch vLLM model) has no `SupportsLoRA`/`packed_modules_mapping` yet. It could only serve rollouts after those are added.
- Stock vLLM 0.20.1 `LoRARequest` requires `lora_path` (it asserts non-empty) and has `load_inplace: bool` to replace an adapter with the same id. In-memory `lora_tensors` exist only through Unsloth's patch.
- vLLM `LLM.sleep(level=1)` offloads the weights to the CPU and discards the KV cache. `wake_up(tags=...)` accepts `"weights"`, `"kv_cache"` and `"scheduling"`.
- Unsloth reference code in `site-packages/unsloth_zoo/vllm_utils.py`:
  - `patch_vllm` (line ~780) sets `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  - `patch_vllm_enable_sleep_mode` (line ~525): standby
  - `_get_vllm_state_dict` (line ~872): the path to the vLLM model
  - `load_lora` (line ~2574): LoRA tensors

### S2a: Engine, in the same process as training

```python
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"     # before importing vllm: keeps the model reachable
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model=str(hf_dir), dtype="bfloat16", max_model_len=640,
          gpu_memory_utilization=0.60,     # ~48 GB: 29.5 GB weights (the only copy) + KV cache
          enable_lora=True, max_lora_rank=32, max_loras=1,
          enable_sleep_mode=True, enable_prefix_caching=True,
          compilation_config=0,            # file 6's workaround: vLLM 0.20.1 compile crashes on Qwen3 + LoRA
          seed=CFG.SEED)
vllm_model = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model
```

Create it while the GPU is still empty: vLLM sizes its KV cache from free memory at start-up.

### S2b: Syncing weights each step (only the LoRA changes)

```python
HF = {"W_query": "self_attn.q_proj", "W_key": "self_attn.k_proj", "W_value": "self_attn.v_proj",
      "out_proj": "self_attn.o_proj", "fc1": "mlp.gate_proj", "fc2": "mlp.up_proj", "fc3": "mlp.down_proj"}

def push_lora(model, out_dir="/dev/shm/grpo_lora"):
    tensors = {}
    for l, blk in enumerate(model.transformer_blocks):
        for ours, hf in HF.items():
            mod = getattr(blk.att if ours in ("W_query", "W_key", "W_value", "out_proj") else blk.ff, ours)
            key = f"base_model.model.model.layers.{l}.{hf}"
            # ours: x @ A @ B, A (in, r), B (r, out)   PEFT: x @ lora_A.T @ lora_B.T
            tensors[f"{key}.lora_A.weight"] = mod.lora.A.detach().T.to(torch.bfloat16).contiguous()
            tensors[f"{key}.lora_B.weight"] = mod.lora.B.detach().T.to(torch.bfloat16).contiguous()
    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, f"{out_dir}/adapter_model.safetensors")          # 257 MB into RAM (tmpfs)
    cfg_path = f"{out_dir}/adapter_config.json"
    if not os.path.exists(cfg_path):
        json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": CFG.RANK,
                   "lora_alpha": CFG.ALPHA, "lora_dropout": 0.0, "bias": "none",
                   "target_modules": list({v.split(".")[-1] for v in HF.values()})}, open(cfg_path, "w"))
    return LoRARequest("policy", 1, out_dir, load_inplace=True)   # same id: replaced, not cached
```

- vLLM applies `alpha / r`, the same scaling as `LoRALayer`.
- Without `load_inplace=True` (or a new id every step), vLLM keeps serving the cached old adapter.
- The upgrade path is in-memory tensors, which needs a patch like Unsloth's `load_lora(..., load_tensors=True)`.

### S2c: One copy of the weights (weight sharing)

The frozen base `Linear` weights of the training model become views of vLLM's fused tensors, which saves ~28 GB.

```python
def share_base_weights(model, vllm_model, cfg):
    q, kv, h = cfg.n_heads * cfg.head_dim, cfg.n_kv_groups * cfg.head_dim, cfg.hidden_dim
    V = dict(vllm_model.named_modules())
    w = lambda n: getattr(V[n], "base_layer", V[n]).weight       # vLLM wraps layers for LoRA
    put = lambda lin, t: setattr(lin.linear, "weight", torch.nn.Parameter(t, requires_grad=False))
    for l, blk in enumerate(model.transformer_blocks):
        qkv = w(f"model.layers.{l}.self_attn.qkv_proj")
        put(blk.att.W_query, qkv[:q]); put(blk.att.W_key, qkv[q:q + kv]); put(blk.att.W_value, qkv[q + kv:])
        put(blk.att.out_proj, w(f"model.layers.{l}.self_attn.o_proj"))
        gu = w(f"model.layers.{l}.mlp.gate_up_proj")
        put(blk.ff.fc1, gu[:h]); put(blk.ff.fc2, gu[h:])
        put(blk.ff.fc3, w(f"model.layers.{l}.mlp.down_proj"))
    model.tok_emb.weight = torch.nn.Parameter(w("model.embed_tokens"), requires_grad=False)
    model.out_head.weight = torch.nn.Parameter(w("lm_head"), requires_grad=False)
    # RMSNorm scales stay ours (fp32, tiny)
```

- The training model must never hold its own GPU copy. Build `Qwen3Model` on the `meta` device, call `apply_lora`, share the weights, then load only the norm scales from the `.pth`.
- Row slices (`qkv[:q]`, `gu[:h]`) are contiguous views, so `F.linear` works on them unchanged.
- Lightning's `.to(device)` does not copy tensors that are already on CUDA.
- After sharing, compare our `token_logprobs` with vLLM's sampled-token logprobs on one batch (`SamplingParams(logprobs=0)`).

### S2d: Standby (release only the KV cache during backward)

Plain `llm.sleep(level=1)` moves the weights to the CPU, which would leave the shared views pointing at freed memory. Unsloth's standby patches vLLM's `CuMemAllocator` to skip allocations tagged `"weights"`:

```python
from vllm.device_allocator.cumem import CuMemAllocator, unmap_and_release, create_and_map

def _sleep(self, offload_tags=None):
    for data in self.pointer_to_data.values():
        if data.tag != "weights":
            unmap_and_release(data.handle)          # KV cache gone, weights stay put
    torch.cuda.empty_cache()

def _wake_up(self, tags=None):
    for data in self.pointer_to_data.values():
        if data.tag != "weights" and (tags is None or data.tag in tags):
            create_and_map(data.handle)

CuMemAllocator.sleep, CuMemAllocator.wake_up = _sleep, _wake_up
```

- Port the tested version, `patch_vllm_enable_sleep_mode` in `unsloth_zoo/vllm_utils.py` (~line 525), rather than this sketch.
- `PYTORCH_CUDA_ALLOC_CONF` must not contain `expandable_segments:True`.
- Unsloth refuses standby on vLLM 0.10.x and 0.14.x (known allocator crashes); 0.20.1 is fine (file 6 ran with it).

### S2e: Sampling, then back into the existing pipeline

```python
def vllm_rollouts(llm, tokenizer, prompts, records, group_size, lora_req, eos_ids,
                  temperature=1.0, top_p=1.0, max_new_tokens=384):
    sp = SamplingParams(n=group_size, temperature=temperature, top_p=top_p, top_k=-1, min_p=0.0,
                        max_tokens=max_new_tokens, stop_token_ids=sorted(eos_ids),
                        skip_special_tokens=False)    # n=G shares one prompt prefill
    llm.wake_up()                                     # KV cache back (weights never left)
    outs = llm.generate(prompts, sp, lora_request=lora_req, use_tqdm=False)
    llm.sleep(level=1)                                # standby before backward
    comp_ids, finish = [], []
    for o in outs:                                    # prompt order kept -> [p0 x G, p1 x G, ...]
        for c in o.outputs:
            ids = list(c.token_ids)
            if c.finish_reason == "stop" and (not ids or ids[-1] not in eos_ids):
                ids.append(tokenizer.eos_token_id)    # EOS stays a trained token, as today
            comp_ids.append(ids)
            finish.append(c.finish_reason)            # "stop" | "length", same vocabulary as now
    return build_rollout_batch(tokenizer, prompts, records, group_size, comp_ids, finish)
```

Wiring in file 4:
```python
# once, before the Lightning model reaches the GPU
engine = make_engine(hf_dir); share_base_weights(model, vllm_model(engine), model_cfg)
# training_step
lora_req = push_lora(self.model)          # policy == the weights the last optimizer step produced
rollouts = vllm_rollouts(engine, self.tokenizer, prompts, records, CFG.GROUP_SIZE, lora_req, eos_ids)
# ...reward / advantages / backward_loss_chunk unchanged
# validation: SamplingParams(temperature=0, n=1), all 1319 prompts in ONE generate call
```

### S2f (optional): Off-policy correction

vLLM's log-probs differ slightly from the trainer's (different kernels, bf16 LoRA). Unsloth's run had the correction off (`vllm_importance_sampling_correction=False`). First log the mismatch (`Training/vllm_logprob_diff` = mean |trainer lp − vLLM lp|). Add truncated importance sampling only if the mismatch grows:
```python
per_token = per_token * torch.exp(trainer_lp.detach() - vllm_lp).clamp(max=2.0)
```

### Step 2 pitfalls (from file 6 and the vLLM source)

- Set `VLLM_ENABLE_V1_MULTIPROCESSING=0` before `import vllm`, or the model lives in another process.
- `compilation_config=0`: vLLM 0.20.1's piecewise compile crashes on Qwen3 + LoRA.
- File 6's `get_dummy_lora_warmup_rank` shim is only needed with unsloth_zoo's LoRA worker manager, not stock vLLM.
- `gpu_memory_utilization` counts the whole GPU. Leave room for training activations (~10–15 GB) plus LoRA and optimizer (~1.5 GB).
- vLLM's greedy eval will not match the from-scratch greedy eval bit for bit. At round 0, Unsloth/vLLM scored 87.41% vs 87.49% from scratch.

### Step 2 acceptance checks

- Round 0 through vLLM ≈ 87.4% (as file 6).
- After `share_base_weights`: trainer log-probs vs vLLM log-probs mean |diff| ≲ 1e-2; GPU memory drops by ~29 GB.
- After one optimizer step, vLLM output changes. This confirms the LoRA reload works (`load_inplace`).
- Measured: step time and eval-round minutes vs Section 1.

---

## 4. Other Unsloth techniques used by file 6 (for reference)

| Technique | What it does | Relevant to file 4? |
|---|---|---|
| `fast_inference=True` | vLLM engine in the same process | = Step 2 |
| `UNSLOTH_VLLM_STANDBY=1` + `vllm_enable_sleep_mode` | KV cache released during training, weights stay resident and shared | = S2c + S2d |
| `model.load_lora(..., load_tensors=True)` | LoRA handed to vLLM as in-memory tensors (no disk) | S2b upgrade path |
| `gpu_memory_utilization=0.9`, `max_lora_rank` | vLLM memory and LoRA sizing | S2a |
| `use_gradient_checkpointing="unsloth"` | activation checkpointing with async CPU offload | memory, not speed; not needed at 35–50 GB |
| Chunked efficient GRPO loss (`UnslothEfficientGRPO`, `unsloth_num_chunks`) | computes log-probs from hidden states in chunks, never building full `(B, T, 151936)` logits | memory; would allow bigger `LOGPROB_CHUNK` |
| Triton kernels (RMSNorm, RoPE, SwiGLU, fused LoRA MLP/QKV) | fewer, fused kernels in forward/backward | speed of the ~3.7 s backward part |
| `adamw_8bit` | 8-bit optimizer states | already used in file 4 (`OPTIM="adamw_8bit"`) |

---

## 5. Recommended order

1. **Profile** (Section 1) to confirm which overhead dominates.
2. **S1a + S1b + S1c**: low risk, stays from scratch, fixes both rollouts and eval. Re-measure step time and eval minutes.
3. **Decide on vLLM (Step 2)**: the clear, measured win is eval (1.7 min vs 11.4 min per round). For training steps Unsloth's own vLLM step was ~18 s on this workload, so don't expect 5×. It pays off more with larger G or longer completions.
4. **S1d (CUDA graphs)** only if the profile still shows kernel-launch gaps after steps 2–3.

---

## 6. Conversation log (optimization-related parts only)

**User (first request, 2026-09-26):** I haven't implemented vLLM to speed up generating answers. Using the ideas in Unsloth, give me a plan to make my from-scratch code faster at inference. Include the techniques Unsloth uses to speed inference: weight sharing between the training engine and the inference engine, Unsloth standby, and anything else `6_Qwen3_Gsm8k_GRPO_Unsloth.py` currently uses. I see that the Unsloth rollout process is fast. Just give me ideas and tell me the plan: if I want to do it, what must I do?

**Assistant (summary of the first speed plan):**
- In bf16 the from-scratch training step is already close to Unsloth's (~20 s vs ~18 s). The real gap is eval: 11.4 min per round vs vLLM's 1.8 min.
- Step 1, fix the decoder: pre-allocated KV cache; SDPA with GQA; GPU-side "finished" check every ~16 tokens and removing finished rows; CUDA graphs / `torch.compile` for the decode step.
- Step 2, vLLM as the rollout engine: in-process engine with `models/qwen3_vllm.py` as a possible base; LoRA-only sync each step via `LoRARequest`, with the name/layout conversion (`lora_A = A.T`, `lora_B = B.T`, fused qkv/gate_up); weight sharing (trainer weights point at vLLM's tensors); standby (sleep mode frees the KV cache during backward); `SamplingParams(n=4, ...)` with prefix caching; greedy vLLM for eval.
- Other file 6 pieces:
  - chunked loss without full logits
  - CPU-offloaded gradient checkpointing
  - Triton kernels (RMSNorm, RoPE, SwiGLU, LoRA)
  - 8-bit Adam
- Watch-outs: log-prob mismatch (keep ratio = 1 or add capped IS), file 6's vLLM 0.20.1 workarounds.

**User (second request, 2026-09-26):** For "Step 1: fix the from-scratch decoder", which files would you change and how? Just the idea and prototype code. Also give prototype code for applying vLLM: weight syncing, one copy of the weights, standby, sampling. I will evaluate and let you implement later.

**Assistant (answer to the second request):** Sections 2 and 3 above, with the file-by-file changes and prototype code for S1a–S1d and S2a–S2f. It was based on reading the installed versions:
- vLLM `LoRARequest` needs `lora_path` and supports `load_inplace`.
- vLLM `sleep(level=1)` offloads weights.
- Unsloth's standby skips the `"weights"` tag in `CuMemAllocator`.
- vLLM's built-in `Qwen3ForCausalLM` supports LoRA; `models/qwen3_vllm.py` does not yet.

Recommended order: profile → S1a–S1c → decide on vLLM → S1d.

**Measured afterwards (added when this file was written, run at step 740/1868):**
- mean step 17.6 s; a sampling-only step 14.7 s, so the backward is ~3.7 s;
- peak memory median 35.6 GB (max 49.5 GB);
- rollout decode ≈ 60 ms per token at batch 16, vs ~20 ms to read the weights once.

This moved the main target from "training" to "rollout decode" (Section 1).

---

## 7. Status

| ID | Item | Status |
|---|---|---|
| P0 | Profile the decode loop (batch 16) and one eval batch (batch 256) | not started |
| S1a | Pre-allocated KV cache | not started |
| S1b | SDPA attention (GQA, masks) + `LOGPROB_CHUNK` 4 | not started |
| S1c | GPU-side decode loop, sync every 16 tokens | not started |
| S1d | CUDA graph / `torch.compile` decode step | not started |
| S2a | vLLM engine in the same process (HF export folder) | not started |
| S2b | LoRA sync via `/dev/shm` + `LoRARequest(load_inplace=True)` | not started |
| S2c | Weight sharing (trainer weights = views of vLLM's) | not started |
| S2d | Standby (keep weights, drop KV cache) | not started |
| S2e | `vllm_rollouts` + `build_rollout_batch` refactor + vLLM eval | not started |
| S2f | Off-policy correction (only if the mismatch grows) | not started |
