"""Batched generation for HuggingFace / Unsloth backbones.

The from-scratch Qwen3Model carries its own ``generate_batch`` (KV cache,
left-padding, the lot). A HuggingFace model already has ``generate()``, so the
eval callback reaches it through these two pieces instead:

    hf_generate_batch   the ``generate_fn`` the callback calls
    UnslothEvalMixin    the mode switching HF+Unsloth needs around a round

Both are backbone concerns, not dataset concerns, which is why they live here
rather than in data/<dataset>/metric.py. Mix them into whichever dataset
callback a run needs:

    class MoEEvalCallback(UnslothEvalMixin, Gsm8kEvalCallback):
        pass
"""


def hf_generate_batch(model, tokenizer, prompts, max_new_tokens=256,
                      max_total_len=None, eos_ids=None, pad_id=None, **_):
    """Greedy-decode a batch -> ([(text, finish_reason), ...], steps, budget).

    Same contract as Qwen3Model.generate_batch, so the eval callback cannot
    tell the two backends apart.

    Prompts are LEFT-padded so every row's next-token slot is the last column;
    HF applies the attention mask so the pad prefix contributes nothing. The
    kwargs the callback passes but HF has no use for are swallowed by **_.
    """
    device = next(model.parameters()).device
    eos_ids = sorted(eos_ids) if eos_ids else [tokenizer.eos_token_id]
    if pad_id is None:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # HF defaults to right padding, which would make the model continue from
    # the pad block instead of the prompt.
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False)
    finally:
        tokenizer.padding_side = previous_side

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    # Never generate past what training ever produced.
    budget = max_new_tokens if max_total_len is None else min(
        max_new_tokens, max_total_len - prompt_len)
    budget = max(1, budget)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=budget,
        do_sample=False,          # greedy, so rounds differ only by the model
        use_cache=True,
        eos_token_id=eos_ids,
        pad_token_id=pad_id,
    )

    generated = out[:, prompt_len:]
    steps = generated.shape[1]
    stop_set = set(eos_ids)

    results = []
    for row in generated:
        ids = row.tolist()
        # No stop token means the row ran out of budget, which the grader
        # reports as truncated_length rather than a reasoning error.
        stop_at = next((i for i, t in enumerate(ids) if t in stop_set), None)
        finish = "stop" if stop_at is not None else "length"
        kept = ids[:stop_at] if stop_at is not None else ids
        results.append(
            (tokenizer.decode(kept, skip_special_tokens=True).strip(), finish))
    return results, steps, budget


class UnslothEvalMixin:
    """Mode switching for an Unsloth backbone, mixed into an eval callback.

    Overrides the callback's ``_set_mode``, which by default is only
    eval()/train(). Two things need more than that:

    ATTENTION: this build resolves Qwen3-MoE to flex_attention, which training
    needs (68.4 GB of 79.25 at 8k tokens; eager OOMs there) but which cannot
    generate — HF's cache-aware mask goes through create_block_mask, which
    raises ValueError on it. Eager is cheap for decoding and the training
    implementation is restored on the way out.

    MODE: gradient checkpointing forces use_cache off, so generating in training
    mode would recompute the prefix for every token.
    """

    # Attention implementation used only while generating. Training runs on
    # whatever the model loaded with.
    INFER_ATTN_IMPL = "eager"

    _train_attn_impl = None     # remembered on the way in, restored on the way out

    def _swap_attn(self, model, impl):
        """Point the model at a different attention implementation, if it allows it."""
        if not impl or getattr(model.config, "_attn_implementation", None) == impl:
            return
        try:
            model.set_attn_implementation(impl)
        except Exception as e:
            print(f"  [eval] could not switch attention to {impl}: "
                  f"{type(e).__name__}: {e}", flush=True)

    def _set_mode(self, model, inference):
        if inference:
            self._train_attn_impl = getattr(model.config, "_attn_implementation", None)
            self._swap_attn(model, self.INFER_ATTN_IMPL)
        else:
            self._swap_attn(model, self._train_attn_impl)

        fn = getattr(model, "for_inference" if inference else "for_training", None)
        if callable(fn):
            try:
                fn()
                return
            except Exception as e:
                print(f"  [eval] Unsloth mode switch failed: {type(e).__name__}: {e}")
        model.eval() if inference else model.train()
