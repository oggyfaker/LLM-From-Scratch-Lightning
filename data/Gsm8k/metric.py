"""GSM8K grading and the periodic generative-evaluation callback.

Everything here is model-agnostic, so every GSM8K run — the from-scratch dense
Qwen3, the Unsloth MoE backbone, whatever comes next — grades with the SAME
code. The two graders have to stay identical or accuracy numbers from different
runs stop being comparable.

Two halves:

    ANSWER GRADING
        classify() / build_record() / summarize(). Pure functions over strings,
        no torch, no Lightning — usable from a notebook or a vLLM script.

    PERIODIC FULL-TEST-SET GENERATIVE EVALUATION
        Gsm8kEvalCallback: fires every N% of training, decodes the whole test
        set, grades it, writes round<N>.json and publishes the metrics that the
        checkpoint callbacks monitor.

Nothing here is tied to Qwen3. The callback reaches a model through three
seams, so a different LLM on this same dataset reuses all of it:

    generate_fn   how to decode a batch. Defaults to the model's own
                  ``model.generate_batch(tokenizer, prompts, ...)``, which
                  Qwen3Model implements with a KV cache. An HF/Unsloth
                  backbone passes its own instead.
    prompt_fn     how to build a prompt. Defaults to this dataset's ChatML
                  template; a model from another family supplies its own.
    _set_mode     how to flip between training and generation. Defaults to
                  eval()/train(); override for Unsloth's for_inference().

Everything else is read off the tokenizer rather than written down: stop-token
ids come from its eos_token_id (``stop_tokens`` only adds extras a chat format
needs on top), and the pad id from its pad_token_id. No model's special tokens
are hardcoded here.

The remaining contract is on the LightningModule, not the model: it must
expose ``_opt_step`` and ``.model``.
"""

import json
import math
import random
import re
import time
from pathlib import Path

import torch
import pytorch_lightning as pl

from .data_utils import format_prompt_only


# -----------------------------------
# ANSWER GRADING  (Correct/Wrong/Bad)
# -----------------------------------

# A well-formed boxed answer: literal \boxed{...} with no nested braces.
# Rejects "\boxed{" (unclosed), "boxed{5}" (no backslash) and "\boxed{}" (empty).
# This one is a property of the TASK, not of any model: the prompt asks for
# \boxed{}, so every model on this dataset is graded on it.
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")

# Where the reasoning stops and the answer begins. Qwen3 and the GSM8K chat
# template both use </think>; a model family that marks reasoning differently
# passes its own through the ``think_close`` argument.
THINK_CLOSE = "</think>"

# The metric the checkpoint callbacks monitor. Exported so a training script
# never has to spell the string out a second time.
ACCURACY_KEY = "Validation/box_answer_match_accuracy"


def normalize_answer(text):
    """GSM8K answers are plain numbers — strip formatting noise, not content."""
    return text.strip().replace(",", "").replace("$", "").rstrip(".").strip()


def to_number(text):
    """Return the numeric value of an answer string, or None if it isn't one."""
    try:
        return float(normalize_answer(text))
    except (TypeError, ValueError):
        return None


def classify(completion, expected, thinking_mode, finish_reason,
             think_close=THINK_CLOSE):
    """Grade one completion -> (matching, prediction, thinking_model, bad_reason).

    ``matching`` is one of three labels:
        Correct  a well-formed \\boxed{...} whose value equals the ground truth
        Wrong    a clean numeric answer, but not the right number
        Bad      no usable answer could be extracted at all

    ``bad_reason`` keeps a budget artefact from being read as a reasoning error:
        empty_output        model returned nothing
        think_not_closed    thinking mode, </think> never emitted
        truncated_length    hit max_new_tokens with no answer
        no_boxed_answer     finished cleanly but never wrote \\boxed{...}
        empty_box           wrote \\boxed{} with nothing inside
        non_numeric_answer  boxed something that isn't a number
    """
    text = completion or ""

    # 1. Split the reasoning off. Only text AFTER </think> may carry the final
    #    answer — scanning the chain of thought would credit a lucky intermediate.
    if thinking_mode:
        if think_close in text:
            thinking_model, _, answer_region = text.partition(think_close)
            thinking_model, answer_region = thinking_model.strip(), answer_region.strip()
        else:
            return ("Bad", None, text.strip(),
                    "empty_output" if not text.strip() else "think_not_closed")
    else:
        thinking_model, answer_region = "", text.strip()

    if not text.strip():
        return "Bad", None, thinking_model, "empty_output"

    # 2. The LAST box wins: reasoning often boxes an intermediate result first.
    matches = BOXED_RE.findall(answer_region)
    if not matches:
        reason = "truncated_length" if finish_reason == "length" else "no_boxed_answer"
        return "Bad", None, thinking_model, reason

    prediction = matches[-1].strip()
    if not prediction:
        return "Bad", None, thinking_model, "empty_box"

    pred_num, gold_num = to_number(prediction), to_number(expected)
    if pred_num is None:
        return "Bad", prediction, thinking_model, "non_numeric_answer"

    matching = "Correct" if (gold_num is not None and pred_num == gold_num) else "Wrong"
    return matching, prediction, thinking_model, None


def build_record(rec, completion, finish_reason, thinking_mode,
                 think_close=THINK_CLOSE):
    """One graded row, in the schema the vLLM inference scripts write."""
    matching, prediction, thinking_model, bad_reason = classify(
        completion, rec["answer"], thinking_mode, finish_reason, think_close
    )
    return {
        "question": rec["question"],
        "thinking": rec["thinking"],
        "answer": rec["answer"],
        "num_gt_tokens": rec["num_gt_tokens"],
        "prediction": prediction,
        "thinking-model": thinking_model,
        "matching": matching,
        # --- diagnostics ---
        "bad_reason": bad_reason,
        "finish_reason": finish_reason,
    }


def summarize(version, rows, elapsed=None):
    """Counts + boxed-answer accuracy for one evaluation round."""
    n = len(rows)
    counts = {k: sum(r["matching"] == k for r in rows)
              for k in ("Correct", "Wrong", "Bad")}
    bad_reasons = {}
    for r in rows:
        if r["bad_reason"]:
            bad_reasons[r["bad_reason"]] = bad_reasons.get(r["bad_reason"], 0) + 1
    return {"version": version, "questions": n, **counts,
            "accuracy": 100.0 * counts["Correct"] / max(1, n),
            "bad_reasons": bad_reasons, "elapsed_sec": elapsed}


# --------------------------------------------
# PERIODIC FULL-TEST-SET GENERATIVE EVALUATION
# --------------------------------------------

class Gsm8kEvalCallback(pl.Callback):
    """Solve the whole held-out test set by generation, N times during training.

    Fires every ``every_pct`` of the planned optimizer steps (0.20 -> 20/40/60/
    80/100%), greedily decodes every test question and grades it with classify().

    This replaces a 10-question probe whose +-32 points of sampling noise made
    round-to-round movement unreadable. Scoring all 1319 only became affordable
    once generation grew a KV cache: one forward per new token instead of a full
    recompute of the sequence, with ``batch_size`` questions decoded at once
    behind a left-padding mask.

    Runs from on_train_batch_end, which Lightning calls just BEFORE the
    validation loop (training_epoch_loop: advance() -> on_advance_end()), so the
    metrics land in callback_metrics in time for the checkpoint callbacks that
    monitor them on_validation_end.

    The LightningModule must expose ``_opt_step`` (completed optimizer steps);
    with gradient accumulation that is the only honest clock for "% of training".

    Args:
        tokenizer: anything with encode/decode + eos_token_id.
        records: test records from load_gsm8k_json.
        total_steps: planned optimizer steps for the whole run.
        output_dir: where round<N>.json lands (logs/<run>/validation/).
        every_pct: fraction of total_steps between rounds.
        num_samples: None scores every record; an int takes a fixed random draw.
        batch_size: questions decoded at once; halved on CUDA OOM.
        max_new_tokens: generation budget, clamped per batch against max_total_len.
        max_total_len: hard ceiling on prompt + generation (the training filter).
        thinking_mode: prompts end at <think>, so the model reasons before answering.
        seed: fixes the num_samples draw so rounds stay comparable.
        generate_fn: ``fn(model, tokenizer, prompts, **kwargs) -> (results, steps,
            budget)`` for a backbone without a ``generate_batch`` method.
        prompt_fn: ``fn(question, include_thinking=bool) -> str``. Defaults to
            the ChatML template the Qwen3 runs train on. A model from another
            family was never trained on ChatML, so it needs its own template
            here or it is being prompted in a format it has never seen.
        stop_tokens: EXTRA turn-enders beyond the tokenizer's own eos_token_id,
            as token strings or ids. Only needed for a chat format with more
            than one; unknown entries are dropped.
        think_close: the marker that ends the reasoning block. Only text after
            it is scanned for the answer.
    """

    def __init__(self, tokenizer, records, total_steps, output_dir,
                 every_pct=0.20, num_samples=None, batch_size=64,
                 max_new_tokens=384, max_total_len=512,
                 thinking_mode=True, seed=1001, generate_fn=None,
                 prompt_fn=None, stop_tokens=None, think_close=THINK_CLOSE):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_total_len = max_total_len
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.total_steps = max(1, total_steps)
        self.every = max(1, int(round(self.total_steps * every_pct)))
        self.thinking_mode = thinking_mode
        self.think_close = think_close
        self.generate_fn = generate_fn or self._model_generate_batch
        self.prompt_fn = prompt_fn or format_prompt_only

        self.records = self._select_records(records, num_samples, seed)

        self.eos_ids = self._resolve_stop_ids(tokenizer, stop_tokens)
        if not self.eos_ids:
            raise ValueError(
                "No stop-token id could be resolved. The tokenizer exposes no "
                "usable eos_token_id, so generation would always run to "
                "max_new_tokens — pass stop_tokens=(...) explicitly.")

        # Only fills the left-padding prefix, which the pad mask removes from
        # every real row's attention, so the exact id never reaches a score.
        self.pad_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_id is None:
            self.pad_id = getattr(tokenizer, "eos_token_id", None)
        if self.pad_id is None:
            self.pad_id = min(self.eos_ids)

        self._fired = set()
        self._round_idx = 0
        self._last_metrics = None   # last successful round, for _guarded_run

    # ------------------------------------------------- dataset-specific seams
    # A dataset whose records or scoring differ subclasses these four. They are
    # the only places the class knows it is looking at GSM8K.

    @staticmethod
    def _select_records(records, num_samples, seed):
        """Which questions to score. None means all of them, which is the point.

        A dataset with rare categories overrides this to sample within each one,
        so a small draw cannot miss a category entirely.
        """
        records = list(records)
        if num_samples is None or num_samples >= len(records):
            return records
        return random.Random(seed).sample(records, num_samples)

    def _build_record(self, rec, completion, finish_reason):
        """Grade one completion into an output row."""
        return build_record(rec, completion, finish_reason,
                            self.thinking_mode, self.think_close)

    def _summarize(self, version, rows, elapsed):
        """Roll the graded rows up into one round's summary."""
        return summarize(version, rows, elapsed)

    def _metrics_from_summary(self, summary):
        """The scalars published to the progress bar, checkpoints and loggers."""
        return {
            ACCURACY_KEY: float(summary['accuracy']),
            'Validation/correct_count': float(summary['Correct']),
            'Validation/wrong_count': float(summary['Wrong']),
            'Validation/bad_count': float(summary['Bad']),
        }

    # ------------------------------------------------------ tokenizer probes
    @staticmethod
    def _token_to_id(tokenizer, token):
        """Resolve one special-token STRING to its id, or None.

        Tokenizers disagree on how to ask. Qwen3Tokenizer (from scratch) keeps a
        _special_to_id dict; HuggingFace answers convert_tokens_to_ids; anything
        else is encoded and accepted only if it came back as a single token
        (more than one means it was split into pieces, so it is not special).
        """
        if isinstance(token, int):
            return token

        special = getattr(tokenizer, "_special_to_id", None)
        if isinstance(special, dict) and token in special:
            return special[token]

        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            return convert(token)

        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            ids = encode(token)
            ids = getattr(ids, "ids", ids)          # tokenizers.Encoding
            if len(ids) == 1:
                return ids[0]
        return None

    @classmethod
    def _resolve_stop_ids(cls, tokenizer, stop_tokens=None):
        """What ends a generated turn, taken from the tokenizer itself.

        The baseline is whatever the tokenizer calls end-of-sequence, so a model
        from any family stops correctly with no configuration. ``stop_tokens``
        adds extras on top for a chat format with more than one turn-ender
        (Qwen3 emits <|im_end|> but also honours <|endoftext|>). Entries the
        tokenizer does not know resolve to None or -1 and are dropped, so a list
        written for one family is harmless on another.
        """
        eos = getattr(tokenizer, "eos_token_id", None)
        ids = set(eos) if isinstance(eos, (list, tuple, set)) else {eos}
        for token in (stop_tokens or ()):
            ids.add(cls._token_to_id(tokenizer, token))
        return {int(i) for i in ids
                if isinstance(i, int) and not isinstance(i, bool) and i >= 0}

    def _prompt_len(self, prompt):
        """Token length of a prompt, for the length sort and the round header.

        HuggingFace's encode() prepends BOS unless told otherwise, which would
        offset every length by one; the sort would not care but the printed
        range would be wrong.
        """
        try:
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        except TypeError:                            # tokenizer takes no such kwarg
            ids = self.tokenizer.encode(prompt)
        return len(getattr(ids, "ids", ids))

    # ---------------------------------------------------------- scheduling
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = pl_module._opt_step
        if step == 0 or step % self.every != 0 or step in self._fired:
            return
        self._fired.add(step)
        self._guarded_run(trainer, pl_module, step)

    def on_train_end(self, trainer, pl_module):
        # Final round, unless a boundary already fired close to the end.
        step = pl_module._opt_step
        if step <= 0 or step in self._fired:
            return
        if self._fired and (step - max(self._fired)) < self.every // 2:
            return
        self._fired.add(step)
        # Lightning forbids pl_module.log() from on_train_end (_FxValidator maps
        # the hook to None), so this round reaches the loggers directly.
        self._guarded_run(trainer, pl_module, step, can_log=False)

    def _guarded_run(self, trainer, pl_module, step, can_log=True):
        """Run a round; never let an eval bug destroy the training run.

        Round 1 re-raises: if it cannot run, the checkpoint callbacks monitor a
        metric that would never exist, and failing early is cheaper.
        """
        try:
            self._run(trainer, pl_module, step, can_log)
        except Exception as e:
            import traceback
            print(f"\n  [eval] round {self._round_idx} FAILED: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()
            self._set_mode(pl_module.model, inference=False)

            if self._last_metrics is None:
                print("  [eval] first round failed — re-raising.", flush=True)
                raise
            # Re-publish the previous round so ModelCheckpoint still finds its
            # monitored key and training carries on to the next boundary.
            print("  [eval] carrying forward the previous round's metrics.", flush=True)
            trainer.callback_metrics.update(
                {k: torch.tensor(v) for k, v in self._last_metrics.items()})

    # ---------------------------------------------------------- generation
    @staticmethod
    def _model_generate_batch(model, tokenizer, prompts, **kwargs):
        """Default backend: the batched decoder the model itself provides."""
        return model.generate_batch(tokenizer, prompts, **kwargs)

    def _set_mode(self, model, inference):
        """Flip the model between training and generation.

        Subclass and override when a backbone needs more than eval()/train() —
        Unsloth, for instance, needs for_inference() and an attention-impl swap.
        """
        model.eval() if inference else model.train()

    def _generate_batch(self, model, prompts):
        """Decode one chunk -> ([(text, finish_reason), ...], steps, budget)."""
        return self.generate_fn(
            model, self.tokenizer, prompts,
            max_new_tokens=self.max_new_tokens,
            max_total_len=self.max_total_len,
            eos_ids=self.eos_ids,
            pad_id=self.pad_id,
        )

    # --------------------------------------------------------------- round
    def _run(self, trainer, pl_module, step, can_log=True):
        model = pl_module.model
        was_training = model.training
        self._set_mode(model, inference=True)

        self._round_idx += 1
        round_idx = self._round_idx
        pct = 100.0 * step / self.total_steps
        n = len(self.records)
        bar = "=" * 78
        prompts = [self.prompt_fn(r["question"], include_thinking=self.thinking_mode)
                   for r in self.records]

        # Even chunks: 1319 at batch 256 becomes 6 x 220, not 5 x 256 + 1 x 39.
        # Wall time is set by the chunk COUNT (a decode step costs the same
        # whatever the batch), so evening them out is free and cuts peak memory.
        total_batches = max(1, math.ceil(n / max(1, self.batch_size)))
        bs = max(1, math.ceil(n / total_batches))

        # Group questions of similar prompt length into the same batch. Two
        # reasons, both material:
        #   1. the max_total_len clamp is per batch, computed from the LONGEST
        #      prompt in it — unsorted, one 217-token prompt would cut every
        #      other row in that batch down to the same short budget;
        #   2. left-padding is to the batch maximum, so mixed lengths pad short
        #      prompts out with dead positions that still cost KV cache.
        # Results are put back in test-set order before grading.
        enc_lens = [self._prompt_len(p) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda k: enc_lens[k])
        sorted_prompts = [prompts[k] for k in order]
        lo, hi = enc_lens[order[0]], enc_lens[order[-1]]
        print(f"\n{bar}\n"
              f"GENERATIVE EVAL  round {round_idx}  |  step {step}/{self.total_steps} "
              f"({pct:.0f}% of training)\n"
              f"{n} questions  |  batch={bs}  |  prompts {lo}-{hi} tok  "
              f"|  max_new_tokens<={self.max_new_tokens} "
              f"(clamped per batch to max_total_len={self.max_total_len} - prompt)  "
              f"|  thinking={self.thinking_mode}\n"
              f"{bar}", flush=True)

        # Release the allocator's cached blocks so the KV cache can use them.
        torch.cuda.empty_cache()

        # One line per batch — six for a full round. A live progress bar was
        # tried here and removed: the terminal redraws it several times a second
        # with a carriage return, and `script` records every redraw, so a single
        # round buried the log under thousands of near-identical lines.
        completions, i, n_batches = [], 0, 0
        started = time.time()
        while i < len(prompts):
            chunk = sorted_prompts[i:i + bs]
            t_batch = time.time()
            try:
                out, steps, budget = self._generate_batch(model, chunk)
            except torch.cuda.OutOfMemoryError:
                # Training state is still resident, so the eval batch has to fit
                # in what is left. Halve and retry rather than kill the run.
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                total_batches = max(1, math.ceil(len(prompts) / bs))
                print(f"  [eval] CUDA OOM -> retrying at batch_size={bs}", flush=True)
                continue
            completions.extend(out)
            i += len(chunk)
            n_batches += 1
            dt = time.time() - t_batch
            rate = i / max(1e-9, time.time() - started)
            stopped = sum(1 for _, f in out if f == "stop")
            print(f"  [eval] batch {n_batches}/{total_batches}  {i}/{len(prompts)} q  "
                  f"|  {steps}/{budget} tok  |  {stopped}/{len(chunk)} finished  "
                  f"|  {dt:.0f}s  |  eta {(len(prompts) - i) / max(1e-9, rate) / 60:.1f} min",
                  flush=True)
        elapsed = time.time() - started
        torch.cuda.empty_cache()

        # Undo the length sort so row i lines up with self.records[i] again.
        in_order = [None] * len(prompts)
        for pos, k in enumerate(order):
            in_order[k] = completions[pos]

        rows = [self._build_record(rec, text, finish)
                for rec, (text, finish) in zip(self.records, in_order)]

        summary = self._summarize(f"round{round_idx}", rows, elapsed)
        self._report(summary)
        self._save_rows(round_idx, rows)
        self._log_metrics(trainer, pl_module, summary, can_log)

        self._set_mode(model, inference=False)
        if not was_training:
            model.eval()

    # -------------------------------------------------------------- output
    def _report(self, summary):
        print(f"  [eval] {summary['Correct']} Correct / {summary['Wrong']} Wrong / "
              f"{summary['Bad']} Bad  ->  {summary['accuracy']:.1f}%  "
              f"in {summary['elapsed_sec'] / 60:.1f} min", flush=True)
        if summary["bad_reasons"]:
            detail = "  ".join(f"{k}={v}" for k, v in sorted(summary["bad_reasons"].items()))
            print(f"  [eval] bad reasons: {detail}", flush=True)

    def _save_rows(self, round_idx, rows):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"round{round_idx}.json"
        out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"  [eval] graded answers -> {out_path}", flush=True)

    def _log_metrics(self, trainer, pl_module, summary, can_log=True):
        """Publish one round to the progress bar, callback_metrics and loggers.

        callback_metrics is what ModelCheckpoint reads on_validation_end, so it
        is written even when pl_module.log() is off limits (on_train_end).
        """
        metrics = self._metrics_from_summary(summary)

        if can_log:
            # log_dict applies one prog_bar flag to every key, so the headline
            # accuracy is logged on its own to keep the other three off the bar.
            pl_module.log(ACCURACY_KEY, metrics[ACCURACY_KEY], logger=False,
                          sync_dist=True, on_step=True, on_epoch=False, prog_bar=True)
            pl_module.log_dict({k: v for k, v in metrics.items() if k != ACCURACY_KEY},
                               logger=False, sync_dist=True, on_step=True, on_epoch=False)

        trainer.callback_metrics.update(
            {k: torch.tensor(v) for k, v in metrics.items()})
        for lg in trainer.loggers:
            lg.log_metrics(metrics, step=trainer.global_step)
        self._last_metrics = metrics
