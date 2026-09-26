"""
Qwen3-14B GSM8K GRPO: Unsloth + TRL reference run
────────────────────────────────────────────────────────────────────────────────
A baseline for 4_Qwen3_Gsm8k_GRPO.py: the same SFT starting point, data, reward
and evaluation, trained by Unsloth's GRPOTrainer the way Unsloth's Qwen3 GRPO
notebook sets it up, instead of by the from-scratch loop.

- Model:    the merged SFT checkpoint (LoRAMergeCheckpoint, 87.11% on test),
            exported once to a HuggingFace folder by utils/hf_export.py and
            loaded in 16-bit, so the starting policy IS the SFT weights. 4-bit
            would quantize dequant(NF4(W)) + LoRA a second time, and a LoRA delta
            about the size of one NF4 step can round away there.
- Training: the notebook's recipe: LoRA r32 / alpha 64, LR 5e-6 linear with 10%
            warmup, adamw_8bit, weight decay 0.001, temperature 1.0, vLLM in
            standby mode, loss type / KL / clipping at Unsloth's defaults.
            One full epoch.
- Data:     as the SFT runs: all 7473 questions of data/Gsm8k/train.json train,
            all 1319 of test.json validate. Nothing is held out or sampled.
- Prompt:   format_prompt_only(), the raw ChatML string the SFT run trained on,
            not the tokenizer's chat template.
- Reward:   file 4's reward (1.0 correct + 0.2 format), registered as two reward
            functions the way the notebook registers its own. The notebook's
            reward functions grade its <SOLUTION> tags, which this model never
            writes.
- Eval:     Gsm8kEvalCallback's evaluation: every test question, greedy, 384 new
            tokens, graded by data/Gsm8k/metric.py into round<N>.json, every 20%
            of training, plus round 0 on the untouched SFT model. The validation
            reward is scored on those same test completions.

    python 6_Qwen3_Gsm8k_GRPO_Unsloth.py           # full epoch
"""

# Unsloth's compiled cache defaults to a RELATIVE path, so it lands in whatever
# directory the process was launched from. Pin it next to this file instead.
# Both variables are read when unsloth is imported.
import os as _os
from pathlib import Path as _Path
_os.environ.setdefault(
    "UNSLOTH_COMPILE_LOCATION",
    str(_Path(__file__).resolve().parent / "unsloth_compiled_cache"),
)
# Notebook setting: vLLM releases its KV cache while the trainer runs forward /
# backward and gets it back to generate. The weights are shared, never copied.
_os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

# Must precede trl / transformers / vllm: Unsloth patches them on import.
from unsloth import FastLanguageModel

# unsloth_zoo swaps in its own copy of vLLM's WorkerLoRAManager, which predates
# get_dummy_lora_warmup_rank. vLLM 0.20 calls it while profiling memory at
# start-up and crashes. This is vLLM's own one-line implementation of it.
from unsloth_zoo import vllm_lora_worker_manager as _unsloth_lora
if not hasattr(_unsloth_lora.WorkerLoRAManager, "get_dummy_lora_warmup_rank"):
    _unsloth_lora.WorkerLoRAManager.get_dummy_lora_warmup_rank = (
        lambda self, default_rank: self._adapter_manager.get_dummy_lora_warmup_rank(default_rank))

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import torch
import pytorch_lightning as pl
from datasets import Dataset
from vllm import SamplingParams
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

QWEN3_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = QWEN3_DIR.parents[1]
COLLECTIONS_DIR = PROJECT_ROOT / "collections"
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(COLLECTIONS_DIR))

from data.Gsm8k.data_utils import load_gsm8k_json, format_prompt_only
from data.Gsm8k.metric import Gsm8kEvalCallback, build_record, summarize, ACCURACY_KEY
from post_training.GRPO import gsm8k_reward
from utils.hf_export import ensure_hf_export
from utils.checkpoint_utils import plot_training_curves


# ------
# CONFIG
# ------
@dataclass
class CFG:
    MODEL_SIZE = "14B"
    INCLUDE_THINKING = True   # prompts end at <think>; answers are graded after </think>

    # Merged .pth from the SFT run, exported once to <stem>_hf/ next to it.
    INIT_CHECKPOINT = (
        QWEN3_DIR / "logs/Qwen3_Gsm8k_14B_QLoRA_4bit_think"
        / "20_09_26/version_0/model_pretrained/00-0.4334-87.11.pth")
    BASE_REPO = "Qwen/Qwen3-14B"   # config + tokenizer of the model the SFT run started from
    SFT_ACCURACY = 87.11           # the SFT run's own last eval round: what round 0 should reproduce

    # --- Unsloth Qwen3 GRPO notebook ---
    LOAD_IN_4BIT = False      # "False for LoRA 16bit"
    LORA_RANK = 32
    LORA_ALPHA = LORA_RANK * 2
    GPU_MEMORY_UTILIZATION = 0.9
    LR = 5e-6
    WEIGHT_DECAY = 0.001
    WARMUP_RATIO = 0.1
    LR_SCHEDULER = "linear"
    OPTIM = "adamw_8bit"
    TEMPERATURE = 1.0
    NUM_GENERATIONS = 4       # answers per question (notebook: 4)
    BATCH_SIZE = 4            # completions per micro-batch; Unsloth resets it to NUM_GENERATIONS anyway
    GRAD_ACCUM = 4            # notebook: "increase to 4 for smoother training" -> 4 prompts per step
    EPOCHS = 1                # notebook: "set to 1 for a full training run"
    MAX_STEPS = -1            # -1: EPOCHS decides
    TRAIN_SEED = 3407         # the notebook's seed

    # --- Lengths (file 4) ---
    MAX_PROMPT_LEN = 256      # train.json prompts max 241 tok, test.json max 217
    MAX_NEW_TOKENS = 384
    MAX_SEQ_LEN = MAX_PROMPT_LEN + MAX_NEW_TOKENS   # notebook: completion = seq - prompt

    # --- Reward (file 4) ---
    REWARD_WEIGHTS = {"correct": 1.0, "format": 0.2}

    # --- Evaluation (file 4) ---
    INFER_EVERY_PERCENTAGE = 0.20
    INFER_STOP_TOKENS = ("<|endoftext|>", "<|im_end|>")

    # --- Lit Logger - Experiment Tracking ---
    TEAMSPACE = "LLM-From-Scratch"


# -------
# LOGGING
# -------
def log_all(loggers, metrics, step):
    """Write one row to every logger. A failing remote logger is dropped rather
    than allowed to end a many-hour run."""
    for lg in list(loggers):
        try:
            lg.log_metrics(metrics, step=step)
            if isinstance(lg, pl.loggers.CSVLogger):
                lg.save()     # CSVLogger only flushes every 100 rows on its own
        except Exception as e:
            print(f"[logger] {type(lg).__name__} failed ({type(e).__name__}: {e}); dropping it")
            loggers.remove(lg)


class MetricsLogger(TrainerCallback):
    """Every TRL log line -> metrics.csv (and LitLogger), under file 4's names.

    Clashing names are renamed, not overwritten: TRL's `reward_std` is the mean
    std WITHIN each group, while file 4's Training/reward_std is the std over all
    of a step's rollouts, so TRL's number is kept as reward_std_in_group.
    """

    RENAME = {
        "loss": "train_loss",
        "learning_rate": "lr",
        "reward": "reward_mean",
        "reward_std": "reward_std_in_group",
        "completions/mean_length": "completion_len_mean",
        "rollout/reward_std": "reward_std",
        "rollout/advantages_mean": "advantages_mean",
        "rollout/advantages_std": "advantages_std",
        "rollout/advantages_unscaled_std": "advantages_unscaled_std",
    }

    def __init__(self, loggers):
        self.loggers = loggers
        self._step_started = None
        self._step_seconds = None

    # Listed before the eval callback, so a step's time never includes an eval round.
    def on_step_begin(self, args, state, control, **kwargs):
        self._step_started = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_started is not None:
            self._step_seconds = time.time() - self._step_started

    def on_log(self, args, state, control, logs=None, **kwargs):
        # The end-of-run summary repeats train_loss as a whole-run average; it
        # would land in the per-step loss column.
        if not logs or "train_runtime" in logs:
            return
        metrics = {f"Training/{self.RENAME.get(k, k)}": float(v)
                   for k, v in logs.items() if isinstance(v, (int, float))}
        if self._step_seconds is not None:
            metrics["Training/time/step_s"] = self._step_seconds
        free, total = torch.cuda.mem_get_info()
        metrics["Training/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        metrics["Training/gpu_used_gb"] = (total - free) / 1024 ** 3   # includes vLLM
        log_all(self.loggers, metrics, state.global_step)


# -------
# REWARDS
# -------
def make_reward_funcs(stop_ids):
    """File 4's reward as two TRL reward functions: correct 1.0 + format 0.2.

    TRL passes every dataset column as a keyword (answer) plus completion_ids,
    which carry what the decoded text cannot: whether the rollout ended on a
    stop token or ran out of budget. TRL sums the two outputs, so the total is
    exactly gsm8k_reward().total, the number file 4 trains on.
    """
    def grade(completions, completion_ids, answer):
        return [gsm8k_reward(text, gold,
                             finish_reason="stop" if ids and ids[-1] in stop_ids else "length",
                             thinking_mode=CFG.INCLUDE_THINKING,
                             weights=CFG.REWARD_WEIGHTS)
                for text, ids, gold in zip(completions, completion_ids, answer)]

    def correct_reward(completions, completion_ids, answer, **kwargs):
        return [CFG.REWARD_WEIGHTS["correct"] * g.components["correct"]
                for g in grade(completions, completion_ids, answer)]

    def format_reward(completions, completion_ids, answer, **kwargs):
        return [CFG.REWARD_WEIGHTS["format"] * g.components["format"]
                for g in grade(completions, completion_ids, answer)]

    return [correct_reward, format_reward]


# -------
# TRAINER
# -------
class GRPOTrainerWithStats(GRPOTrainer):
    """Unsloth's GRPOTrainer plus the rollout statistics file 4 logs.

    TRL logs the reward mean and the within-group std, but not file 4's
    Training/reward_std (std over the whole step) nor anything about the
    advantages. They are computed here from the rewards and advantages TRL has
    just produced for the generation batch, so the two runs line up column for
    column.

    advantages_unscaled_std is the std of (reward - group mean) BEFORE TRL
    divides by the group std. File 4 never divides (Dr. GRPO), so that is the
    number to compare with its Training/advantages_std.
    """

    def _generate_and_score_completions(self, inputs):
        mode = "train" if self.model.training else "eval"   # read before, as TRL does
        started = time.time()
        output = super()._generate_and_score_completions(inputs)
        # vLLM rollouts + rewards + reference log-probs, once per optimizer step
        self._metrics[mode]["time/generate_and_score_s"].append(time.time() - started)

        per_func = torch.tensor(
            [list(self._logs["rewards"][name]) for name in self.reward_func_names]).T
        rewards = (per_func * self.reward_weights.cpu()).sum(dim=1)
        advantages = torch.tensor(list(self._logs["advantages"]))
        if rewards.numel() and rewards.numel() % self.num_generations == 0:
            grouped = rewards.view(-1, self.num_generations)
            centered = grouped - grouped.mean(dim=1, keepdim=True)
            metrics = self._metrics[mode]
            metrics["rollout/reward_std"].append(rewards.std(unbiased=False).item())
            metrics["rollout/advantages_mean"].append(advantages.mean().item())
            metrics["rollout/advantages_std"].append(advantages.std(unbiased=False).item())
            metrics["rollout/advantages_unscaled_std"].append(centered.std(unbiased=False).item())
        return output


# ----------
# EVALUATION
# ----------
class Gsm8kVllmEval(TrainerCallback):
    """Gsm8kEvalCallback's evaluation, decoded by the trainer's own vLLM engine.

    Everything that decides the score is file 4's: the questions (all of
    test.json), greedy decoding, the 384-token budget clamped to MAX_SEQ_LEN
    minus the prompt, the stop tokens and the grader (data/Gsm8k/metric.py).
    Only the decoder differs: vLLM with the live LoRA instead of
    Qwen3Model.generate_batch.

    Rounds fire at step 0 (the untouched SFT model, which file 4 never scores),
    every every_pct of the planned steps and at the end. Each round also scores
    the same test completions with the training reward (Validation/reward_*),
    and keeps the LoRA of the best round after step 0.
    """

    def __init__(self, tokenizer, test_records, output_dir, loggers, every_pct=0.20):
        self.records = list(test_records)
        self.stop_ids = sorted(
            Gsm8kEvalCallback._resolve_stop_ids(tokenizer, CFG.INFER_STOP_TOKENS))
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.loggers = loggers
        self.every_pct = every_pct
        self.trainer = None       # set once GRPOTrainer exists: it owns the vLLM engine
        self.every = 1
        self.total = 0
        self.fired = set()
        self.rounds = []          # (step, summary, metrics) per round
        self.best = None          # (accuracy, round, step), rounds after step 0 only

    # ---------------------------------------------------------- scheduling
    def on_train_begin(self, args, state, control, **kwargs):
        self.total = state.max_steps
        self.every = max(1, round(self.total * self.every_pct))
        self._guarded_run(0)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step % self.every == 0 and step not in self.fired:
            self._guarded_run(step)

    def on_train_end(self, args, state, control, **kwargs):
        # Final round, unless a boundary already fired close to the end.
        step = state.global_step
        if step not in self.fired and step - max(self.fired, default=0) >= self.every // 2:
            self._guarded_run(step)
        self._print_history()

    def _guarded_run(self, step):
        """A failed round must not end the run. Round 0 re-raises: if it cannot
        run, no later round can either, and failing early is cheaper."""
        self.fired.add(step)
        try:
            self._run(step)
        except Exception as e:
            print(f"\n  [eval] round at step {step} FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            if not self.rounds:
                raise

    # ---------------------------------------------------------- generation
    def _decode(self, prompts):
        """Greedy decode with the current LoRA -> [(text, finish_reason, n_tokens)].

        Same wake / LoRA / sleep sequence the trainer runs around its own
        rollouts: in standby mode the KV cache only exists while vLLM is awake.
        """
        trainer = self.trainer
        llm, sleep_mode = trainer.llm, trainer.args.vllm_enable_sleep_mode
        lens = [len(self.tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]
        params = [SamplingParams(temperature=0.0,
                                 max_tokens=max(1, min(CFG.MAX_NEW_TOKENS, CFG.MAX_SEQ_LEN - n)),
                                 stop_token_ids=self.stop_ids)
                  for n in lens]
        if sleep_mode:
            torch.cuda.empty_cache()
            llm.wake_up()
        try:
            lora = trainer.model.load_lora("grpo_trainer_lora_model", load_tensors=True)
            outputs = llm.generate(prompts, params, use_tqdm=False, lora_request=lora)
        finally:
            if sleep_mode:
                llm.sleep(level=1)
        return [(o.outputs[0].text.strip(), o.outputs[0].finish_reason,
                 len(o.outputs[0].token_ids)) for o in outputs]

    # --------------------------------------------------------------- round
    def _run(self, step):
        round_idx = len(self.rounds)
        pct = 100.0 * step / max(1, self.total)
        bar = "=" * 78
        print(f"\n{bar}\n"
              f"GENERATIVE EVAL  round {round_idx}  |  step {step}/{self.total} "
              f"({pct:.0f}% of training)\n"
              f"{len(self.records)} test questions (all of test.json)  "
              f"|  vLLM greedy  |  max_new_tokens<={CFG.MAX_NEW_TOKENS}  "
              f"|  thinking={CFG.INCLUDE_THINKING}\n"
              f"{bar}", flush=True)

        prompts = [format_prompt_only(r["question"], include_thinking=CFG.INCLUDE_THINKING)
                   for r in self.records]
        started = time.time()
        outputs = self._decode(prompts)
        elapsed = time.time() - started

        rows = [build_record(rec, text, finish, CFG.INCLUDE_THINKING)
                for rec, (text, finish, _) in zip(self.records, outputs)]
        summary = summarize(f"round{round_idx}", rows, elapsed)

        rewards = torch.tensor([
            gsm8k_reward(text, rec["answer"], finish, thinking_mode=CFG.INCLUDE_THINKING,
                         weights=CFG.REWARD_WEIGHTS).total
            for rec, (text, finish, _) in zip(self.records, outputs)])
        lens = torch.tensor([float(n) for _, _, n in outputs])

        metrics = {
            ACCURACY_KEY: float(summary["accuracy"]),
            "Validation/correct_count": float(summary["Correct"]),
            "Validation/wrong_count": float(summary["Wrong"]),
            "Validation/bad_count": float(summary["Bad"]),
            "Validation/reward_mean": rewards.mean().item(),
            "Validation/reward_std": rewards.std(unbiased=False).item(),
            "Validation/completion_len_mean": lens.mean().item(),
        }

        print(f"  [eval] {summary['Correct']} Correct / {summary['Wrong']} Wrong / "
              f"{summary['Bad']} Bad  ->  {summary['accuracy']:.2f}%  "
              f"in {elapsed / 60:.1f} min", flush=True)
        if summary["bad_reasons"]:
            detail = "  ".join(f"{k}={v}" for k, v in sorted(summary["bad_reasons"].items()))
            print(f"  [eval] bad reasons: {detail}", flush=True)
        print(f"  [eval] test reward {metrics['Validation/reward_mean']:.4f} "
              f"+- {metrics['Validation/reward_std']:.4f}  |  "
              f"completion len {metrics['Validation/completion_len_mean']:.1f} tok", flush=True)
        if round_idx == 0:
            print(f"  [eval] round 0 = the SFT checkpoint as loaded; the SFT run's own "
                  f"eval scored {CFG.SFT_ACCURACY:.2f}%", flush=True)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"round{round_idx}.json"
        out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [eval] graded answers -> {out_path}", flush=True)

        acc = summary["accuracy"]
        if round_idx > 0 and (self.best is None or acc > self.best[0]):
            self.best = (acc, round_idx, step)
            best_dir = self.output_dir.parent / "best_lora"
            self.trainer.model.save_lora(str(best_dir))
            (best_dir / "eval_round.json").write_text(json.dumps(
                {"round": round_idx, "step": step, ACCURACY_KEY: acc}, indent=2))
            print(f"  [eval] best round so far -> LoRA saved to {best_dir}", flush=True)

        self.rounds.append((step, summary, metrics))
        log_all(self.loggers, metrics, step)

    def _print_history(self):
        if not self.rounds:
            return
        base = self.rounds[0][1]["accuracy"]
        bar = "=" * 78
        print(f"\n{bar}\nGRPO (Unsloth) - full test set accuracy by round\n{bar}")
        print(f"{'round':>5} {'step':>6} {'accuracy':>9} {'vs step 0':>10} "
              f"{'correct':>8} {'wrong':>6} {'bad':>5} {'test reward':>12}")
        for i, (step, s, m) in enumerate(self.rounds):
            print(f"{i:>5} {step:>6} {s['accuracy']:>8.2f}% {s['accuracy'] - base:>+10.2f} "
                  f"{s['Correct']:>8} {s['Wrong']:>6} {s['Bad']:>5} "
                  f"{m['Validation/reward_mean']:>12.4f}")
        if self.best is None:
            print("no round after step 0 beat the SFT starting point")
        else:
            print(f"best: round {self.best[1]} (step {self.best[2]}) {self.best[0]:.2f}%")
        print(bar, flush=True)


if __name__ == '__main__':

    # ---- Data Preparation (as the SFT runs: all of train.json trains, all of test.json validates)
    DATA_PATH = PROJECT_ROOT / "data" / "Gsm8k"
    train_records = load_gsm8k_json(str(DATA_PATH / "train.json"))
    test_records = load_gsm8k_json(str(DATA_PATH / "test.json"))
    print(f"Train records: {len(train_records)}, Val records: {len(test_records)}")

    prompts_per_step = CFG.BATCH_SIZE * CFG.GRAD_ACCUM // CFG.NUM_GENERATIONS
    CFG.STEPS = (CFG.MAX_STEPS if CFG.MAX_STEPS > 0
                 else len(train_records) // prompts_per_step * CFG.EPOCHS)
    save_every = max(1, round(CFG.STEPS * CFG.INFER_EVERY_PERCENTAGE))
    print(f"Optimizer steps: {CFG.STEPS} | {prompts_per_step} prompts x "
          f"{CFG.NUM_GENERATIONS} rollouts per step | eval + checkpoint every {save_every} steps")

    # ---- Model & Tokenizer: the merged SFT weights in 16-bit, a fresh LoRA on top
    model_dir = ensure_hf_export(CFG.INIT_CHECKPOINT, repo_id=CFG.BASE_REPO)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=CFG.MAX_SEQ_LEN,
        load_in_4bit=CFG.LOAD_IN_4BIT,
        fast_inference=True,              # vLLM rollouts
        max_lora_rank=CFG.LORA_RANK,
        gpu_memory_utilization=CFG.GPU_MEMORY_UTILIZATION,
        # vLLM 0.20.1's piecewise-compile pass (split_graph) crashes on this
        # Qwen3 + LoRA graph. 0 skips torch.compile; CUDA graphs are vLLM's call.
        compilation_config=0,
    )
    print(f"[init] RL starting from SFT checkpoint: {CFG.INIT_CHECKPOINT}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=CFG.LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=CFG.LORA_ALPHA,
        use_gradient_checkpointing="unsloth",
        random_state=CFG.TRAIN_SEED,
    )
    stop_ids = Gsm8kEvalCallback._resolve_stop_ids(tokenizer, CFG.INFER_STOP_TOKENS)

    # ---- Dataset: the raw ChatML prompt; "answer" reaches the reward functions
    train_dataset = Dataset.from_list([
        {"prompt": format_prompt_only(r["question"], include_thinking=CFG.INCLUDE_THINKING),
         "answer": r["answer"]}
        for r in train_records])

    # ---- Loggers (CSV + LitLogger, as file 4)
    date = datetime.now().strftime("%d_%m_%y")
    mode_tag = "QLoRA_4bit" if CFG.LOAD_IN_4BIT else "LoRA_16bit"
    think_suffix = "_think" if CFG.INCLUDE_THINKING else ""
    exp_name = f'Qwen3_Gsm8k_GRPO_Unsloth_{CFG.MODEL_SIZE}_{mode_tag}{think_suffix}'
    csv_logger = pl.loggers.CSVLogger(save_dir=str(QWEN3_DIR / 'logs' / exp_name), name=date)
    loggers = [csv_logger]

    RUN_DIR = Path(csv_logger.log_dir)
    VALIDATION_DIR = RUN_DIR / 'validation'
    run_name = f'{exp_name}_{date}_v{csv_logger.version}'
    print(f"[logger] run directory: {RUN_DIR}")
    print(f"[logger] experiment name: {run_name}")
    print(f"[eval] per-round answers -> {VALIDATION_DIR}/round<N>.json")

    metadata = {
        'model': f'Qwen3-{CFG.MODEL_SIZE}',
        'dataset': 'gsm8k',
        'method': 'grpo_unsloth',
        'init_checkpoint': str(CFG.INIT_CHECKPOINT),
        'tuning_mode': mode_tag,
        'rank': str(CFG.LORA_RANK),
        'alpha': str(CFG.LORA_ALPHA),
        'lr': str(CFG.LR),
        'group_size': str(CFG.NUM_GENERATIONS),
        'prompts_per_step': str(prompts_per_step),
        'grad_accum': str(CFG.GRAD_ACCUM),
        'temperature': str(CFG.TEMPERATURE),
        'max_new_tokens': str(CFG.MAX_NEW_TOKENS),
        'reward_weights': str(CFG.REWARD_WEIGHTS),
        'include_thinking': str(CFG.INCLUDE_THINKING),
    }
    csv_logger.log_hyperparams(metadata)
    try:
        lit_logger = pl.loggers.LitLogger(
            name=run_name, teamspace=CFG.TEAMSPACE, root_dir=str(RUN_DIR),
            save_logs=False, metadata=metadata)
        lit_logger._version = str(csv_logger.version)
        loggers.append(lit_logger)
        print(f"[logger]   teamspace: {CFG.TEAMSPACE}")
    except Exception as e:
        print(f"[logger] LitLogger unavailable ({type(e).__name__}: {e}); CSV only")

    # ---- GRPO config: the notebook's, with file 4's lengths and group size
    training_args = GRPOConfig(
        # Notebook, verbatim. This Unsloth build forwards only list-valued fields
        # of vllm_sampling_params (stop), so rollouts are plain temperature-1.0
        # sampling: the same behaviour policy file 4 samples from.
        vllm_sampling_params=SamplingParams(
            min_p=0.1, top_p=1.0, top_k=-1, seed=3407,
            stop=[tokenizer.eos_token], include_stop_str_in_output=True),
        temperature=CFG.TEMPERATURE,
        learning_rate=CFG.LR,
        weight_decay=CFG.WEIGHT_DECAY,
        warmup_steps=CFG.WARMUP_RATIO,    # transformers 5 takes the notebook's warmup_ratio here
        lr_scheduler_type=CFG.LR_SCHEDULER,
        optim=CFG.OPTIM,
        logging_steps=1,
        per_device_train_batch_size=CFG.BATCH_SIZE,
        gradient_accumulation_steps=CFG.GRAD_ACCUM,
        num_generations=CFG.NUM_GENERATIONS,
        max_prompt_length=CFG.MAX_PROMPT_LEN,
        max_completion_length=CFG.MAX_NEW_TOKENS,
        num_train_epochs=CFG.EPOCHS,
        max_steps=CFG.MAX_STEPS,
        save_steps=save_every,
        save_total_limit=2,
        report_to="none",
        output_dir=str(RUN_DIR / 'checkpoints'),
        seed=CFG.TRAIN_SEED,
    )

    boxed_eval = Gsm8kVllmEval(
        tokenizer, test_records,
        output_dir=VALIDATION_DIR,
        loggers=loggers,
        every_pct=CFG.INFER_EVERY_PERCENTAGE,
    )

    # Unsloth writes its LoRA hand-off folder (grpo_trainer_lora_model/) to the
    # current directory; keep it inside the run instead of the source tree.
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(RUN_DIR)
    trainer = GRPOTrainerWithStats(
        model=model,
        processing_class=tokenizer,
        reward_funcs=make_reward_funcs(stop_ids),
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[MetricsLogger(loggers), boxed_eval],
    )
    boxed_eval.trainer = trainer

    a = trainer.args
    print(f"[grpo] loss_type={a.loss_type} beta={a.beta} epsilon={a.epsilon}/{a.epsilon_high} "
          f"scale_rewards={a.scale_rewards} num_iterations={a.num_iterations} "
          f"mask_truncated={a.mask_truncated_completions}")
    print(f"[grpo] lr={a.learning_rate} {a.lr_scheduler_type} warmup={a.warmup_steps} "
          f"optim={a.optim} weight_decay={a.weight_decay} max_grad_norm={a.max_grad_norm}")
    print(f"[grpo] generation_batch={a.generation_batch_size} "
          f"steps_per_generation={a.steps_per_generation} stop_ids={sorted(stop_ids)} "
          f"vllm_sleep_mode={a.vllm_enable_sleep_mode}")

    trainer.train()

    # ---- Save + Plot Results ----
    final_dir = RUN_DIR / 'final_lora'
    model.save_lora(str(final_dir))
    print(f"Final LoRA saved to: {final_dir}")
    for lg in loggers:
        try:
            lg.finalize("success")
        except Exception as e:
            print(f"[logger] {type(lg).__name__}.finalize failed: {e}")
    plot_training_curves(csv_logger)
