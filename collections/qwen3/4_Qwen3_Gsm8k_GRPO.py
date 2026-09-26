import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLLECTIONS_DIR = PROJECT_ROOT / "collections"
MODELS_DIR = Path(__file__).resolve().parent / "models"
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(COLLECTIONS_DIR))
sys.path.append(str(MODELS_DIR))

import json
import math
import time
import bitsandbytes as bnb
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('high')

from data.Gsm8k.data_utils import load_gsm8k_json
from data.Gsm8k.metric import build_record, summarize, ACCURACY_KEY
from utils.checkpoint_utils import plot_training_curves

from post_training.GRPO import (
    PromptDataset, prompt_collate_fn,
    group_advantages, grpo_loss, token_logprobs,
    sample_rollouts, reward_groups, gsm8k_reward,
)
from post_training.GRPO.rewards import strip_end_tokens

from qwen3_dense import (
    Qwen3Model, from_pretrained, from_local_pth,
    QWEN_06B_CFG, QWEN_1B7_CFG, QWEN_4B_CFG,
    QWEN_8B_CFG, QWEN_14B_CFG, QWEN_32B_CFG
)
from qwen_tokenizer import Qwen3Tokenizer


# ------
# CONFIG
# ------
QWEN3_MODELS = {
    "0.6B": (QWEN_06B_CFG, "Qwen/Qwen3-0.6B"),
    "1.7B": (QWEN_1B7_CFG, "Qwen/Qwen3-1.7B"),
    "4B":   (QWEN_4B_CFG,  "Qwen/Qwen3-4B"),
    "8B":   (QWEN_8B_CFG,  "Qwen/Qwen3-8B"),
    "14B":  (QWEN_14B_CFG, "Qwen/Qwen3-14B"),
    "32B":  (QWEN_32B_CFG, "Qwen/Qwen3-32B"),
}


@dataclass
class CFG:
    """Defaults mirror the Unsloth baseline (6_Qwen3_Gsm8k_GRPO_Unsloth.py,
    87.41% -> 92.42% on test.json), so the two runs differ only in the
    implementation: every curve should line up step for step."""
    EPOCHS = 1
    WORKERS = 2
    SEED = 1001

    MODEL_SIZE = "14B"        # "0.6B" | "1.7B" | "4B" | "8B" | "14B" | "32B"
    INCLUDE_THINKING = True   # GRPO on reasoning: rollouts must emit </think> then \boxed{}

    # Merged .pth from an SFT run (LoRAMergeCheckpoint output). RL from a base
    # model rarely produces \boxed{} at all, so every reward is 0 and no group
    # ever has a gradient. None falls back to the pretrained weights.
    INIT_CHECKPOINT = str(
        PROJECT_ROOT / "collections/qwen3/logs/Qwen3_Gsm8k_14B_QLoRA_4bit_think"
        / "20_09_26/version_0/model_pretrained/00-0.4334-87.11.pth")

    # --- Rollouts ---
    # 4 prompts x 4 answers = 16 rollouts per optimizer step -> 1868 steps per
    # epoch, the Unsloth run's layout (batch 4 x grad-accum 4 / 4 generations).
    PROMPTS_PER_STEP = 4
    GROUP_SIZE = 4            # G rollouts per prompt; G >= 4 keeps whole-group collapse rare
    GRAD_ACCUM = 1            # generation calls per optimizer step
    MAX_PROMPT_LEN = 256      # train.json prompts: p99 = 157 tok, max = 241

    # Completion cap, constant by design. GSM8K thinking-mode ground truth runs
    # p99 = 221 and max = 346 tokens, so 384 clears everything the SFT policy was
    # trained on with room to spare while keeping truncation rare.
    MAX_NEW_TOKENS = 384
    TEMPERATURE = 1.0         # must be > 0 or every rollout in a group is identical
    TOP_P = 1.0               # 1.0 keeps the behaviour policy equal to the scored policy

    # --- GRPO objective (Unsloth's run: loss_type=bnpo, scale_rewards=group, beta=0.001) ---
    LOSS_TYPE = "bnpo"        # "bnpo": mean over completion tokens | "dr_grpo": sum / (N * MAX_NEW_TOKENS)
    SCALE_ADV_BY_STD = True   # (r - mean) / (std + 1e-4) per group. False = Dr. GRPO
    KL_BETA = 0.001           # k3 KL to the SFT policy (LoRA off). 0 skips the reference forward
    CLIP_EPS_LOW = 0.2
    CLIP_EPS_HIGH = 0.28
    LOGPROB_CHUNK = 2         # Chunk rollouts - forward+backward loss per chunk logprob - save Vram

    # --- Reward ---
    REWARD_WEIGHTS = {"correct": 1.0, "format": 0.2}
    BAD_PENALTY = 0.0
    TRUNCATION_PENALTY = 0.0

    # --- LoRA (Unsloth run: r32 / alpha 64 on q k v o gate up down) ---
    RANK = 32
    ALPHA = 64
    # "lora" keeps the base in bf16: bit-identical to the weights the Unsloth run
    # started from. "qlora" re-quantizes the merged SFT weights to NF4, and that
    # rounding error is 70-97% the size of the whole SFT delta (measured on
    # layers 0/20/39), so RL would start from a different policy than SFT's.
    TUNING_MODE = "lora"      # "lora" | "qlora"
    QUANT_BITS = "4bit"       # "4bit" | "8bit"
    LORAPLUS_RATIO = None

    # --- Optimizer (Unsloth run: adamw_8bit, betas 0.9/0.999, wd 0.001) ---
    OPTIM = "adamw_8bit"      # "adamw_8bit" (bitsandbytes) | "adamw" (torch, fp32 states)
    LR = 5e-6
    BETAS = (0.9, 0.999)
    WEIGHT_DECAY = 0.001
    LR_SCHEDULER = "linear"   # "linear": HF warmup then linear to 0 | "cosine": to MIN_LR
    MIN_LR = LR * 0.1         # cosine only
    WARMUP_RATIO = 0.1        # of the optimizer steps (HF rounds it up: 187 of 1868)
    GRAD_NORM = 1.0

    # --- Validation: greedy decode of ALL of test.json, once per round ---
    # Round 0 runs before the first step (the untouched SFT policy), then every
    # VAL_EVERY_PERCENTAGE of training. One decode feeds every Validation/ metric.
    VAL_BATCH_SIZE = 256
    VAL_EVERY_PERCENTAGE = 0.20

    # --- Lit Logger - Experiment Tracking ---
    TEAMSPACE = "LLM-From-Scratch"

# -----------------
# LOW-RANK ADAPTION
# -----------------

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        # Kaiming on the (rank, in_dim) layout, then transposed: fan_in must be
        # in_dim, as for nn.Linear(in_dim, rank) / PEFT's lora_A. Initialising
        # the (in_dim, rank) tensor directly takes fan_in = rank, which makes A
        # sqrt(in_dim / rank) = 13-23x larger and every update to B that much
        # bigger in weight space.
        A = torch.empty(rank, in_dim)
        torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        self.A = torch.nn.Parameter(A.T.contiguous())
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.scaling * (x @ self.A @ self.B)


class LinearWithLoRA(torch.nn.Module):
    """`enabled` is what makes the reference policy free: switch every adapter
    off and the frozen base weights are pi_ref."""

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        self.enabled = True

    def forward(self, x):
        out = self.linear(x)
        return out + self.lora(x) if self.enabled else out


def apply_lora(model, rank, alpha, target_modules):
    def replace_linear_with_lora(module, rank, alpha, target_modules=None):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                if target_modules is None or name in target_modules:
                    setattr(module, name, LinearWithLoRA(child, rank, alpha))
            else:
                replace_linear_with_lora(child, rank, alpha, target_modules)

    for param in model.parameters():
        param.requires_grad = False

    replace_linear_with_lora(model, rank=rank, alpha=alpha, target_modules=target_modules)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable LoRA params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    return model


# -----------------
# 4BIT QUANTIZATION
# -----------------

def apply_4bit_quantization(model, skip_modules=None, compute_dtype=torch.bfloat16,
                            quant_type="nf4"):
    if skip_modules is None:
        skip_modules = {"out_head", "tok_emb"}

    for name, module in model.named_children():
        if name in skip_modules:
            continue

        if isinstance(module, LinearWithLoRA):
            old_linear = module.linear
            new_linear = bnb.nn.Linear4bit(
                old_linear.in_features, old_linear.out_features,
                bias=old_linear.bias is not None,
                compute_dtype=compute_dtype, quant_type=quant_type,
            )
            new_linear.load_state_dict(old_linear.state_dict())
            new_linear.requires_grad_(False)
            module.linear = new_linear

        elif isinstance(module, torch.nn.Linear):
            if not any(p.requires_grad for p in module.parameters()):
                new_linear = bnb.nn.Linear4bit(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=compute_dtype, quant_type=quant_type,
                )
                new_linear.load_state_dict(module.state_dict())
                new_linear.requires_grad_(False)
                setattr(model, name, new_linear)
        else:
            apply_4bit_quantization(module, skip_modules, compute_dtype, quant_type)

    return model


# -----------------
# 8BIT QUANTIZATION
# -----------------

def apply_8bit_quantization(model, skip_modules=None, threshold=6.0):
    if skip_modules is None:
        skip_modules = {"out_head", "tok_emb"}

    for name, module in model.named_children():
        if name in skip_modules:
            continue

        if isinstance(module, LinearWithLoRA):
            old_linear = module.linear
            new_linear = bnb.nn.Linear8bitLt(
                old_linear.in_features, old_linear.out_features,
                bias=old_linear.bias is not None,
                has_fp16_weights=False, threshold=threshold,
            )
            new_linear.load_state_dict(old_linear.state_dict())
            new_linear.requires_grad_(False)
            module.linear = new_linear

        elif isinstance(module, torch.nn.Linear):
            if not any(p.requires_grad for p in module.parameters()):
                new_linear = bnb.nn.Linear8bitLt(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False, threshold=threshold,
                )
                new_linear.load_state_dict(module.state_dict())
                new_linear.requires_grad_(False)
                setattr(model, name, new_linear)
        else:
            apply_8bit_quantization(module, skip_modules, threshold)

    return model


# ---------------------
# LIGHTNING DATA MODULE
# ---------------------

class LightningDataModule(pl.LightningDataModule):
    """Prompts only. A batch is (list[str] prompts, list[dict] records); the
    rollout sampler tokenises, because only it knows the group width."""

    def __init__(self, tokenizer, train_records, val_records, cfg):
        super().__init__()
        self.train_dataset = PromptDataset(
            train_records, tokenizer,
            max_prompt_len=cfg.MAX_PROMPT_LEN, include_thinking=cfg.INCLUDE_THINKING
        )
        self.val_dataset = PromptDataset(
            val_records, tokenizer,
            max_prompt_len=cfg.MAX_PROMPT_LEN, include_thinking=cfg.INCLUDE_THINKING
        )
        print(f"Train prompts (after prompt_len filter): {len(self.train_dataset)}")
        print(f"Val prompts (test.json, after prompt_len filter): {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=CFG.PROMPTS_PER_STEP,
            collate_fn=prompt_collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=CFG.WORKERS
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=CFG.VAL_BATCH_SIZE,
            collate_fn=prompt_collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=CFG.WORKERS
        )


# ------------------------
# LIGHTNING TRAIN PIPELINE
# ------------------------

class Qwen3_GRPO_Lightning(pl.LightningModule):
    def __init__(self, model, tokenizer, validation_dir):
        super().__init__()
        self.automatic_optimization = False

        self.model = model
        self.tokenizer = tokenizer
        self.validation_dir = Path(validation_dir)
        self._opt_step = 0
        self._lora_modules = []

        # Accumulated over the GRAD_ACCUM micro-batches of one optimizer step
        self._reset_accumulators()

        # Accumulated over one validation pass (test.json)
        self._val_round = 0
        self._val_rows = []
        self._val_started = None

    def _reset_accumulators(self):
        self._accum_loss = 0.0
        self._accum_policy_loss = 0.0
        self._accum_kl = []
        self._accum_rewards = []
        self._accum_advantages = []
        self._accum_gen_lens = []
        self._accum_finish = []
        self._step_started = None

    def configure_model(self):
        # Lightning calls this hook on every trainer entry point; wrap only once.
        if self._lora_modules:
            return

        precision_to_dtype = {
            "16": torch.float16, "16-mixed": torch.float16,
            "bf16": torch.bfloat16, "bf16-mixed": torch.bfloat16,
            "32": torch.float32,
        }
        self.model.dtype = precision_to_dtype.get(str(self.trainer.precision), torch.float32)

        self.model = apply_lora(
            self.model, rank=CFG.RANK, alpha=CFG.ALPHA,
            target_modules=["W_query", "W_key", "W_value", "out_proj", "fc1", "fc2", "fc3"]
        )
        self._lora_modules = [m for m in self.model.modules() if isinstance(m, LinearWithLoRA)]

        if CFG.TUNING_MODE == "lora":
            return
        if CFG.TUNING_MODE != "qlora":
            raise ValueError(f"Unknown TUNING_MODE: {CFG.TUNING_MODE}. Use 'lora' or 'qlora'.")

        if CFG.QUANT_BITS == "4bit":
            self.model = apply_4bit_quantization(self.model)
        elif CFG.QUANT_BITS == "8bit":
            self.model = apply_8bit_quantization(self.model)
        else:
            raise ValueError(f"Unknown QUANT_BITS: {CFG.QUANT_BITS}. Use '4bit' or '8bit'.")

    @contextmanager
    def lora_disabled(self):
        """The reference policy pi_ref: the SFT weights, every adapter off."""
        for m in self._lora_modules:
            m.enabled = False
        try:
            yield
        finally:
            for m in self._lora_modules:
                m.enabled = True

    # --------------------------------------------------------------- training
    def backward_loss_chunk(self, rollouts, advantages):
        r"""
        Forward + backward a few rollouts at a time to save Vram
        Chunk rollouts - Calculate token-probs + loss - Backward each chunk

        Example:
            row 0  (group 0, advantage +0.00)
            text        <pad>  <pad>      2      +      3      =  |      \  boxed      {      5      }  <eos>  <pad>  <pad>  <pad>
            id         151643 151643     17     10     18     28  |     59  79075     90     20     92 151645 151643 151643 151643
            pad_mask        1      1      0      0      0      0  |      0      0      0      0      0      0      1      1      1
            logprob_mask           0      0      0      0      0  |      1      1      1      1      1      1      0      0      0

        Returns (loss, policy_loss, kl_per_seq list) summed over the chunks.
        """
        logprob_mask = rollouts.logprob_mask
        num_tokens = int(logprob_mask.sum())      # bnpo: every chunk divides by the whole batch's tokens
        total_loss, policy_loss, kl_per_seq = 0.0, 0.0, []

        for start in range(0, rollouts.n_rollouts, CFG.LOGPROB_CHUNK):
            # 0. Chunk and pick group n-row (For example: slice(0,2) -> pick row[0,1])
            chunk = slice(start, start + CFG.LOGPROB_CHUNK)

            # 1. Skip group rollouts if:
            # Sum logprob_mask of groups is 0 (easy understand that no generated tokens)
            # Or max value of advantages groups is 0 (easy understand all answers wrong or all correct). Gradient useless 
            if logprob_mask[chunk].sum() == 0 or advantages[chunk].abs().max() == 0:
                continue

            # 2. Init value before loss. Drop the columns that are padding in
            #    EVERY row of the chunk: the batch is padded to its longest
            #    prompt and longest completion, which this chunk may not have.
            #    RoPE is relative, so shifting the whole chunk left changes no score.
            pad_mask = rollouts.pad_mask[chunk]
            real_cols = (~pad_mask).any(dim=0).nonzero().squeeze(-1)
            lo, hi = int(real_cols[0]), int(real_cols[-1]) + 1
            sequences = rollouts.sequences[chunk, lo:hi]
            pad_mask = pad_mask[:, lo:hi]
            chunk_mask, chunk_adv = logprob_mask[chunk, lo:hi - 1], advantages[chunk]

            # 3. Token logprobs of group rollouts (+ the reference policy's, for the KL)
            logprobs = token_logprobs(
                self.model,
                sequences,
                pad_mask=pad_mask,
                temperature=CFG.TEMPERATURE
            )
            ref_logprobs = None
            if CFG.KL_BETA:
                with torch.no_grad(), self.lora_disabled():
                    ref_logprobs = token_logprobs(
                        self.model, sequences, pad_mask=pad_mask,
                        temperature=CFG.TEMPERATURE)

            # 4. GRPO loss
            loss, stats = grpo_loss(
                logprobs, chunk_adv, chunk_mask,
                max_completion_len=CFG.MAX_NEW_TOKENS,
                num_rollouts=rollouts.n_rollouts,
                grad_accum=CFG.GRAD_ACCUM,
                clip_eps_low=CFG.CLIP_EPS_LOW,
                clip_eps_high=CFG.CLIP_EPS_HIGH,
                loss_type=CFG.LOSS_TYPE,
                num_tokens=num_tokens,
                ref_logprobs=ref_logprobs,
                kl_beta=CFG.KL_BETA,
                return_stats=True,
            )

            # 5. Backward loss of chunked group rollouts
            self.manual_backward(loss)
            total_loss += loss.item()
            policy_loss += stats["policy_loss"].item()
            
            if stats["kl_per_seq"] is not None:
                kl_per_seq.extend(stats["kl_per_seq"][chunk_mask.any(dim=1)].tolist())

        return total_loss, policy_loss, kl_per_seq


    def training_step(self, batch, batch_idx):
        prompts, records = batch
        opt, sch = self.optimizers(), self.lr_schedulers()
        if self._step_started is None:
            self._step_started = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        # 1. Sampling the rollouts
        rollouts = sample_rollouts(
            self.model, self.tokenizer, prompts, records,
            group_size=CFG.GROUP_SIZE,
            max_new_tokens=CFG.MAX_NEW_TOKENS,
            temperature=CFG.TEMPERATURE,
            top_p=CFG.TOP_P,
        )

        # 2. Reward group rollouts
        rewards = reward_groups(
            completions=rollouts.texts,
            finish_reasons=rollouts.finish_reasons,
            answers=[r["answer"] for r in records],
            group_size=CFG.GROUP_SIZE,
            thinking_mode=CFG.INCLUDE_THINKING,
            weights=CFG.REWARD_WEIGHTS,
            bad_penalty=CFG.BAD_PENALTY,
            truncation_penalty=CFG.TRUNCATION_PENALTY,
        )
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # 3. Advantages group rollouts
        advantages, _ = group_advantages(
            rewards,
            CFG.GROUP_SIZE,
            scale_by_std=CFG.SCALE_ADV_BY_STD
        )

        # 4. Loss group rollouts
        loss, policy_loss, kl_per_seq = self.backward_loss_chunk(rollouts, advantages)

        # 5. Tracking the rewards - advantages - polices - KL - loss - policy loss
        self._accum_loss += loss
        self._accum_policy_loss += policy_loss
        self._accum_kl.extend(kl_per_seq)
        self._accum_rewards.append(rewards)
        self._accum_advantages.append(advantages.detach())
        self._accum_gen_lens.extend(rollouts.gen_lens)
        self._accum_finish.extend(rollouts.finish_reasons)

        # 6. Optimizer step after gradient accumulation
        if (batch_idx + 1) % CFG.GRAD_ACCUM == 0:
            self._optimizer_step(opt, sch)


    def _optimizer_step(self, opt, sch):
        # 1. Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=CFG.GRAD_NORM)

        # 2. Optimize + Scheduler (log the LR this step actually used)
        lr_now = opt.param_groups[0]['lr']
        opt.step()
        opt.zero_grad()
        sch.step()
        self._opt_step += 1

        # 3. Statistics over every rollout of the optimizer step, not a mean of
        #    per-micro-batch stds
        rewards = torch.cat(self._accum_rewards)
        advantages = torch.cat(self._accum_advantages)
        completion_len_mean = sum(self._accum_gen_lens) / len(self._accum_gen_lens)
        kl = sum(self._accum_kl) / len(self._accum_kl) if self._accum_kl else 0.0

        # 4. How every rollout of the step ended. A rollout either emits EOS or
        #    runs out of MAX_NEW_TOKENS, so the two always sum to
        #    PROMPTS_PER_STEP x GROUP_SIZE x GRAD_ACCUM (= 16).
        by_completion = sum(f == "stop" for f in self._accum_finish)
        by_max_length = len(self._accum_finish) - by_completion

        # 5. Log — reward, advantage, loss terms, completion, LR, grad-norm and memory
        gpu_mem_gb = (
            torch.cuda.max_memory_allocated() / 1024 ** 3
            if torch.cuda.is_available() else 0.0
        )
        step_s = time.time() - self._step_started

        extra = {
            'Training/reward_mean': rewards.mean().item(),
            'Training/reward_std': rewards.std(unbiased=False).item(),
            'Training/advantages_mean': advantages.mean().item(),
            'Training/advantages_std': advantages.std(unbiased=False).item(),
            'Training/train_loss': self._accum_loss,
            'Training/policy_loss': self._accum_policy_loss,
            'Training/kl': kl,
            'Training/grad_norm': grad_norm.item(),
            'Training/lr': lr_now,
            'Training/completion_len_mean': completion_len_mean,
            'Training/gpu_mem_gb': gpu_mem_gb,
            'Training/time_step_s': step_s,
            'Completion/answer_terminated_by_completion': float(by_completion),
            'Completion/answer_terminated_by_max_length': float(by_max_length),
        }
        if CFG.LORAPLUS_RATIO is not None and len(opt.param_groups) > 1:
            extra['Training/lr_lora_B'] = opt.param_groups[1]['lr']

        self._log_to_loggers(extra)

        # 6. One line per step: the Rich progress bar prints nothing through `tee`
        print(f"[step {self._opt_step}/{CFG.STEPS}] reward {rewards.mean().item():.3f}"
              f"+-{extra['Training/reward_std']:.3f} | adv std {extra['Training/advantages_std']:.3f}"
              f" | loss {self._accum_loss:+.4f} (policy {self._accum_policy_loss:+.4f}, kl {kl:.5f})"
              f" | gn {grad_norm.item():.3f} | lr {lr_now:.2e} | len {completion_len_mean:.0f}"
              f" | eos {by_completion}/{len(self._accum_finish)} | {gpu_mem_gb:.1f} GB | {step_s:.1f}s",
              flush=True)

        # 7. Reset accumulators
        self._reset_accumulators()


    def _log_to_loggers(self, metrics):
        """Write a row at step = completed optimizer steps, the x-axis the
        Unsloth run logs on. self.log() would stamp it with Lightning's
        batch counter, which lags one step behind in manual optimization."""
        for lg in self.trainer.loggers:
            lg.log_metrics(metrics, step=self._opt_step)

    def on_train_epoch_end(self):
        # Flush any leftover accumulated gradients at epoch boundary
        if self._accum_rewards:
            opt = self.optimizers()
            sch = self.lr_schedulers()
            self._optimizer_step(opt, sch)

    # ------------------------------------------------------------- validation
    def on_validation_epoch_start(self):
        self._val_rows = []
        self._val_started = time.time()
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        prompts, records = batch

        # Step 1: Rollouts sampling (greedy)
        rollouts = sample_rollouts(
            self.model, self.tokenizer, prompts, records,
            group_size=1,
            max_new_tokens=CFG.MAX_NEW_TOKENS,
            temperature=0.0,
            top_p=CFG.TOP_P,
        )

        # Step 2: Grade (Correct / Wrong / Bad) + reward, per question
        for rec, text, finish, n_tokens in zip(
                records, rollouts.texts, rollouts.finish_reasons, rollouts.gen_lens):
            completion = strip_end_tokens(text).strip()
            row = build_record(rec, completion, finish, CFG.INCLUDE_THINKING)
            row["reward"] = gsm8k_reward(
                completion, rec["answer"], finish,
                thinking_mode=CFG.INCLUDE_THINKING,
                weights=CFG.REWARD_WEIGHTS,
                bad_penalty=CFG.BAD_PENALTY,
                truncation_penalty=CFG.TRUNCATION_PENALTY,
            ).total
            row["completion_tokens"] = n_tokens
            self._val_rows.append(row)

    def on_validation_epoch_end(self):
        rows = self._val_rows
        if not rows:
            return
        elapsed = time.time() - self._val_started
        round_idx = self._val_round
        summary = summarize(f"round{round_idx}", rows, elapsed)
        rewards = torch.tensor([r["reward"] for r in rows], dtype=torch.float32)
        lengths = torch.tensor([r["completion_tokens"] for r in rows], dtype=torch.float32)

        metrics = {
            ACCURACY_KEY: float(summary["accuracy"]),
            'Validation/correct_count': float(summary["Correct"]),
            'Validation/wrong_count': float(summary["Wrong"]),
            'Validation/bad_count': float(summary["Bad"]),
            'Validation/answer_completion_length_mean': lengths.mean().item(),
            'Validation/reward_mean': rewards.mean().item(),
            'Validation/reward_std': rewards.std(unbiased=False).item(),
        }

        bar = "=" * 78
        print(f"\n{bar}\nVALIDATION  round {round_idx}  |  step {self._opt_step}/{CFG.STEPS}  |  "
              f"{len(rows)} test questions  |  greedy  |  {elapsed / 60:.1f} min\n"
              f"  {summary['Correct']} Correct / {summary['Wrong']} Wrong / {summary['Bad']} Bad"
              f"  ->  {summary['accuracy']:.2f}%\n"
              f"  reward {metrics['Validation/reward_mean']:.4f} +- {metrics['Validation/reward_std']:.4f}"
              f"  |  completion len {metrics['Validation/answer_completion_length_mean']:.1f} tok", flush=True)
        if summary["bad_reasons"]:
            detail = "  ".join(f"{k}={v}" for k, v in sorted(summary["bad_reasons"].items()))
            print(f"  bad reasons: {detail}", flush=True)

        self.validation_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.validation_dir / f"round{round_idx}.json"
        out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  graded answers -> {out_path}\n{bar}", flush=True)

        # Rows go to the loggers directly at the optimizer step (round 0 -> step 0:
        # Lightning discards every self.log() of the sanity pass anyway).
        # self.log(logger=False) only feeds ModelCheckpoint's callback_metrics.
        self._log_to_loggers(metrics)
        if not self.trainer.sanity_checking:
            self.log(ACCURACY_KEY, metrics[ACCURACY_KEY], sync_dist=True, prog_bar=True,
                     logger=False)
            self.log_dict({k: v for k, v in metrics.items() if k != ACCURACY_KEY},
                          sync_dist=True, prog_bar=False, logger=False)

        self._val_round += 1
        self._val_rows = []
        torch.cuda.empty_cache()

    # ------------------------------------------------------------- checkpoint
    def on_save_checkpoint(self, checkpoint):
        # Keep only what training changed: the LoRA adapters (~0.5 GB). The
        # frozen base is CFG.INIT_CHECKPOINT, already on disk; saving it again
        # would cost ~30 GB per checkpoint. Reload with load_state_dict(strict=False).
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items() if k in trainable}

    # -------------------------------------------------------------- optimizer
    def _make_adamw(self, optim_groups, lr):
        if CFG.OPTIM == "adamw_8bit":
            return bnb.optim.AdamW8bit(optim_groups, lr=lr, betas=CFG.BETAS, eps=1e-8)
        if CFG.OPTIM == "adamw":
            return torch.optim.AdamW(optim_groups, lr=lr, betas=CFG.BETAS, eps=1e-8)
        raise ValueError(f"Unknown OPTIM: {CFG.OPTIM}. Use 'adamw_8bit' or 'adamw'.")

    def loraplus_optimizer(self):
        lr, lr_ratio = CFG.LR, CFG.LORAPLUS_RATIO
        group_A, group_B, group_B_nodecay = [], [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if '.lora.B' in name or param.ndim == 1:
                (group_B if param.ndim >= 2 else group_B_nodecay).append(param)
            else:
                group_A.append(param)

        optim_groups = [
            {'params': group_A, 'lr': lr, 'weight_decay': CFG.WEIGHT_DECAY},
            {'params': group_B, 'lr': lr * lr_ratio, 'weight_decay': CFG.WEIGHT_DECAY},
            {'params': group_B_nodecay, 'lr': lr * lr_ratio, 'weight_decay': 0.0},
        ]
        optim_groups = [g for g in optim_groups if len(g['params']) > 0]
        return self._make_adamw(optim_groups, lr)

    def standard_optimizer(self):
        params = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in params.values() if p.dim() >= 2]
        nodecay_params = [p for p in params.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': CFG.WEIGHT_DECAY},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        optim_groups = [g for g in optim_groups if len(g['params']) > 0]
        return self._make_adamw(optim_groups, CFG.LR)

    def configure_optimizers(self):
        optimizer = (self.loraplus_optimizer() if CFG.LORAPLUS_RATIO is not None
                     else self.standard_optimizer())

        def linear_lambda(current_step):
            # transformers.get_linear_schedule_with_warmup: step 0 runs at LR 0
            if current_step < CFG.WARMUP_STEPS:
                return current_step / max(1, CFG.WARMUP_STEPS)
            return max(0.0, (CFG.STEPS - current_step) / max(1, CFG.STEPS - CFG.WARMUP_STEPS))

        def cosine_lambda(current_step):
            if current_step < CFG.WARMUP_STEPS:
                return (current_step + 1) / CFG.WARMUP_STEPS
            if current_step > CFG.STEPS:
                return CFG.MIN_LR / CFG.LR
            decay_ratio = (current_step - CFG.WARMUP_STEPS) / (CFG.STEPS - CFG.WARMUP_STEPS)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return (CFG.MIN_LR + coeff * (CFG.LR - CFG.MIN_LR)) / CFG.LR

        lr_lambda = {"linear": linear_lambda, "cosine": cosine_lambda}[CFG.LR_SCHEDULER]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]


if __name__ == '__main__':

    pl.seed_everything(CFG.SEED, workers=True)

    # ---- Data Preparation (as the SFT runs: all of train.json trains, all of test.json validates)
    DATA_PATH = PROJECT_ROOT / "data" / "Gsm8k"
    train_records = load_gsm8k_json(str(DATA_PATH / "train.json"))
    test_records = load_gsm8k_json(str(DATA_PATH / "test.json"))
    print(f"Train records: {len(train_records)}, Val records: {len(test_records)}")

    CFG.STEPS = (len(train_records) // CFG.PROMPTS_PER_STEP // CFG.GRAD_ACCUM) * CFG.EPOCHS
    CFG.WARMUP_STEPS = math.ceil(CFG.STEPS * CFG.WARMUP_RATIO)
    print(f"Optimizer steps per epoch: {CFG.STEPS} (warmup {CFG.WARMUP_STEPS})")
    print(f"Rollouts per optimizer step: "
          f"{CFG.PROMPTS_PER_STEP * CFG.GROUP_SIZE * CFG.GRAD_ACCUM}")

    # Floor, not round: Lightning validates only when the batch count is an exact
    # multiple, and round(1868 * 0.2) = 374 puts the 5th round at 1870 > 1868,
    # so the end of training would never be scored or checkpointed.
    CFG.VAL_EVERY_N_STEPS = CFG.GRAD_ACCUM * max(
        1, int(CFG.STEPS * CFG.VAL_EVERY_PERCENTAGE))
    print(f"Validation every {CFG.VAL_EVERY_N_STEPS} batches "
          f"({CFG.VAL_EVERY_PERCENTAGE:.0%} of training) + round 0 before the first step")

    # ---- Model & Tokenizer
    model_cfg, repo_id = QWEN3_MODELS[CFG.MODEL_SIZE]
    model = Qwen3Model(model_cfg)
    if CFG.INIT_CHECKPOINT:
        model = from_local_pth(model, CFG.INIT_CHECKPOINT)
        print(f"[init] RL starting from SFT checkpoint: {CFG.INIT_CHECKPOINT}")
    else:
        model = from_pretrained(model, repo_id=repo_id)
        print(f"[init] RL starting from pretrained weights: {repo_id}")
    tokenizer = Qwen3Tokenizer(str(MODELS_DIR / "tokenizer.json"))

    # ---- DataModule
    datamodule = LightningDataModule(
        tokenizer=tokenizer,
        train_records=train_records,
        val_records=test_records,
        cfg=CFG
    )

    # ---- Lit-Logger
    date = datetime.now().strftime("%d_%m_%y")
    mode_tag = "LoRA" if CFG.TUNING_MODE == "lora" else f"QLoRA_{CFG.QUANT_BITS}"
    think_suffix = "_think" if CFG.INCLUDE_THINKING else ""

    exp_name = f'Qwen3_Gsm8k_GRPO_{CFG.MODEL_SIZE}_{mode_tag}{think_suffix}'
    csv_logger = pl.loggers.CSVLogger(save_dir=str(Path('./logs') / exp_name), name=date,
                                      flush_logs_every_n_steps=1)
    loggers = [csv_logger]

    RUN_DIR = Path(csv_logger.log_dir)
    VALIDATION_DIR = RUN_DIR / 'validation'

    run_name = f'{exp_name}_{date}_v{csv_logger.version}'
    print(f"[logger] run directory: {RUN_DIR}")
    print(f"[logger] experiment name: {run_name}")
    print(f"[eval] per-round answers -> {VALIDATION_DIR}/round<N>.json")

    # ---- Lightning Train
    qwen3_module = Qwen3_GRPO_Lightning(model=model, tokenizer=tokenizer,
                                        validation_dir=VALIDATION_DIR)

    metadata = {
        'model': f'Qwen3-{CFG.MODEL_SIZE}',
        'dataset': 'gsm8k',
        'method': 'grpo',
        'init_checkpoint': str(CFG.INIT_CHECKPOINT),
        'tuning_mode': CFG.TUNING_MODE,
        'quant_bits': str(CFG.QUANT_BITS),
        'rank': str(CFG.RANK),
        'alpha': str(CFG.ALPHA),
        'optim': CFG.OPTIM,
        'lr': str(CFG.LR),
        'lr_scheduler': CFG.LR_SCHEDULER,
        'warmup_steps': str(CFG.WARMUP_STEPS),
        'weight_decay': str(CFG.WEIGHT_DECAY),
        'group_size': str(CFG.GROUP_SIZE),
        'prompts_per_step': str(CFG.PROMPTS_PER_STEP),
        'grad_accum': str(CFG.GRAD_ACCUM),
        'loss_type': CFG.LOSS_TYPE,
        'scale_adv_by_std': str(CFG.SCALE_ADV_BY_STD),
        'kl_beta': str(CFG.KL_BETA),
        'temperature': str(CFG.TEMPERATURE),
        'max_new_tokens': str(CFG.MAX_NEW_TOKENS),
        'reward_weights': str(CFG.REWARD_WEIGHTS),
        'include_thinking': str(CFG.INCLUDE_THINKING),
    }
    csv_logger.log_hyperparams(metadata)
    lit_logger = pl.loggers.LitLogger(
        name=run_name,
        teamspace=CFG.TEAMSPACE,
        root_dir=str(RUN_DIR),
        save_logs=False,
        metadata=metadata,
    )
    lit_logger._version = str(csv_logger.version)
    loggers.append(lit_logger)
    print(f"[logger]   teamspace: {CFG.TEAMSPACE}")
    print(f"[logger]   experiment: {run_name}")

    # ---- Checkpoint: best test accuracy, LoRA adapters only (on_save_checkpoint)
    CKPT_NAME = '{epoch:02d}-{step}-{Validation/reward_mean:.4f}-{' + ACCURACY_KEY + ':.2f}'
    ckpt = pl.callbacks.ModelCheckpoint(
        monitor=ACCURACY_KEY,
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        filename=CKPT_NAME,
        auto_insert_metric_name=False,
        mode='max',
        save_on_train_epoch_end=False,
        dirpath=str(RUN_DIR / 'checkpoints'),
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        callbacks=[ckpt],
        logger=loggers,
        max_epochs=CFG.EPOCHS,
        precision='bf16-mixed',
        val_check_interval=CFG.VAL_EVERY_N_STEPS,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        num_sanity_val_steps=-1,     # round 0: the whole test set on the SFT policy
    )

    trainer.fit(model=qwen3_module, datamodule=datamodule)
    print(f"[ckpt] best LoRA adapters: {ckpt.best_model_path} "
          f"({ACCURACY_KEY} = {ckpt.best_model_score})")

    # ---- Plot Results ----
    plot_training_curves(csv_logger)
