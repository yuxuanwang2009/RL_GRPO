"""Unified GRPO training script.

Replaces the three original training scripts:
  - train_grpo.py (basic single-run training)
  - train_grpo_multi_epoch.py (checkpointing + resume)
  - run_p3.py (periodic cross-format evaluation)

Usage examples:
  # Basic training (like old train_grpo.py):
  python scripts/train.py

  # With checkpointing and resume (like old train_grpo_multi_epoch.py):
  python scripts/train.py --run_name myrun --resume

  # With periodic cross-format eval (like old run_p3.py):
  python scripts/train.py --run_name p3 --eval_every 100

  # One-shot training:
  python scripts/train.py --oneshot --run_name oneshot_run

  # Mixed one-shot/zero-shot curriculum:
  python scripts/train.py --mix_oneshot 0.5 --run_name halfshot
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import bitsandbytes as bnb
import random
import json
import os
from collections import deque

from grpo.config import GRPOConfig, OUTPUTS_DIR
from grpo.prompts import get_prompt, make_prompt, make_natural_prompt
from grpo.validation import reward_function, check_answer, check_answer_natural
from grpo.datasets import generate_problems
from grpo.training import get_batch_logprobs, evaluate_accuracy
from grpo.checkpoint import parse_args, apply_overrides, save_checkpoint, load_checkpoint
from grpo.plotting import plot_metrics


MAX_NEW_TOKENS_EVAL = 160


# -----------------------------------------------------------------------------
# Cross-format evaluation helpers (for --eval_every mode)
# -----------------------------------------------------------------------------
def _eval_compare_infer(policy, tokenizer, problems, oneshot, device, batch_size):
    """Run eval_compare-style inference on in-memory policy model."""
    correct = 0
    for i in range(0, len(problems), batch_size):
        batch = problems[i:i+batch_size]
        prompts = [make_prompt(p, tokenizer, oneshot) for p in batch]

        tokenizer.padding_side = 'left'
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = policy.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS_EVAL,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        completions = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        for comp, prob in zip(completions, batch):
            if check_answer(comp, prob):
                correct += 1

    return correct / len(problems)


def _eval_natural_infer(policy, tokenizer, problems, device, batch_size):
    """Run eval_natural-style inference on in-memory policy model."""
    correct = 0
    for i in range(0, len(problems), batch_size):
        batch = problems[i:i+batch_size]
        prompts = [make_natural_prompt(p, tokenizer) for p in batch]

        tokenizer.padding_side = 'left'
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = policy.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS_EVAL,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        completions = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        for comp, prob in zip(completions, batch):
            if check_answer_natural(comp, prob):
                correct += 1

    return correct / len(problems)


def run_evals(policy, tokenizer, device, num_count=3, num_problems=100, seed=142, batch_size=16):
    """Run cross-format evaluation on in-memory policy.

    Uses fixed seed for reproducible problem sets across checkpoints.
    Saves/restores random state to avoid perturbing training randomness.
    """
    rng_state = random.getstate()

    problems = generate_problems(num_problems, seed, num_count)
    nat_problems = generate_problems(num_problems, seed, num_count)

    policy.gradient_checkpointing_disable()
    policy.config.use_cache = True
    policy.eval()

    results = {}
    results["zs"] = _eval_compare_infer(policy, tokenizer, problems, oneshot=False, device=device, batch_size=batch_size)
    results["os"] = _eval_compare_infer(policy, tokenizer, problems, oneshot=True, device=device, batch_size=batch_size)
    results["natural"] = _eval_natural_infer(policy, tokenizer, nat_problems, device=device, batch_size=batch_size)

    policy.config.use_cache = False
    policy.train()
    policy.gradient_checkpointing_enable()

    random.setstate(rng_state)
    return results


def _build_val_accs_for_plot(eval_results):
    """Convert eval_results (every N steps) to val_accuracies_avg format
    for plot_metrics (one entry per 10-step logging interval)."""
    val_accs = {"3num_zs": [], "3num_os": [], "natural": []}
    steps = eval_results.get("steps", [])

    for i in range(len(steps)):
        zs_val = eval_results["zs"][i] if i < len(eval_results["zs"]) else 0.0
        os_val = eval_results["os"][i] if i < len(eval_results["os"]) else 0.0
        nat_val = eval_results["natural"][i] if i < len(eval_results["natural"]) else 0.0

        if i + 1 < len(steps):
            num_intervals = (steps[i + 1] - steps[i]) // 10
        else:
            num_intervals = 1  # last eval point gets a single entry

        val_accs["3num_zs"].extend([zs_val] * num_intervals)
        val_accs["3num_os"].extend([os_val] * num_intervals)
        val_accs["natural"].extend([nat_val] * num_intervals)

    return val_accs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = GRPOConfig()
    apply_overrides(cfg, args)

    eval_every = args.eval_every  # None = disabled, int = run evals every N steps

    start_step = 0
    rewards_avg = []
    train_accuracies_avg = []
    val_accuracies_avg = []
    kl_avg = []
    eval_results = {"steps": [], "zs": [], "os": [], "natural": []}

    # Determine model paths
    if args.resume:
        ckpt = load_checkpoint(cfg.run_name)
        model_path = ckpt["policy_path"]
        ref_path = args.ref_model or ckpt["policy_path"]
        start_step = ckpt["step"]
        rewards_avg = ckpt["metrics"].get("rewards_avg", [])
        train_accuracies_avg = ckpt["metrics"].get("train_accuracies_avg", [])
        val_accuracies_avg = ckpt["metrics"].get("val_accuracies_avg", [])
        kl_avg = ckpt["metrics"].get("kl_avg", [])
        eval_results = ckpt["metrics"].get("eval_results", eval_results)
    else:
        model_path = args.model_name or cfg.model_name
        ref_path = args.ref_model or model_path

    print(f"Loading policy from {model_path} on {cfg.device}...")
    print(f"Loading ref from {ref_path}")

    # 1. Load Models
    attn_impl = "eager" if cfg.device == "mps" else None

    # Policy
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(cfg.device)
    policy.train()
    policy.gradient_checkpointing_enable()

    # Reference, used for KL divergence
    ref = AutoModelForCausalLM.from_pretrained(
        ref_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(cfg.device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Reset generation config to avoid model defaults (Qwen sets top_k=20, top_p=0.8, etc.)
    policy.generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 8-bit AdamW quantizes optimizer m/v states to 8-bit, saving ~6 GB for a 1.5B model.
    optimizer = bnb.optim.AdamW8bit(policy.parameters(), lr=cfg.learning_rate)

    # Restore optimizer state if resuming
    if args.resume and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if args.learning_rate is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.learning_rate
            print(f"  Updated optimizer LR to {cfg.learning_rate}")

    # Recent metrics for averaging
    recent_rewards = deque(maxlen=20)
    recent_losses = deque(maxlen=20)
    recent_accuracies = deque(maxlen=20)
    recent_kl = deque(maxlen=20)

    print(f"Starting Training (steps {start_step} -> {cfg.num_iterations})...")
    for step in range(start_step, cfg.num_iterations):

        # --- Cross-format eval (if enabled) ---
        if eval_every and step % eval_every == 0:
            results = run_evals(policy, tokenizer, cfg.device, num_count=cfg.num_count)
            eval_results["steps"].append(step)
            for k in ["zs", "os", "natural"]:
                eval_results[k].append(results[k])
            print(f"EVAL step {step}: zs={results['zs']:.1%}, os={results['os']:.1%}, nat={results['natural']:.1%}")

        # --- Free stale gradients before the memory-heavy generation phase ---
        optimizer.zero_grad(set_to_none=True)

        # --- Move ref to GPU for this step ---
        ref.to(cfg.device)

        # 1. Generate Batch
        prompts = []
        answers = []
        # Decide one-shot vs zero-shot for entire batch (avoids padding waste)
        if cfg.mix_oneshot > 0:
            oneshot_prob = 1.0 - step / (cfg.num_iterations - 1)
            batch_oneshot = random.random() < oneshot_prob
        else:
            batch_oneshot = cfg.oneshot
        for _ in range(cfg.batch_size):
            p, a = get_prompt(split="train", tokenizer=tokenizer, num_count=cfg.num_count, oneshot=batch_oneshot)
            prompts.append(p)
            answers.append(a)

        tokenizer.padding_side = 'left'
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(cfg.device)
        prompt_len = inputs.input_ids.shape[1]

        # Sampling
        policy.gradient_checkpointing_disable()
        policy.config.use_cache = True
        policy.eval()
        with torch.no_grad():
            input_ids_expanded = inputs.input_ids.repeat_interleave(cfg.num_generations, dim=0)
            attn_mask_expanded = inputs.attention_mask.repeat_interleave(cfg.num_generations, dim=0)

            sequences = policy.generate(
                input_ids=input_ids_expanded,
                attention_mask=attn_mask_expanded,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id
            )
        policy.config.use_cache = False
        policy.train()
        policy.gradient_checkpointing_enable()

        # Mask out everything after first EOS (include the EOS itself)
        completion_ids = sequences[:, prompt_len:]
        is_eos = (completion_ids == tokenizer.eos_token_id)
        eos_cumsum = is_eos.cumsum(dim=1)
        shifted = torch.cat([torch.zeros_like(eos_cumsum[:, :1]), eos_cumsum[:, :-1]], dim=1)
        mask = (~shifted.bool()).float()  # 1 up to and including first EOS, 0 after
        completions_text = tokenizer.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True)

        answers_expanded = []
        for a in answers:
            answers_expanded.extend([a] * cfg.num_generations)

        reward_list, accuracy_list = reward_function(completions_text, answers_expanded)
        rewards = torch.tensor(reward_list, device=cfg.device)

        rewards_view = rewards.view(cfg.batch_size, cfg.num_generations)
        mean_r = rewards_view.mean(dim=1, keepdim=True)
        std_r = rewards_view.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards_view - mean_r) / std_r
        advantages = advantages.view(-1)

        # Optimization
        back_input_ids = sequences
        back_attn_mask = (sequences != tokenizer.pad_token_id).long()
        total_samples = back_input_ids.shape[0]
        micro_batch = total_samples // cfg.grad_accum_steps

        with torch.no_grad():
            ref_logprobs = torch.cat([
                get_batch_logprobs(ref, back_input_ids[s:s+micro_batch], back_attn_mask[s:s+micro_batch], prompt_len)
                for s in range(0, total_samples, micro_batch)
            ], dim=0)
            ref.to("cpu")  # Free ~3GB VRAM — ref not needed until next step
            torch.cuda.empty_cache()

        old_logprobs = None

        for inner_idx in range(cfg.num_inner_updates):
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            accum_kl = 0.0
            old_logprobs_chunks = []

            for ga_idx in range(cfg.grad_accum_steps):
                s = ga_idx * micro_batch
                e = s + micro_batch

                mb_curr_lp = get_batch_logprobs(policy, back_input_ids[s:e], back_attn_mask[s:e], prompt_len)

                # First inner update: old policy == current policy,
                # reuse detached logprobs instead of a separate forward pass.
                if inner_idx == 0:
                    mb_old_lp = mb_curr_lp.detach()
                    old_logprobs_chunks.append(mb_old_lp)
                else:
                    mb_old_lp = old_logprobs[s:e]

                ratio = torch.exp(mb_curr_lp - mb_old_lp)
                surr1 = ratio * advantages[s:e].unsqueeze(1)
                surr2 = torch.clamp(ratio, 1-cfg.epsilon, 1+cfg.epsilon) * advantages[s:e].unsqueeze(1)

                ppo_values = -torch.min(surr1, surr2)
                ppo_per_sample = (ppo_values * mask[s:e]).sum(dim=1) / (mask[s:e].sum(dim=1) + 1e-8)
                ppo_loss = ppo_per_sample.mean()

                kl = torch.exp(mb_curr_lp - ref_logprobs[s:e]) - (mb_curr_lp - ref_logprobs[s:e]) - 1
                kl_per_sample = (kl * mask[s:e]).sum(dim=1) / (mask[s:e].sum(dim=1) + 1e-8)
                kl_loss = kl_per_sample.mean()

                mb_loss = (ppo_loss + cfg.beta * kl_loss) / cfg.grad_accum_steps
                mb_loss.backward()

                accum_loss += ppo_loss.item() / cfg.grad_accum_steps
                accum_kl += kl_loss.item() / cfg.grad_accum_steps

            if inner_idx == 0:
                old_logprobs = torch.cat(old_logprobs_chunks, dim=0)

            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.clip_grad_norm)
            optimizer.step()

        # Collect metrics every step
        avg_reward = sum(reward_list) / len(reward_list) if reward_list else 0.0
        avg_acc = sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0.0
        recent_rewards.append(avg_reward)
        recent_accuracies.append(avg_acc)
        recent_losses.append(accum_loss)
        recent_kl.append(accum_kl)

        # Log training metrics every 10 steps
        if step % 10 == 0:
            l = len(recent_rewards)
            avg_reward_20 = sum(recent_rewards) / l
            avg_loss_20 = sum(recent_losses) / l
            train_acc_20 = sum(recent_accuracies) / l
            avg_kl_20 = sum(recent_kl) / l

            # Run inline val if cross-format eval is NOT enabled
            val_acc = None
            if not eval_every:
                val_prompts = []
                val_answers = []
                for _ in range(cfg.val_batch_size):
                    p, a = get_prompt(split="val", tokenizer=tokenizer, num_count=cfg.num_count, oneshot=batch_oneshot)
                    val_prompts.append(p)
                    val_answers.append(a)
                val_acc = evaluate_accuracy(
                    policy, tokenizer, val_prompts, val_answers, cfg.device, cfg.max_new_tokens
                )

            print("=" * 50)
            if val_acc is not None:
                print(f"Step {step} | Reward={avg_reward_20:.2f}, TrainAcc={train_acc_20:.2f}, ValAcc={val_acc:.2f}, Loss={avg_loss_20:.4f}, KL={avg_kl_20:.4f}")
            else:
                print(f"Step {step} | Reward={avg_reward_20:.2f}, TrainAcc={train_acc_20:.2f}, Loss={avg_loss_20:.4f}, KL={avg_kl_20:.4f}")
            print(f"  Completion: {completions_text[0]}")

            rewards_avg.append(avg_reward_20)
            train_accuracies_avg.append(train_acc_20)
            kl_avg.append(avg_kl_20)

            if eval_every:
                # Use cross-format eval results for plotting
                val_accs_plot = _build_val_accs_for_plot(eval_results)
            else:
                if val_acc is not None:
                    val_accuracies_avg.append(val_acc)
                val_accs_plot = val_accuracies_avg

            metrics = {
                "rewards_avg": rewards_avg,
                "train_accuracies_avg": train_accuracies_avg,
                "val_accuracies_avg": val_accs_plot,
                "kl_avg": kl_avg,
            }
            if eval_every:
                metrics["eval_results"] = eval_results

            plot_metrics(metrics, smooth=False, output_prefix=cfg.run_name, verbose=False)

    # --- Final eval + checkpoint ---
    if eval_every:
        results = run_evals(policy, tokenizer, cfg.device, num_count=cfg.num_count)
        eval_results["steps"].append(cfg.num_iterations)
        for k in ["zs", "os", "natural"]:
            eval_results[k].append(results[k])
        print(f"EVAL step {cfg.num_iterations}: zs={results['zs']:.1%}, os={results['os']:.1%}, nat={results['natural']:.1%}")

    metrics = {
        "rewards_avg": rewards_avg,
        "train_accuracies_avg": train_accuracies_avg,
        "val_accuracies_avg": _build_val_accs_for_plot(eval_results) if eval_every else val_accuracies_avg,
        "kl_avg": kl_avg,
    }
    if eval_every:
        metrics["eval_results"] = eval_results

    save_checkpoint(policy, tokenizer, optimizer, cfg.num_iterations, metrics, cfg)

    metrics_path = os.path.join(OUTPUTS_DIR, f"{cfg.run_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
