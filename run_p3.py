"""P3 training module: GRPO training with periodic checkpoint + eval.

Trains for 1200 steps with NO validation prompts during training.
Every 100 steps (including step 0 and step 1200): checkpoints the model
(overwriting) and runs eval_compare + eval_natural to measure zs, os,
and natural language accuracy on a fixed problem set.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import bitsandbytes as bnb
import random
import json
import os
from collections import deque

from train_grpo_multi_epoch import GRPOConfig, get_prompt, reward_function, get_batch_logprobs
from checkpoint import parse_args, apply_overrides, save_checkpoint, load_checkpoint
from plot_metrics import plot_metrics
import eval_compare
import eval_natural


# -----------------------------------------------------------------------------
# Evaluation helpers (use in-memory policy, avoid model reload)
# -----------------------------------------------------------------------------
def _eval_compare_infer(policy, tokenizer, problems, oneshot, device, batch_size):
    """Run eval_compare-style inference on in-memory policy model."""
    correct = 0
    for i in range(0, len(problems), batch_size):
        batch = problems[i:i+batch_size]
        prompts = [eval_compare.make_prompt(p, tokenizer, oneshot) for p in batch]

        tokenizer.padding_side = 'left'
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = policy.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=eval_compare.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        completions = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        for comp, prob in zip(completions, batch):
            if eval_compare.check_answer(comp, prob):
                correct += 1

    return correct / len(problems)


def _eval_natural_infer(policy, tokenizer, problems, device, batch_size):
    """Run eval_natural-style inference on in-memory policy model."""
    correct = 0
    for i in range(0, len(problems), batch_size):
        batch = problems[i:i+batch_size]
        prompts = [eval_natural.make_prompt(p, tokenizer) for p in batch]

        tokenizer.padding_side = 'left'
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = policy.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=eval_natural.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        completions = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        for comp, prob in zip(completions, batch):
            if eval_natural.check_answer(comp, prob):
                correct += 1

    return correct / len(problems)


def run_evals(policy, tokenizer, device, num_count=3, num_problems=100, seed=142, batch_size=16):
    """Run eval_compare (zs + os) and eval_natural on in-memory policy.

    Uses fixed seed for reproducible problem sets across checkpoints.
    Saves/restores random state to avoid perturbing training randomness.
    """
    rng_state = random.getstate()

    problems = eval_compare.generate_problems(num_problems, seed, num_count)
    nat_problems = eval_natural.generate_problems(num_problems, seed, num_count)

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
    """Convert eval_results (every 100 steps) to val_accuracies_avg format
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

    start_step = 0
    rewards_avg = []
    train_accuracies_avg = []
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
        kl_avg = ckpt["metrics"].get("kl_avg", [])
        eval_results = ckpt["metrics"].get("eval_results", eval_results)
    else:
        model_path = args.model_name or cfg.model_name
        ref_path = args.ref_model or model_path

    print(f"Loading policy from {model_path} on {cfg.device}...")
    print(f"Loading ref from {ref_path}")

    # Load models
    attn_impl = "eager" if cfg.device == "mps" else None

    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(cfg.device)
    policy.train()
    policy.gradient_checkpointing_enable()

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

    policy.generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    optimizer = bnb.optim.AdamW8bit(policy.parameters(), lr=cfg.learning_rate)

    if args.resume and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if args.learning_rate is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.learning_rate
            print(f"  Updated optimizer LR to {cfg.learning_rate}")

    recent_rewards = deque(maxlen=20)
    recent_losses = deque(maxlen=20)
    recent_accuracies = deque(maxlen=20)
    recent_kl = deque(maxlen=20)

    print(f"Starting Training (steps {start_step} -> {cfg.num_iterations})...")
    for step in range(start_step, cfg.num_iterations):

        # --- EVAL every 100 steps (including step 0) ---
        if step % 100 == 0:
            results = run_evals(policy, tokenizer, cfg.device, num_count=cfg.num_count)
            eval_results["steps"].append(step)
            for k in ["zs", "os", "natural"]:
                eval_results[k].append(results[k])
            print(f"EVAL step {step}: zs={results['zs']:.1%}, os={results['os']:.1%}, nat={results['natural']:.1%}")

        # --- TRAINING STEP ---
        prompts = []
        answers = []
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

        # Mask out EOS and everything after it
        completion_ids = sequences[:, prompt_len:]
        is_eos = (completion_ids == tokenizer.eos_token_id)
        mask = (~is_eos.cumsum(dim=1).bool()).float()
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
        with torch.no_grad():
            ref_logprobs = get_batch_logprobs(ref, back_input_ids, prompt_len)
            old_logprobs = get_batch_logprobs(policy, back_input_ids, prompt_len)

        for _ in range(cfg.num_inner_updates):
            curr_logprobs = get_batch_logprobs(policy, back_input_ids, prompt_len)

            ratio = torch.exp(curr_logprobs - old_logprobs)
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1-cfg.epsilon, 1+cfg.epsilon) * advantages.unsqueeze(1)

            ppo_values = -torch.min(surr1, surr2)
            ppo_per_sample = (ppo_values * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            ppo_loss = ppo_per_sample.mean()

            kl = torch.exp(curr_logprobs - ref_logprobs) - (curr_logprobs - ref_logprobs) - 1
            kl_per_sample = (kl * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            kl_loss = kl_per_sample.mean()

            loss = ppo_loss + cfg.beta * kl_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.clip_grad_norm)
            optimizer.step()

        # Collect metrics every step
        avg_reward = sum(reward_list) / len(reward_list) if reward_list else 0.0
        avg_acc = sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0.0
        recent_rewards.append(avg_reward)
        recent_accuracies.append(avg_acc)
        recent_losses.append(loss.item())
        recent_kl.append(kl_loss.item())

        # Log training metrics every 10 steps (NO val prompts)
        if step % 10 == 0:
            l = len(recent_rewards)
            avg_reward_20 = sum(recent_rewards) / l
            avg_loss_20 = sum(recent_losses) / l
            train_acc_20 = sum(recent_accuracies) / l
            avg_kl_20 = sum(recent_kl) / l

            print("=" * 50)
            print(f"Step {step} | Reward={avg_reward_20:.2f}, TrainAcc={train_acc_20:.2f}, Loss={avg_loss_20:.4f}, KL={avg_kl_20:.4f}")
            print(f"  Completion: {completions_text[0]}")

            rewards_avg.append(avg_reward_20)
            train_accuracies_avg.append(train_acc_20)
            kl_avg.append(avg_kl_20)

            metrics = {
                "rewards_avg": rewards_avg,
                "train_accuracies_avg": train_accuracies_avg,
                "val_accuracies_avg": _build_val_accs_for_plot(eval_results),
                "kl_avg": kl_avg,
                "eval_results": eval_results,
            }
            plot_metrics(metrics, smooth=False, output_prefix=cfg.run_name, verbose=False)

    # --- FINAL EVAL + CHECKPOINT after last training step ---
    results = run_evals(policy, tokenizer, cfg.device, num_count=cfg.num_count)
    eval_results["steps"].append(cfg.num_iterations)
    for k in ["zs", "os", "natural"]:
        eval_results[k].append(results[k])
    print(f"EVAL step {cfg.num_iterations}: zs={results['zs']:.1%}, os={results['os']:.1%}, nat={results['natural']:.1%}")

    metrics = {
        "rewards_avg": rewards_avg,
        "train_accuracies_avg": train_accuracies_avg,
        "val_accuracies_avg": _build_val_accs_for_plot(eval_results),
        "kl_avg": kl_avg,
        "eval_results": eval_results,
    }
    save_checkpoint(policy, tokenizer, optimizer, cfg.num_iterations, metrics, cfg)

    with open(f"{cfg.run_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {cfg.run_name}_metrics.json")


if __name__ == "__main__":
    main()
