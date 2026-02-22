import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import bitsandbytes as bnb
from dataclasses import dataclass
import random
import re
import os
import json
import ast
from collections import deque
from plot_metrics import plot_metrics
from checkpoint import parse_args, apply_overrides, save_checkpoint, load_checkpoint


# -----------------------------------------------------------------------------
# GRPO Configuration
# -----------------------------------------------------------------------------
@dataclass
class GRPOConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # Training
    learning_rate: float = 2e-6
    batch_size: int = 4
    val_batch_size: int = 64
    num_generations: int = 16
    max_new_tokens: int = 160
    num_iterations: int = 1200
    temperature: float = 1.2

    # PPO/GRPO
    beta: float = 0.01
    epsilon: float = 0.2
    num_inner_updates: int = 1
    clip_grad_norm: float = 0.5

    # Task
    num_count: int = 3
    oneshot: bool = False
    mix_oneshot: float = 0.0  # fraction of prompts that get one-shot (0.0 = all zero-shot, 0.5 = 50/50)

    # Run management
    run_name: str = "grpo"


_EXAMPLE_Q = (
    "Using the numbers [3, 5, 2], create an expression that equals 13. "
    "You must use all 3 numbers, each exactly once. Available operations: +, -, * (no division). "
    "Show your reasoning in <think>...</think>, then write your final expression in <answer>...</answer>."
)
_EXAMPLE_A = (
    "<think>\n"
    "I need to reach 13 using [3, 5, 2].\n"
    "Try 3 + 5 + 2 = 10. Too small, need multiplication.\n"
    "Try 3 * 5 + 2 = 17. Too large.\n"
    "Try 5 * 2 + 3 = 13. Yes!\n"
    "</think>\n"
    "<answer>5 * 2 + 3</answer>"
)


# -----------------------------------------------------------------------------
# Task: Countdown
# Given a set of numbers, find an expression using +, -, * that equals a target.
# Reward: +1.0 if the expression evaluates to the target using only available numbers.
# -----------------------------------------------------------------------------
def get_prompt(split="train", tokenizer=None, num_count=3, oneshot=False):
    # Hidden split rule:
    # train -> at least one number is odd
    # val   -> all numbers are even
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    while True:
        nums = [random.randint(1, 10) for _ in range(num_count)]

        all_even = all(n % 2 == 0 for n in nums)
        if split == "train" and all_even:
            continue
        if split == "val" and not all_even:
            continue

        # Generate a solvable target using all numbers (left-associative)
        ops = ['+', '-', '*']
        chosen_ops = [random.choice(ops) for _ in range(num_count - 1)]
        expr = str(nums[0])
        for i, op in enumerate(chosen_ops):
            expr = f"({expr} {op} {nums[i+1]})"
        target = eval(expr)

        # Filter for reasonable positive targets; exclude trivial cases where target is already in nums
        if 1 <= target <= 100 and target not in nums:
            break

    random.shuffle(nums)
    answer_data = json.dumps({"target": target, "nums": nums})
    question = (
        f"Using the numbers {nums}, create an expression that equals {target}. "
        f"You must use all {num_count} numbers, each exactly once. Available operations: +, -, * (no division). "
        f"Show your reasoning in <think>...</think>, then write your final expression in <answer>...</answer>."
    )
    if tokenizer is not None:
        messages = [
            {"role": "system", "content": "You are a mathematical puzzle solver."},
        ]
        if oneshot:
            messages.append({"role": "user", "content": _EXAMPLE_Q})
            messages.append({"role": "assistant", "content": _EXAMPLE_A})
        messages.append({"role": "user", "content": question})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"User: {question}\nAssistant:"
    return text, answer_data


def evaluate_accuracy(model, tokenizer, prompts, answers, device, max_new_tokens):
    tokenizer.padding_side = 'left'
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    prompt_len = inputs.input_ids.shape[1]

    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    with torch.no_grad():
        sequences = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    model.config.use_cache = False
    model.train()
    model.gradient_checkpointing_enable()

    completions_text = tokenizer.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True)
    _, accuracy_list = reward_function(completions_text, answers)
    return sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0.0

def safe_eval(expr_str):
    """Safely evaluate a mathematical expression containing only +, -, *."""
    try:
        tree = ast.parse(expr_str.strip(), mode='eval')
    except SyntaxError:
        return None

    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.USub)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            return None

    try:
        return eval(compile(tree, '<string>', 'eval'))
    except Exception:
        return None

def extract_numbers(expr_str):
    """Extract all integer literals from an expression string."""
    return [int(x) for x in re.findall(r'\d+', expr_str)]

def reward_function(completions, answers):
    rewards = []
    accuracies = []

    for completion, answer_json in zip(completions, answers):
        r = 0.0
        acc = 0.0
        info = json.loads(answer_json)
        target = info["target"]
        available = sorted(info["nums"])

        # Format reward: +0.1 for <think> tag, +0.1 for <answer> tag
        has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
        has_answer = bool(re.search(r'<answer>.*?</answer>', completion, re.DOTALL))
        r += 0.1 * has_think
        r += 0.1 * has_answer

        if match := re.search(r'<answer>(.*?)</answer>', completion):
            expr = match.group(1).strip()
            result = safe_eval(expr)

            if result is not None and result == target:
                # Verify all available numbers are used, each exactly once
                used = sorted(extract_numbers(expr))
                pool = available.copy()
                valid = True
                for n in used:
                    if n in pool:
                        pool.remove(n)
                    else:
                        valid = False
                        break

                if valid and len(pool) == 0:  # all numbers consumed
                    r = 1.0  # Full reward overwrites format partial credit
                    acc = 1.0

        rewards.append(r)
        accuracies.append(acc)

    return rewards, accuracies


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_batch_logprobs(model, input_ids, prompt_len):
    outputs = model(input_ids)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    chosen_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return chosen_log_probs[:, prompt_len-1:]

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = GRPOConfig()
    apply_overrides(cfg, args)

    start_step = 0
    rewards_avg = []
    train_accuracies_avg = []
    val_accuracies_avg = []
    kl_avg = []

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
        # Apply new learning rate if overridden
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

        if step % 10 == 0:
            l = len(recent_rewards)
            avg_reward_20 = sum(recent_rewards) / l
            avg_loss_20 = sum(recent_losses) / l
            train_acc_20 = sum(recent_accuracies) / l
            avg_kl_20 = sum(recent_kl) / l

            val_prompts = []
            val_answers = []
            for _ in range(cfg.val_batch_size):
                p, a = get_prompt(split="val", tokenizer=tokenizer, num_count=cfg.num_count, oneshot=batch_oneshot)
                val_prompts.append(p)
                val_answers.append(a)
            val_acc = evaluate_accuracy(
                policy,
                tokenizer,
                val_prompts,
                val_answers,
                cfg.device,
                cfg.max_new_tokens
            )

            print("="*50)
            print(f"Step {step} | Reward={avg_reward_20:.2f}, TrainAcc={train_acc_20:.2f}, ValAcc={val_acc:.2f}, Loss={avg_loss_20:.4f}, KL={avg_kl_20:.4f}")
            print(f"  Completion: {completions_text[0]}")

            rewards_avg.append(avg_reward_20)
            train_accuracies_avg.append(train_acc_20)
            val_accuracies_avg.append(val_acc)
            kl_avg.append(avg_kl_20)

            metrics = {
                "rewards_avg": rewards_avg,
                "train_accuracies_avg": train_accuracies_avg,
                "val_accuracies_avg": val_accuracies_avg,
                "kl_avg": kl_avg
            }
            plot_metrics(metrics, smooth=False, output_prefix=cfg.run_name, verbose=False)

    # Save checkpoint at end
    metrics = {
        "rewards_avg": rewards_avg,
        "train_accuracies_avg": train_accuracies_avg,
        "val_accuracies_avg": val_accuracies_avg,
        "kl_avg": kl_avg
    }
    save_checkpoint(policy, tokenizer, optimizer, cfg.num_iterations, metrics, cfg)

    # Also save metrics JSON for plot_metrics.py compatibility
    with open(f"{cfg.run_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {cfg.run_name}_metrics.json")

if __name__ == "__main__":
    main()
