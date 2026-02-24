import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
from dataclasses import dataclass, fields
import argparse
import random
import re
import os
import json
import ast
from collections import deque
from plot_metrics import plot_metrics


# -----------------------------------------------------------------------------
# GRPO Configuration
# -----------------------------------------------------------------------------
@dataclass
class GRPOConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training
    learning_rate: float = 2e-6 # Raised from 1e-7 — model needs stronger signal to learn reasoning patterns
    batch_size: int = 4
    val_batch_size: int = 16
    num_generations: int = 16
    max_new_tokens: int = 160 # Raised for countdown task chain-of-thought reasoning
    num_iterations: int = 1200
    
    # PPO/GRPO
    beta: float = 0.01 # Lowered from 0.025 — allow more policy divergence for exploration
    epsilon: float = 0.2
    num_inner_updates: int = 1 # Key stability fix: 1 update per batch to prevent KL explosion
    clip_grad_norm: float = 0.5 # Stricter clipping

cfg = GRPOConfig()

# -----------------------------------------------------------------------------
# Task: Countdown
# Given a set of numbers, find an expression using +, -, * that equals a target.
# Reward: +1.0 if the expression evaluates to the target using only available numbers.
# -----------------------------------------------------------------------------
_EXAMPLE_Q = (
    "Using the numbers [3, 5, 2], create an expression that equals 13. "
    "You must use all 3 numbers, each exactly once. Available operations: +, -, * (no division). "
    "Show your reasoning in <think>...</think>, then write only the bare expression in <answer>...</answer>."
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

def get_prompt(split="train", tokenizer=None):
    # Hidden split rule:
    # train -> at least one of the 3 numbers is odd
    # val   -> all 3 numbers are even
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    while True:
        nums = [random.randint(1, 10) for _ in range(3)]

        all_even = all(n % 2 == 0 for n in nums)
        if split == "train" and all_even:
            continue
        if split == "val" and not all_even:
            continue

        # Generate a solvable target using all 3 numbers
        ops = ['+', '-', '*']
        op1 = random.choice(ops)
        op2 = random.choice(ops)
        target = eval(f"({nums[0]} {op1} {nums[1]}) {op2} {nums[2]}")

        # Filter for reasonable positive targets; exclude trivial cases where target is already in nums
        if 1 <= target <= 100 and target not in nums:
            break

    random.shuffle(nums)
    answer_data = json.dumps({"target": target, "nums": nums})
    question = (
        f"Using the numbers {nums}, create an expression that equals {target}. "
        f"You must use all 3 numbers. Available operations: +, -, *. "
        f"Show your reasoning in <think>...</think>, then write only the bare expression in <answer>...</answer>."
    )
    if tokenizer is not None:
        messages = [
            {"role": "system", "content": "You are a mathematical puzzle solver."},
            {"role": "user", "content": _EXAMPLE_Q},
            {"role": "assistant", "content": _EXAMPLE_A},
            {"role": "user", "content": question},
        ]
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

        # Format rewards: +0.1 each for <think> and <answer> tags
        has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
        has_answer = bool(re.search(r'<answer>.*?</answer>', completion, re.DOTALL))
        r += 0.1 * has_think + 0.1 * has_answer

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
def get_batch_logprobs(model, input_ids, prompt_len, chunk_size=4):
    all_logprobs = []
    for i in range(0, input_ids.shape[0], chunk_size):
        chunk_ids = input_ids[i:i+chunk_size]
        logits = model(chunk_ids).logits[:, :-1, :]   # [chunk, T-1, vocab]
        labels = chunk_ids[:, 1:]                       # [chunk, T-1]
        # F.cross_entropy computes -log_softmax(logits)[label] in a fused kernel,
        # avoiding the full [chunk, T, 152K] log_softmax allocation.
        # Negating gives us the log prob of each generated token.
        chosen_log_probs = -F.cross_entropy(
            logits.transpose(1, 2),                     # [chunk, vocab, T-1]
            labels,                                     # [chunk, T-1]
            reduction='none'                            # [chunk, T-1]
        )
        all_logprobs.append(chosen_log_probs[:, prompt_len-1:])
        del logits, labels, chosen_log_probs
    return torch.cat(all_logprobs, dim=0)

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    print(f"Loading {cfg.model_name} on {cfg.device}...")
    
    # Check if saved model exists
    saved_model_path = "saved_model"
    if os.path.exists(saved_model_path):
        print(f"Found saved model at '{saved_model_path}', loading from there...")
        model_path = saved_model_path
    else:
        model_path = cfg.model_name
    
    # 1. Load Models
    attn_impl = "eager" if cfg.device == "mps" else None # MPS compatibility for SPDA is low

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
        model_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        trust_remote_code=True # needed for Qwen because it has some custom code
    ).to(cfg.device)
    ref.eval() # eval mode turns off dropout but not gradients. gradients off next.
    for p in ref.parameters():
        p.requires_grad = False # make sure backprop does not flow through reference.
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Reset generation config to avoid model defaults (Qwen sets top_k=20, top_p=0.8, etc.)
    from transformers import GenerationConfig
    policy.generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

        
    # AdamW basically adopts a non-Euclidean geometry in the parameter space, ensuring isotropy of gradient std.
    # 8-bit AdamW quantizes optimizer m/v states to 8-bit, saving ~6 GB for a 1.5B model.
    optimizer = bnb.optim.AdamW8bit(policy.parameters(), lr=cfg.learning_rate)
    
    # Recent metrics for averaging, using a deque for efficient popping from left and appending to right
    recent_rewards = deque(maxlen=20)
    recent_losses = deque(maxlen=20)
    recent_accuracies = deque(maxlen=20)
    recent_kl = deque(maxlen=20)
    
    rewards_avg = []
    train_accuracies_avg = []
    val_accuracies_avg = []
    kl_avg = []
    
    print("Starting Training...")
    for step in range(cfg.num_iterations):
        # # Update reference model periodically to allow continued improvement
        # if step % 200 == 0 and step > 0:
        #     print(f"Step {step}: Updating reference model to current policy.")
        #     ref.load_state_dict(policy.state_dict())

        ref.to(cfg.device)  # Move ref back to GPU for this step

        # 1. Generate Batch
        prompts = []
        answers = []
        for _ in range(cfg.batch_size):
            p, a = get_prompt(split="train", tokenizer=tokenizer)
            prompts.append(p)
            answers.append(a)
            
        tokenizer.padding_side = 'left'
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(cfg.device)
        prompt_len = inputs.input_ids.shape[1]
        
        # Sampling
        # Generate completions for get_logprobs on these completions later. get_logprobs is batched so more efficient. right now, no grad needed.
        policy.gradient_checkpointing_disable()
        policy.config.use_cache = True
        policy.eval()
        with torch.no_grad():
            # batch_size prompts, each prompt has num_generations repetitions
            input_ids_expanded = inputs.input_ids.repeat_interleave(cfg.num_generations, dim=0)
            # attention mask masks the padding tokens
            attn_mask_expanded = inputs.attention_mask.repeat_interleave(cfg.num_generations, dim=0)

            sequences = policy.generate(
                input_ids=input_ids_expanded,
                attention_mask=attn_mask_expanded,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=1.2,
                top_k=0,           # disable top-k filtering
                top_p=1.0,         # disable nucleus sampling
                repetition_penalty=1.0,  # disable repetition penalty
                pad_token_id=tokenizer.pad_token_id
            )
        policy.config.use_cache = False  # must be False before gradient_checkpointing_enable() to suppress warning
        policy.train()
        policy.gradient_checkpointing_enable()

        # Note: The returned sequences are padded to the longest sequence in the batch.
        # If some sequences stop early (e.g., generate EOS), they are right-padded with pad_token_id
        # to match the longest sequence. This means the sequences tensor contains padding tokens
        # that will be included in loss calculations unless explicitly masked out.

        # Mask out EOS and everything after it
        completion_ids = sequences[:, prompt_len:]
        is_eos = (completion_ids == tokenizer.eos_token_id)
        mask = (~is_eos.cumsum(dim=1).bool()).float()  # 1 before first EOS, 0 after
        # Skips padding IDs.
        completions_text = tokenizer.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True)
        
        answers_expanded = []
        for a in answers:
            answers_expanded.extend([a] * cfg.num_generations)
            
        reward_list, accuracy_list = reward_function(completions_text, answers_expanded)
        rewards = torch.tensor(reward_list, device=cfg.device)
        
        rewards_view = rewards.view(cfg.batch_size, cfg.num_generations)
        mean_r = rewards_view.mean(dim=1, keepdim=True) # taking the mean over generations for each prompt
        std_r = rewards_view.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards_view - mean_r) / std_r  # Shape: [batch_size, num_generations] after broadcasting
        advantages = advantages.view(-1)  # Flatten to [batch_size * num_generations]
        
        # Optimization
        back_input_ids = sequences
        with torch.no_grad(): # No grad for ref and old policy
            ref_logprobs = get_batch_logprobs(ref, back_input_ids, prompt_len)
            ref.to("cpu")  # Free ~3GB VRAM — ref not needed until next step
            torch.cuda.empty_cache()
            # the only difference between old and new policy is torch.no_grad() when cfg.num_inner_updates = 1.
            old_logprobs = get_batch_logprobs(policy, back_input_ids, prompt_len)

        for _ in range(cfg.num_inner_updates): # multiple GRPO updates with the old model fixed.
            curr_logprobs = get_batch_logprobs(policy, back_input_ids, prompt_len)
            
            ratio = torch.exp(curr_logprobs - old_logprobs)
            surr1 = ratio * advantages.unsqueeze(1) # broadcasting: [batch_size * num_generations, T] * [batch_size * num_generations, 1]
            surr2 = torch.clamp(ratio, 1-cfg.epsilon, 1+cfg.epsilon) * advantages.unsqueeze(1)

            # Masked loss: exclude padding tokens
            ppo_values = -torch.min(surr1, surr2)  # [batch_size * num_generations, T]
            ppo_per_sample = (ppo_values * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # Average per sequence
            ppo_loss = ppo_per_sample.mean()  # Average across batch

            kl = torch.exp(curr_logprobs - ref_logprobs) - (curr_logprobs - ref_logprobs) - 1
            kl_per_sample = (kl * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # Average per sequence
            kl_loss = kl_per_sample.mean()  # Average across batch
            
            loss = ppo_loss + cfg.beta * kl_loss
            
            optimizer.zero_grad(set_to_none=True) # clear gradients
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
                p, a = get_prompt(split="val", tokenizer=tokenizer)
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
            print(f"Last 20 steps avg: Reward={avg_reward_20:.2f}, TrainAcc={train_acc_20:.2f}, ValAcc={val_acc:.2f}, Loss={avg_loss_20:.4f}, KL={avg_kl_20:.4f}")
            # print(f"  Prompt: {prompts[0]}")
            print(f"  Completion: {completions_text[0]}")
            
            # Plotting Update
            rewards_avg.append(avg_reward_20)
            train_accuracies_avg.append(train_acc_20)
            val_accuracies_avg.append(val_acc)
            kl_avg.append(avg_kl_20)
            
            # Plot metrics every 10 steps
            plot_metrics({
                "rewards_avg": rewards_avg,
                "train_accuracies_avg": train_accuracies_avg,
                "val_accuracies_avg": val_accuracies_avg,
                "kl_avg": kl_avg
            }, smooth=False, verbose=False)

    # Save the trained model
    print("Saving the trained model...")
    policy.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")
    print("Model saved to 'saved_model' directory.")
    
    # Save metrics data
    metrics_data = {
        "rewards_avg": rewards_avg,
        "train_accuracies_avg": train_accuracies_avg,
        "val_accuracies_avg": val_accuracies_avg,
        "kl_avg": kl_avg
    }
    with open("grpo_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)
    print("Metrics saved to 'grpo_metrics.json'.")

if __name__ == "__main__":
    main()
