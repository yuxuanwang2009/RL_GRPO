import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import random
import re
import os
import json
from collections import deque
from plot_metrics import plot_metrics


# -----------------------------------------------------------------------------
# GRPO Configuration
# -----------------------------------------------------------------------------
@dataclass
class GRPOConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training
    learning_rate: float = 2e-6 # Raised from 1e-7 — model needs stronger signal to learn reasoning patterns
    batch_size: int = 4
    val_batch_size: int = 32
    num_generations: int = 16
    max_new_tokens: int = 64 # Raised from 16 — critical for chain-of-thought reasoning
    num_iterations: int = 1200
    
    # PPO/GRPO
    beta: float = 0.01 # Lowered from 0.025 — allow more policy divergence for exploration
    epsilon: float = 0.2
    num_inner_updates: int = 1 # Key stability fix: 1 update per batch to prevent KL explosion
    clip_grad_norm: float = 0.5 # Stricter clipping

cfg = GRPOConfig()

# -----------------------------------------------------------------------------
# Task: Multi-step Arithmetic
# Task: "What is (X op1 Y) op2 Z?" with order of operations
# Reward: +0.0 for <answer>...</answer> format, +0.0 if within ±1, +1.0 if exact
# -----------------------------------------------------------------------------
def get_prompt(split="train"):
    # Hidden split rule:
    # train -> at least one of a, b, c is odd
    # val   -> all of a, b, c are even
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    while True:
        a = random.randint(50, 70)
        b = random.randint(50, 70)
        c = random.randint(50, 70)
        if split == "train" and ((a % 2 == 1) or (b % 2 == 1) or (c % 2 == 1)):
            break
        if split == "val" and ((a % 2 == 0) and (b % 2 == 0) and (c % 2 == 0)):
            break

    op1 = random.choice(['+', '-', '*'])
    op2 = random.choice(['+', '-', '*'])
    
    expression = f"({a} {op1} {b}) {op2} {c}"
    answer = eval(expression)  # Safe since controlled input
    text = f"User: What is {expression}? Think step by step, then put your final answer in <answer>...</answer>.\nAssistant:"
    return text, str(answer)


def evaluate_accuracy(model, tokenizer, prompts, answers, device, max_new_tokens):
    tokenizer.padding_side = 'left'
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        sequences = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    completions_text = tokenizer.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True)
    _, accuracy_list = reward_function(completions_text, answers)
    return sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0.0

def reward_function(completions, answers):
    rewards = []
    accuracies = []
    
    for completion, correct_ans in zip(completions, answers):
        r = 0.0
        acc = 0.0
        correct = int(correct_ans)

        if match := re.search(r'<answer>(.*?)</answer>', completion):
            answer_text = match.group(1).strip()
            r += 0.0
            
            try:
                pred = int(answer_text)
                if abs(pred - correct) <= 1:
                    r += 0.0
                if pred == correct:
                    r += 1.0
                    acc = 1.0
            except ValueError:
                pass  # Not a number, no extra reward
        
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
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(cfg.device)
    policy.train()

    # Reference, used for KL divergence
    ref = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
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
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)
    
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
        # Update reference model periodically to allow continued improvement
        if step % 200 == 0 and step > 0:
            print(f"Step {step}: Updating reference model to current policy.")
            ref.load_state_dict(policy.state_dict())

        # 1. Generate Batch
        prompts = []
        answers = []
        for _ in range(cfg.batch_size):
            p, a = get_prompt(split="train")
            prompts.append(p)
            answers.append(a)
            
        tokenizer.padding_side = 'left'
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(cfg.device)
        prompt_len = inputs.input_ids.shape[1]
        
        # Sampling
        # Generate completions for get_logprobs on these completions later. get_logprobs is batched so more efficient. right now, no grad needed.
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
                p, a = get_prompt(split="val")
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

            print(f"Last 20 steps avg: Reward={avg_reward_20:.2f}, TrainAcc={train_acc_20:.2f}, ValAcc={val_acc:.2f}, Loss={avg_loss_20:.4f}, KL={avg_kl_20:.4f}")
            print(f"  Prompt: {prompts[0]}")
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
