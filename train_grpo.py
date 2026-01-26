import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import random
import re
import os
from collections import deque

# -----------------------------------------------------------------------------
# GRPO Configuration
# -----------------------------------------------------------------------------
@dataclass
class GRPOConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cuda"
    
    # Training
    learning_rate: float = 1e-7 # Lowered from 1e-6 to prevent collapse
    batch_size: int = 4
    num_generations: int = 8
    max_new_tokens: int = 16
    num_iterations: int = 2000
    
    # PPO/GRPO
    beta: float = 0.04 # Keep beta (or increase to 0.1 if stable) but lower LR first.
    epsilon: float = 0.2
    num_inner_updates: int = 5 # Reduce updates per step to effectively lower LR further
    clip_grad_norm: float = 0.5 # Stricter clipping

cfg = GRPOConfig()

# -----------------------------------------------------------------------------
# Task: Multi-step Arithmetic
# Task: "What is (X op1 Y) op2 Z?" with order of operations
# Reward: +0.2 for <answer>...</answer> format, +0.2 if within ±3, +0.6 if exact
# -----------------------------------------------------------------------------
def get_prompt():
    # Generate a simple multi-step expression: (a op1 b) op2 c
    a = random.randint(0, 10)
    b = random.randint(0, 10)
    op1 = random.choice(['+', '-', '*'])
    c = random.randint(0, 10)
    op2 = random.choice(['+', '-', '*'])
    
    expression = f"({a} {op1} {b}) {op2} {c}"
    answer = eval(expression)  # Safe since controlled input
    text = f"User: What is {expression}? Answer in <answer>...</answer>. \nAssistant:"
    return text, str(answer)

def reward_function(completions, answers):
    rewards = []
    accuracies = []
    
    for completion, correct_ans in zip(completions, answers):
        r = 0.0
        acc = 0.0
        correct = int(correct_ans)

        if match := re.search(r'<answer>(.*?)</answer>', completion):
            answer_text = match.group(1).strip()
            r += 0.2
            
            try:
                pred = int(answer_text)
                if abs(pred - correct) <= 1:
                    r += 0.2
                if pred == correct:
                    r += 0.6
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
        dtype=torch.float32,
        trust_remote_code=True
    ).to(cfg.device)
    policy.train()
    
    # Reference
    ref = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation=attn_impl,
        dtype=torch.float32,
        trust_remote_code=True
    ).to(cfg.device)
    ref.eval() # eval mode turns off dropout but not gradients. gradients off next.
    for p in ref.parameters():
        p.requires_grad = False # make sure backprop does not flow through reference.
        
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # AdamW basically adopts a non-Euclidean geometry in the parameter space, ensuring isotropy of gradient std.
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)
    
    # Recent metrics for averaging, using a deque for efficient popping from left and appending to right
    recent_rewards = deque(maxlen=10)
    recent_losses = deque(maxlen=10)
    recent_accuracies = deque(maxlen=10)
    recent_kl = deque(maxlen=10)
    
    # Plotting Setup
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    rewards_avg = []
    accuracies_avg = []
    kl_avg = []
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel("Step (x10)")
    ax1.set_ylabel("Reward", color="tab:blue")
    ax2.set_ylabel("Accuracy", color="tab:orange")
    
    print("Starting Training...")
    
    torch.cuda.empty_cache()
    
    for step in range(cfg.num_iterations):
        # 1. Generate Batch
        prompts = []
        answers = []
        for _ in range(cfg.batch_size):
            p, a = get_prompt()
            prompts.append(p)
            answers.append(a)
            
        tokenizer.padding_side = 'left'
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(cfg.device)
        prompt_len = inputs.input_ids.shape[1]
        
        # Sampling
        # Generate completions for get_logprobs on these completions later. get_logprobs is batched so more efficient. right now, no grad needed.
        with torch.no_grad():
            input_ids_expanded = inputs.input_ids.repeat_interleave(cfg.num_generations, dim=0)
            attn_mask_expanded = inputs.attention_mask.repeat_interleave(cfg.num_generations, dim=0)
            
            sequences = policy.generate(
                input_ids=input_ids_expanded,
                attention_mask=attn_mask_expanded,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Skips text after EOT.
        completions_text = tokenizer.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True)
        
        answers_expanded = []
        for a in answers:
            answers_expanded.extend([a] * cfg.num_generations)
            
        reward_list, accuracy_list = reward_function(None, completions_text, answers_expanded)
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
            old_logprobs = get_batch_logprobs(policy, back_input_ids, prompt_len)
            
        for _ in range(cfg.num_inner_updates): # multiple GRPO updates with the old model fixed.
            curr_logprobs = get_batch_logprobs(policy, back_input_ids, prompt_len)
            
            ratio = torch.exp(curr_logprobs - old_logprobs)
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1-cfg.epsilon, 1+cfg.epsilon) * advantages.unsqueeze(1)
            ppo_loss = -torch.min(surr1, surr2).mean()
            
            kl = torch.exp(curr_logprobs - ref_logprobs) - (curr_logprobs - ref_logprobs) - 1
            kl_loss = kl.mean()
            
            loss = ppo_loss + cfg.beta * kl_loss
            
            optimizer.zero_grad() # clear gradients
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
        
        if step % 10 == 0 and len(recent_rewards) >= 10:
             avg_reward_10 = sum(recent_rewards) / 10
             avg_loss_10 = sum(recent_losses) / 10
             avg_acc_10 = sum(recent_accuracies) / 10
             avg_kl_10 = sum(recent_kl) / 10
             print(f"Last 10 steps avg: Reward={avg_reward_10:.2f}, Acc={avg_acc_10:.2f}, Loss={avg_loss_10:.4f}, KL={avg_kl_10:.4f}")
             print(f"  Prompt: {prompts[0]}")
             print(f"  Completion: {completions_text[0]}")
             
             # Plotting Update
             rewards_avg.append(avg_reward_10)
             accuracies_avg.append(avg_acc_10)
             kl_avg.append(avg_kl_10)
             
             ax1.clear()
             ax2.clear()
             ax1.set_xlabel("Step (x10)")
             ax1.set_ylabel("Reward", color="tab:blue")
             ax2.yaxis.set_label_position('right')
             ax2.set_ylabel("Accuracy", color="tab:orange")
             
             ax1.plot(rewards_avg, color="tab:blue", label="Reward (10-step avg)")
             ax2.plot(accuracies_avg, color="tab:orange", label="Accuracy (10-step avg)")
             
             ax1.set_title("Reward, Accuracy, and KL Divergence (10-step averages)")
             
             # Combine legends
             lines1, labels1 = ax1.get_legend_handles_labels()
             lines2, labels2 = ax2.get_legend_handles_labels()
             ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(0.05, 0.95))
             
             fig.savefig("grpo_training_curve.png", dpi=100, bbox_inches="tight")
             
             # Create separate plot for KL divergence
             fig_kl, ax_kl = plt.subplots()
             ax_kl.plot(kl_avg, color="tab:red", label="KL Divergence (10-step avg)")
             ax_kl.set_xlabel("Step (x10)")
             ax_kl.set_ylabel("KL Divergence", color="tab:red")
             ax_kl.tick_params(axis='y', labelcolor="tab:red")
             ax_kl.set_title("KL Divergence over Training")
             ax_kl.legend(loc="upper left")
             fig_kl.savefig("grpo_kl_divergence.png", dpi=100, bbox_inches="tight")
             plt.close(fig_kl)

    # Save the trained model
    print("Saving the trained model...")
    policy.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")
    print("Model saved to 'saved_model' directory.")
    
    # Save metrics data
    import json
    metrics_data = {
        "rewards_avg": rewards_avg,
        "accuracies_avg": accuracies_avg,
        "kl_avg": kl_avg
    }
    with open("grpo_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)
    print("Metrics saved to 'grpo_metrics.json'.")

if __name__ == "__main__":
    main()
