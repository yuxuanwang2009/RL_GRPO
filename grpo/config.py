import os
import torch
from dataclasses import dataclass

# All generated outputs (plots, metrics, checkpoints) go here
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


@dataclass
class GRPOConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # Training
    learning_rate: float = 2e-6  # Raised from 1e-7 — model needs stronger signal to learn reasoning patterns
    batch_size: int = 4
    val_batch_size: int = 64
    num_generations: int = 16
    max_new_tokens: int = 160  # Raised for countdown task chain-of-thought reasoning
    num_iterations: int = 1200
    temperature: float = 1.2

    # PPO/GRPO
    beta: float = 0.01  # Lowered from 0.025 — allow more policy divergence for exploration
    epsilon: float = 0.2
    num_inner_updates: int = 1  # Key stability fix: 1 update per batch to prevent KL explosion
    clip_grad_norm: float = 0.5  # Stricter clipping
    grad_accum_steps: int = 1  # Split PPO update into N gradient accumulation steps to reduce peak memory

    # Task
    num_count: int = 3
    oneshot: bool = False
    mix_oneshot: float = 0.0  # decay rate for one-shot curriculum (0.0 = always one-shot, 0.5 = decay to 50%, 1.0 = fully zero-shot by end)

    # Run management
    run_name: str = "grpo"
