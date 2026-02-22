import argparse
import json
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--run_name", type=str, default="grpo")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override model path for fresh start (e.g. saved_model)")
    parser.add_argument("--ref_model", type=str, default=None,
                        help="Explicit path to ref model (default: same as policy)")

    # Config overrides
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_generations", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--num_count", type=int, default=None, help="Numbers per problem (default: 3)")
    parser.add_argument("--oneshot", action="store_true", default=None, help="Include one-shot example in prompt")
    parser.add_argument("--mix_oneshot", type=float, default=None, help="Fraction of prompts with one-shot (0.0-1.0)")
    return parser.parse_args()


def apply_overrides(cfg, args):
    override_fields = [
        "learning_rate", "beta", "temperature", "num_iterations",
        "max_new_tokens", "num_generations", "batch_size", "val_batch_size",
        "epsilon", "num_count", "oneshot", "mix_oneshot", "run_name",
    ]
    for name in override_fields:
        val = getattr(args, name, None)
        if val is not None:
            print(f"  Override: {name} = {val}")
            setattr(cfg, name, val)


def checkpoint_dir(run_name):
    return os.path.join("checkpoints", run_name)


def save_checkpoint(policy, tokenizer, optimizer, step, metrics, cfg):
    ckpt_dir = checkpoint_dir(cfg.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Model + tokenizer
    policy_dir = os.path.join(ckpt_dir, "policy")
    policy.save_pretrained(policy_dir)
    tokenizer.save_pretrained(policy_dir)

    # Optimizer state
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))

    # Training state
    state = {
        "step": step,
        "metrics": metrics,
        "config": {
            "learning_rate": cfg.learning_rate,
            "beta": cfg.beta,
            "temperature": cfg.temperature,
            "num_iterations": cfg.num_iterations,
            "max_new_tokens": cfg.max_new_tokens,
            "num_generations": cfg.num_generations,
            "batch_size": cfg.batch_size,
            "val_batch_size": cfg.val_batch_size,
            "epsilon": cfg.epsilon,
            "run_name": cfg.run_name,
        },
    }
    with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2)

    print(f"Checkpoint saved to {ckpt_dir}/ (step {step})")


def load_checkpoint(run_name):
    ckpt_dir = checkpoint_dir(run_name)
    state_path = os.path.join(ckpt_dir, "training_state.json")

    if not os.path.exists(state_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_dir}/")

    with open(state_path, "r") as f:
        state = json.load(f)

    policy_path = os.path.join(ckpt_dir, "policy")
    optimizer_path = os.path.join(ckpt_dir, "optimizer.pt")

    print(f"Resuming from {ckpt_dir}/ (step {state['step']})")
    return {
        "policy_path": policy_path,
        "optimizer_state": torch.load(optimizer_path, weights_only=False) if os.path.exists(optimizer_path) else None,
        "step": state["step"],
        "metrics": state["metrics"],
        "config": state.get("config", {}),
    }
