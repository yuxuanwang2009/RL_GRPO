# GRPO (Group Relative Policy Optimization) — Countdown Task

**[Writeup: Teaching a 1.5B Model Arithmetic with RL: What Transfers, What Doesn't, and When to Stop](docs/writeup.pdf)**

## Overview
A minimal GRPO training loop that teaches Qwen2.5-1.5B-Instruct to solve countdown puzzles: given 3 numbers and a target, find an arithmetic expression using all 3 numbers exactly once with `+`, `-`, `*`.

The policy is trained against a frozen reference model with a KL penalty (Schulman estimator) to prevent collapse, using PPO-style clipped objectives.

## Project Structure

```
RL_GRPO/
├── grpo/                  # Core library package
│   ├── config.py          # GRPOConfig dataclass
│   ├── prompts.py         # Prompt templates and builders
│   ├── validation.py      # safe_eval, reward_function, answer checking
│   ├── datasets.py        # Problem generation
│   ├── training.py        # Batch log-probs, evaluation helpers
│   ├── checkpoint.py      # Save/load/resume utilities
│   └── plotting.py        # Training curve visualization
├── scripts/               # Entry points
│   ├── train.py           # Unified training script
│   ├── eval_compare.py    # One-shot vs zero-shot evaluation
│   ├── eval_natural.py    # Natural language evaluation
│   ├── inference.py       # Single-question inference
│   └── mem_profile.py     # VRAM profiling
├── slurm/                 # HPC job scripts
│   └── run_grpo.sh
├── docs/                  # Research documentation
│   ├── writeup.md         # Main findings
│   ├── writeup.pdf
│   └── log.md             # Detailed experiment log
└── outputs/               # Generated artifacts (gitignored)
    ├── *.png              # Training curves
    ├── *_metrics.json     # Logged metrics
    └── checkpoints/       # Model checkpoints
```

## How to Run

### Train (`scripts/train.py`)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train.py [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--run_name` | str | `grpo` | Name for this run (used in output filenames and checkpoint dir) |
| `--resume` | flag | off | Resume training from `outputs/checkpoints/{run_name}/` |
| `--model_name` | str | `Qwen/Qwen2.5-1.5B-Instruct` | Policy model path (HuggingFace ID or local dir) |
| `--ref_model` | str | same as policy | Reference model path (for KL penalty) |
| `--oneshot` | flag | off | Use one-shot prompt (with worked example) |
| `--mix_oneshot` | float | `0.0` | Fraction of one-shot prompts; linear decay from 100% to 0% over training |
| `--eval_every` | int | disabled | Run cross-format eval (zs + os + natural) every N steps |
| `--num_count` | int | `3` | Numbers per problem |
| `--num_iterations` | int | `1200` | Total training steps |
| `--learning_rate` | float | `2e-6` | Learning rate |
| `--batch_size` | int | `4` | Prompts per step |
| `--val_batch_size` | int | `64` | Validation problems per eval |
| `--num_generations` | int | `16` | Completions sampled per prompt |
| `--max_new_tokens` | int | `160` | Max tokens per completion |
| `--temperature` | float | `1.2` | Sampling temperature |
| `--beta` | float | `0.01` | KL penalty weight |
| `--epsilon` | float | `0.2` | PPO clip range |

Examples:
```bash
# Basic zero-shot training
python scripts/train.py

# One-shot training with custom name
python scripts/train.py --oneshot --run_name r1

# Short run (your writeup found 200 steps generalizes best)
python scripts/train.py --num_iterations 200 --run_name zs200

# With periodic cross-format evaluation every 100 steps
python scripts/train.py --eval_every 100 --run_name p3_zeroshot

# Resume an interrupted run
python scripts/train.py --resume --run_name p3_zeroshot

# Mixed one-shot/zero-shot curriculum
python scripts/train.py --mix_oneshot 0.5 --run_name halfshot

# 4-number problems
python scripts/train.py --num_count 4 --run_name four_num
```

Outputs (all in `outputs/`):
- `{run_name}_training_curve.png`, `{run_name}_kl_divergence.png` — updated live
- `{run_name}_metrics.json` — reward, accuracy, KL per logging interval
- `checkpoints/{run_name}/` — model weights, optimizer state, training state

### Evaluate (`scripts/eval_compare.py`)

Compare models on one-shot vs zero-shot prompts with `<think>`/`<answer>` tags.

```bash
python scripts/eval_compare.py [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--models` | str+ | (required) | `label:run_name` pairs; use `label:hf://ID` for HuggingFace models |
| `--num_count` | int | `3` | Numbers per problem |
| `--num_problems` | int | `100` | Size of eval set |
| `--seed` | int | `142` | RNG seed for reproducible problems |
| `--oneshot_only` | flag | off | Only run one-shot eval |
| `--zeroshot_only` | flag | off | Only run zero-shot eval |

Example:
```bash
python scripts/eval_compare.py --models "Base:hf://Qwen/Qwen2.5-1.5B-Instruct" "R1:r1"
```

### Evaluate Natural (`scripts/eval_natural.py`)

Evaluate on plain language prompts (no `<think>`/`<answer>` tags). Answer extracted by finding the last valid math expression in the output.

```bash
python scripts/eval_natural.py [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--models` | str+ | (required) | `label:run_name` pairs; use `label:hf://ID` for HuggingFace models |
| `--num_count` | int | `3` | Numbers per problem |
| `--num_problems` | int | `100` | Size of eval set |
| `--seed` | int | `142` | RNG seed for reproducible problems |

Example:
```bash
python scripts/eval_natural.py --models "Base:hf://Qwen/Qwen2.5-1.5B-Instruct" "ZS200:zs200"
```

### Inference (`scripts/inference.py`)

Run a trained model on a single question.

```bash
python scripts/inference.py QUESTION [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `QUESTION` | str | (required) | The countdown puzzle question |
| `--run_name` | str | (required) | Run name (loads from `outputs/checkpoints/<run_name>/policy`) |
| `--no_oneshot` | flag | off | Skip one-shot example in prompt |

Example:
```bash
python scripts/inference.py --run_name r1 "Using the numbers [4, 7, 3], create an expression that equals 25."
python scripts/inference.py --run_name r1 --no_oneshot "..."
```

### Plot Metrics (`grpo.plotting`)

Re-plot from a saved metrics JSON.

```bash
python -m grpo.plotting [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | str | `grpo_metrics.json` | Path to metrics JSON file |
| `--prefix` | str | derived from input | Output filename prefix |
| `--smooth` | flag | off | Apply moving average smoothing (window=10) |

Example:
```bash
python -m grpo.plotting --input outputs/grpo_metrics.json --smooth
```

## Implementation Notes
- Uses **8-bit AdamW** (bitsandbytes) to reduce optimizer memory
- **Gradient checkpointing** enabled during training, disabled during generation/evaluation
- KV cache toggled on for generation, off during training (incompatible with gradient checkpointing)
- Reference model is frozen with `requires_grad=False`, never updated
- Optimizer: `bnb.optim.AdamW8bit`, model precision: `bfloat16`
- All outputs go to `outputs/` (resolved from package location, works regardless of cwd)
- Requires `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid CUDA OOM from memory fragmentation
