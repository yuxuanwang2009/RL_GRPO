# GRPO (Group Relative Policy Optimization) — Reinforcement Learning

## Overview
This project implements a minimal, end-to-end GRPO training loop for a small instruction-tuned model, with a focus on multi-step arithmetic as the first testbed. The code trains a policy model against a frozen reference model, uses a KL regularizer computed with the Schulman estimator to keep KL strictly positive at small sample sizes, and tracks reward/accuracy/KL throughout training.

Key design goals:
- Keep the GRPO loop readable and auditable.
- Use a realistic task (multi-step arithmetic) where the base model already performs reasonably.
- Shape rewards to encourage correct formatting and correct final answers.

## What’s Included
- Training script: [train_grpo.py](train_grpo.py)
- Inference script: [inference.py](inference.py)
- Metrics plotting: [plot_metrics.py](plot_metrics.py)
- Saved model directory (after training): (saved_model/)

## Algorithm Highlights
- **GRPO updates**: multiple inner policy updates per batch using a PPO-style clipped objective.
- **Schulman KL estimator**: KL is estimated as $\exp(\Delta) - \Delta - 1$ (where $\Delta=\log \pi_\theta - \log \pi_{ref}$), which stays strictly positive even with small sample sizes.
- **Reward shaping**: rewards are assigned for correct output formatting first, then final correctness.

## Task and Reward
The task generates expressions of the form $(a \;\text{op}_1\; b)\; \text{op}_2\; c$ and prompts the model to answer in `<answer>...</answer>` format.

Reward structure (see [train_grpo.py](train_grpo.py)):
- +0.2 for correct `<answer>...</answer>` formatting
- +0.2 if the numeric answer is within ±1 of the ground truth
- +0.6 for the exact numeric answer

## Training Results (Reported)
From the latest run described in the project notes:
- Accuracy on multi-step arithmetic improved from ~30% to ~70% after 2,000 steps.
- KL divergence stayed low (< 2.0) relative to the starting model.
- The training curve is noisy but shows a learning signal.

Generated plots (run [plot_metrics.py](plot_metrics.py) or training to create them):

![Reward and Accuracy](grpo_training_curve.png)

![KL Divergence](grpo_kl_divergence.png)

## Practical Insights from Implementation
- **Implementation is the easy part**: getting GRPO to work required careful task choice, reward design, and hyperparameter tuning.
- **Catastrophic forgetting and reward hacking**: both were observed and mitigated by tuning `beta` and making the reward more robust.
- **Stop criteria matter**: training longer did not reliably improve outcomes—knowing when to stop RL is important.
- **Task choice matters**: spelling a word backwards was too difficult due to tokenization, while multi-step arithmetic provided a realistic and learnable task.

## How to Run
### 1) Train
Run the training loop and save the model and metrics:

```bash
python train_grpo.py
```

Outputs:
- saved model in [saved_model/](saved_model/)
- metrics in [grpo_metrics.json](grpo_metrics.json)
- plots: [grpo_training_curve.png](grpo_training_curve.png), [grpo_kl_divergence.png](grpo_kl_divergence.png)

### 2) Inference
Use the trained model to answer a question:

```bash
python inference.py "What is (4 - 9) * 6?"
```

### 3) Plot Metrics
Re-generate plots from saved metrics:

```bash
python plot_metrics.py
```

## Configuration
Key hyperparameters (see [train_grpo.py](train_grpo.py)):
- `learning_rate`: 1e-7
- `batch_size`: 4
- `num_generations`: 8
- `max_new_tokens`: 16
- `num_iterations`: 2000
- `beta`: 0.04
- `epsilon`: 0.2
- `num_inner_updates`: 5
- `clip_grad_norm`: 0.5

## Notes
- The default device is CUDA; change `cfg.device` if needed (e.g., `mps` on Apple Silicon).
- The saved model directory is used automatically if present.

## Acknowledgments
The training loop is inspired by the GRPO formulation and follows practical guidance from the DeepSeekMath paper, with KL stabilized via the Schulman estimator.
