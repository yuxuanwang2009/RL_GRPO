# Experiment Log

---

# Part I: One-shot + CoT training

GRPO training on 3-number countdown with Qwen2.5-1.5B-Instruct. The training prompt includes a one-shot example with chain-of-thought reasoning in `<think>...</think>` and answer in `<answer>...</answer>` tags. R1 denotes the model after 1200 steps of this training.

One-shot example used during training:
```
User: Using the numbers [3, 5, 2], create an expression that equals 13.
You must use all 3 numbers, each exactly once. Available operations: +, -, *
(no division). Show your reasoning in <think>...</think>, then write only the
bare expression in <answer>...</answer>.

Assistant:
<think>
I need to reach 13 using [3, 5, 2].
Try 3 + 5 + 2 = 10. Too small, need multiplication.
Try 3 * 5 + 2 = 17. Too large.
Try 5 * 2 + 3 = 13. Yes!
</think>
<answer>5 * 2 + 3</answer>
```

## Eval 1: Base vs R1 (3-number countdown, seed=142, n=100)

| Model | Prompt | Numbers | Correct | Accuracy |
|---|---|---|---|---|
| Base (Qwen2.5-1.5B-Instruct) | one-shot | 3 | 24/100 | 24.0% |
| Base (Qwen2.5-1.5B-Instruct) | zero-shot | 3 | 2/100 | 2.0% |
| R1 (RL, one-shot trained) | one-shot | 3 | 69/100 | 69.0% |
| R1 (RL, one-shot trained) | zero-shot | 3 | 3/100 | 3.0% |

**Takeaways:**
- RL training with one-shot improved one-shot accuracy from 24% to 69% (+45pp).
- Zero-shot performance did not transfer: 2% -> 3%. The one-shot example was load-bearing for format compliance.
- The model learned to solve the task *conditional on the one-shot prompt*, not unconditionally.

## Eval 2: Base vs R1 (4-number countdown, seed=142, n=100)

One-shot only. Tests whether 3-number RL training transfers to harder 4-number problems.

| Model | Prompt | Numbers | Correct | Accuracy |
|---|---|---|---|---|
| Base (Qwen2.5-1.5B-Instruct) | one-shot | 4 | 5/100 | 5.0% |
| R1 (RL, one-shot trained, 3 nums) | one-shot | 4 | 21/100 | 21.0% |

**Takeaways:**
- RL training on 3-number problems transferred partially to 4-number problems: 5% -> 21% (+16pp).
- The improvement is smaller than on 3 numbers (+45pp), as expected — 4 numbers is a harder task the model was never trained on.
- The arithmetic and format skills learned on 3 numbers generalize somewhat to 4 numbers.

## Eval 3: Zero-shot fine-tuning — warm start vs cold start (3-number, seed=142, n=100)

Base+ZS200: Base model fine-tuned 200 steps on zero-shot prompts (with `<answer>` tags, no one-shot example).
R1+ZS200: R1 (one-shot trained, 1200 steps) fine-tuned 200 steps on zero-shot prompts.
We evaluate whether R1's one-shot training helps subsequent zero-shot training.

| Model | Prompt | Numbers | Correct | Accuracy |
|---|---|---|---|---|
| Base | one-shot | 3 | 24/100 | 24.0% |
| Base | zero-shot | 3 | 2/100 | 2.0% |
| R1 | one-shot | 3 | 69/100 | 69.0% |
| R1 | zero-shot | 3 | 3/100 | 3.0% |
| Base+ZS200 | one-shot | 3 | 56/100 | 56.0% |
| Base+ZS200 | zero-shot | 3 | 35/100 | 35.0% |
| R1+ZS200 | one-shot | 3 | 70/100 | 70.0% |
| R1+ZS200 | zero-shot | 3 | 35/100 | 35.0% |

**Takeaways:**
- **No warm-start advantage for zero-shot.** Both Base+ZS200 and R1+ZS200 reach 35% zero-shot accuracy — R1's prior one-shot training gave no head start.
- **No catastrophic forgetting.** R1+ZS200 retains 70% one-shot accuracy (vs R1's 69%). Zero-shot fine-tuning didn't damage the one-shot skill.
- **Cross-prompt transfer from zero-shot training.** Base+ZS200 jumped from 24% to 56% on one-shot (+32pp) despite never training on one-shot prompts. Zero-shot RL improved general arithmetic ability that transfers to one-shot.
- **Asymmetric transfer.** One-shot RL didn't help zero-shot (Eval 1: 3%), but zero-shot RL *did* help one-shot (24% -> 56%). The zero-shot task forces the model to internalize the skill more deeply since it can't rely on the example.
- **CoT is not load-bearing for 3-number problems.** R1+ZS200, when prompted zero-shot, outputs the same expression in both `<think>` and `<answer>` — no genuine reasoning. The 3-number task has few enough combinations that the model can jump straight to the answer. This explains why one-shot training (which teaches CoT by example) didn't transfer: the model learned format mimicry, not reasoning. A harder task (4-5 numbers) might require real CoT and show different transfer dynamics.

## Eval 4: Transfer to 4-number zero-shot (seed=142, n=100)

Tests whether zero-shot 3-number training generalizes to harder 4-number problems.

| Model | Prompt | Numbers | Correct | Accuracy |
|---|---|---|---|---|
| Base | zero-shot | 4 | 1/100 | 1.0% |
| R1 | zero-shot | 4 | 1/100 | 1.0% |
| Base+ZS200 | zero-shot | 4 | 11/100 | 11.0% |
| R1+ZS200 | zero-shot | 4 | 6/100 | 6.0% |

**Takeaways:**
- **Base+ZS200 outperforms R1+ZS200 on 4-number zero-shot** (11% vs 6%). Prior one-shot training may have hurt generalization to harder problems.
- **Base+ZS200 zero-shot (11%) > Base one-shot (5%)** on 4 numbers, despite never training on 4-number problems. Zero-shot RL on 3 numbers transfers better to harder tasks than one-shot prompting alone.
- **R1+ZS200 zero-shot (6%) << R1 one-shot (21%)** on 4 numbers. R1's strong 4-number performance was contingent on the one-shot prompt; without it, the skill largely disappears, and the 200 steps of zero-shot training weren't enough to rebuild it.
- **One-shot RL may over-specialize.** R1 learned representations tightly coupled to the one-shot format. When that scaffolding is removed, it performs worse than a model that learned from scratch without it.

## Eval 5: Natural language evaluation — no format tags (3-number, seed=142, n=100)

Tests whether RL training transfers to completely unstructured prompts with no `<think>` or `<answer>` tags. Answer extracted by finding the last valid math expression in the output.

| Model | Prompt | Numbers | Correct | Accuracy |
|---|---|---|---|---|
| Base | natural | 3 | 6/100 | 6.0% |
| Base+ZS200 | natural | 3 | 21/100 | 21.0% |

**Takeaways:**
- RL training transfers to tag-free prompts: 6% -> 21% (+15pp). The model learned real arithmetic, not just tag formatting.
- The gap vs tagged zero-shot (35%) is partly due to noisier answer extraction and partly because the model does benefit from `<answer>` as a structured output signal.

---

# Part I Summary

One-shot + CoT training teaches the model to solve countdown problems effectively (69% on 3 numbers) but the skill is tightly coupled to the training prompt format. Key findings:
1. One-shot skill doesn't transfer to zero-shot (69% -> 3%)
2. Zero-shot training produces more generalizable skills than one-shot training
3. CoT (`<think>`) is not load-bearing for 3-number problems — the model skips reasoning
4. `<answer>` tags provide clean reward signal and the underlying arithmetic skill transfers even without them (Eval 5: 21%)

---

# Part II: Prompt strategy comparison (3-number countdown)

Compare three training configurations on 3-number countdown, all starting from base Qwen2.5-1.5B-Instruct. The prompt includes <think> and <answer> tags. Both get format reward (+0.1 each). max_new_tokens = 160.

| Run | Config | Description |
|---|---|---|
| p2_zs | zero-shot | Zero-shot only. Baseline from Part I findings. |
| p2_os | one-shot | One-shot only. CoT example provided every step. |
| p2_halfshot | half-shot | Linear decay: 100% one-shot at step 0, 0% at step 1200. Per-batch coin flip. |

**Questions:**
1. Does one-shot CoT scaffolding help training accuracy vs zero-shot?
2. Does alternating (half-shot) get the best of both — learning reasoning from the example while forcing generalization on zero-shot steps?
3. At eval time, how do all three perform on both one-shot and zero-shot prompts? Does half-shot training produce better zero-shot transfer than pure one-shot?

## Eval 6: p2_zeroshot and p2_halfshot (3-number, seed=142, n=200)

All models: 1200 steps, max_new_tokens=160, +0.1 format reward for both <think> and <answer>.
- p2_zeroshot: pure zero-shot training.
- p2_halfshot: linear decay from 100% one-shot at step 0 to 0% at step 1200. Per-batch coin flip.

| Model | Training | Prompt | Numbers | Correct | Accuracy |
|---|---|---|---|---|---|
| R1 (Part I) | one-shot | one-shot | 3 | 69/100 | 69.0% |
| R1 (Part I) | one-shot | zero-shot | 3 | 3/100 | 3.0% |
| p2_zeroshot | zero-shot | one-shot | 3 | 76/200 | 38.0% |
| p2_zeroshot | zero-shot | zero-shot | 3 | 75/200 | 37.5% |
| p2_halfshot | half-shot | one-shot | 3 | 121/200 | 60.5% |
| p2_halfshot | half-shot | zero-shot | 3 | 68/200 | 34.0% |

**Takeaways:**
- **Both skills saturate early.** Zero-shot: 35% at 200 steps (Part I), 34–37.5% at 600–1200 steps. One-shot: 60.5% at ~600 batches vs 69% at 1200. Diminishing returns in both cases.
- **One-shot and zero-shot are largely independent skills at 1.5B scale.** Interleaving does not force shared representations. The model learns separate prompt-specific behaviors. There may be some shared learning (both directions show slightly better than half-budget performance), but the effect is small and confounded by early saturation.

## Eval 7: Natural language prompts (3-number, seed=142, n=200)

Tests whether training strategy affects generalization to natural language prompts with no format tags. Answer extracted by finding the last valid math expression in output.

| Model | Training | Prompt | Numbers | Correct | Accuracy |
|---|---|---|---|---|---|
| Base (Part I) | — | natural | 3 | 6/100 | 6.0% |
| Base+ZS200 (Part I) | zero-shot | natural | 3 | 21/100 | 21.0% |
| p2_zeroshot | zero-shot | natural | 3 | 33/200 | 16.5% |
| p2_halfshot | half-shot | natural | 3 | 33/200 | 16.5% |

**Takeaways:**
- **Identical natural language performance.** p2_zeroshot and p2_halfshot both get 16.5%. One-shot exposure in p2_halfshot did not help or hurt generalization to unstructured prompts.
- **Both improve over base (6%)** but are below Base+ZS200 (21%) from Part I. The difference may be due to eval sample size (n=100 vs n=200) or other training differences.
- **Format-specific skills don't transfer to natural prompts.** The gap between tagged eval (~34–38%) and natural eval (16.5%) shows the models rely heavily on `<answer>` tags for structured output.
- **Nevertheless, half-shot training yields well-balanced results.** p2_halfshot matches p2_zeroshot on both zero-shot (34% vs 37.5%) and natural (16.5% vs 16.5%) prompts, while being substantially better on one-shot (60.5% vs 38%). It's the only model that performs reasonably across all prompt types.

---

# Part III: Are 1200 steps too long?

Part II showed that p2_zeroshot (1200 steps) underperformed Base+ZS200 (Part I, 200 steps) on one-shot and natural prompts despite training longer. Hypothesis: longer training overfits to the training prompt format, hurting cross-format generalization.

## Eval 8: p2_zeroshot_200 vs p2_zeroshot (3-number, seed=142, n=200)

Both models: pure zero-shot training, +0.1 format reward for both <think> and <answer>, max_new_tokens=160. Only difference is training steps.

| Model | Steps | Prompt | Numbers | Correct | Accuracy |
|---|---|---|---|---|---|
| p2_zeroshot_200 | 200 | one-shot | 3 | 103/200 | 51.5% |
| p2_zeroshot_200 | 200 | zero-shot | 3 | 58/200 | 29.0% |
| p2_zeroshot_200 | 200 | natural | 3 | 44/200 | 22.0% |
| p2_zeroshot | 1200 | one-shot | 3 | 76/200 | 38.0% |
| p2_zeroshot | 1200 | zero-shot | 3 | 75/200 | 37.5% |
| p2_zeroshot | 1200 | natural | 3 | 33/200 | 16.5% |

**Takeaways:**
- **Longer training hurts generalization.** 200 steps beats 1200 steps on one-shot (51.5% vs 38%) and natural (22% vs 16.5%), despite 1200 steps being better on zero-shot (37.5% vs 29%).
- **Early training builds general skill, late training overfits.** The first 200 steps improve arithmetic broadly across all prompt formats. The remaining 1000 steps only improve the training distribution (zero-shot tagged) while degrading other formats.
- **Confirms Part I.** p2_zeroshot_200 closely matches Base+ZS200 from Part I (51.5% vs 56% one-shot, 29% vs 35% zero-shot, 22% vs 21% natural), validating those earlier results.
- **Training reward is misleading.** The reward continues to increase from step 200 to 1200, but this reflects overfitting to the prompt format, not genuine improvement in arithmetic ability.

## Eval 9: p2_halfshot_400 vs p2_halfshot (3-number, seed=142, n=200)

Both models: half-shot training (linear decay from 100% one-shot to 0%), +0.1 format reward for both <think> and <answer>, max_new_tokens=160. Only difference is training steps.

| Model | Steps | Prompt | Numbers | Correct | Accuracy |
|---|---|---|---|---|---|
| p2_halfshot_400 | 400 | one-shot | 3 | 108/200 | 54.0% |
| p2_halfshot_400 | 400 | zero-shot | 3 | 64/200 | 32.0% |
| p2_halfshot_400 | 400 | natural | 3 | 49/200 | 24.5% |
| p2_halfshot | 1200 | one-shot | 3 | 121/200 | 60.5% |
| p2_halfshot | 1200 | zero-shot | 3 | 68/200 | 34.0% |
| p2_halfshot | 1200 | natural | 3 | 33/200 | 16.5% |

**Takeaways:**
- **Same overfitting pattern as zero-shot.** 400 steps beats 1200 on natural (24.5% vs 16.5%), while 1200 is only marginally better on one-shot (60.5% vs 54%) and zero-shot (34% vs 32%).
- **p2_halfshot_400 is the best natural language performer across all experiments** (24.5%), slightly beating p2_zeroshot_200 (22%). Half-shot at 400 steps may be the sweet spot for broad generalization.
- **Half-shot overfits slower than zero-shot.** Zero-shot peaked at ~200 steps for generalization; half-shot peaks later (~400 steps), likely because the one-shot batches provide more diverse training signal early on.

## Eval 10: Periodic eval over extended zero-shot training (3-number, seed=142, n=100)

Pure zero-shot training evaluated every 100 steps with eval_compare (zs + os) and eval_natural on a fixed problem set, extended to ~2800 steps.

![p3_zeroshot training curve](p3_zeroshot_training_curve.png)

**Takeaways:**
- **Overfitting is progressive and format-dependent.** The further a prompt format is from the training distribution (zero-shot tagged), the earlier it peaks and starts declining: one-shot peaks first (~step 200–300 at ~50%+), natural plateaus before step 1000 then decays, while zero-shot keeps climbing.
- **Zero-shot is the only metric that doesn't plateau.** Train accuracy and zero-shot eval continue to rise past step 2000, confirming that the model keeps learning — but only for the specific format it's trained on.
- **Confirms the overfitting hypothesis from Evals 8–9** with continuous measurement rather than point comparisons. Longer training narrows the model's competence toward the exact training prompt format.

---

# Part IV: Conclusions

1. **One-shot learning is a highly efficient way to enhance performance.** Providing an in-context example during RL training leads to rapid accuracy gains (24% → 69% on 3-number countdown in 1200 steps).

2. **One-shot learning does not transfer beyond a threshold.** Skills learned via one-shot training are tightly coupled to the one-shot prompt format and do not transfer to zero-shot or natural language inference (69% one-shot → 3% zero-shot).

3. **Zero-shot learning transfers to one-shot and natural language prompts — if training is stopped at the right time.** Early zero-shot training builds general arithmetic skill that benefits all prompt formats (e.g., 200 steps of zero-shot training: 29% ZS, 51.5% OS, 22% natural). Continued training erodes this generalization.

4. **Curriculum learning (one-shot → zero-shot) does not show meaningful skill transfer at 1.5B scale.** Gradually shifting from one-shot to zero-shot prompts during training does not produce shared representations. The model learns separate prompt-specific behaviors, and interleaving does not force generalization beyond what each format achieves independently.

5. **Reward is not a reliable stopping criterion.** Optimizing for zero-shot too long causes deterioration at one-shot and natural language inference, even while training reward continues to rise. In practice, training should be guided by a custom eval suite covering the tasks we care about, not by the training reward signal.
