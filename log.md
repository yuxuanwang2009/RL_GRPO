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
| p2_mix | half-shot | Linear decay: 100% one-shot at step 0, 0% at step 1200. Per-batch coin flip. |

**Questions:**
1. Does one-shot CoT scaffolding help training accuracy vs zero-shot?
2. Does alternating (half-shot) get the best of both — learning reasoning from the example while forcing generalization on zero-shot steps?
3. At eval time, how do all three perform on both one-shot and zero-shot prompts? Does half-shot training produce better zero-shot transfer than pure one-shot?

## Eval 6: p2_think preliminary (3-number, seed=142, n=100)

p2_think: zero-shot trained, 1200 steps, prompt mentions <think> but no <think> format reward, max_new_tokens=160. Note: R1 (Part I) had +0.1 format reward for <think>; p2_think does not. The model learned to output literal <think>...</think> (copying the prompt instruction) then <answer>expr</answer> — no genuine reasoning.

| Model | Prompt | Numbers | Correct | Accuracy |
|---|---|---|---|---|
| R1 (Part I) | one-shot | 3 | 69/100 | 69.0% |
| R1 (Part I) | zero-shot | 3 | 3/100 | 3.0% |
| p2_think | one-shot | 3 | 32/100 | 32.0% |
| p2_think | zero-shot | 3 | 39/100 | 39.0% |

**Takeaways:**
- **Zero-shot > one-shot for p2_think** (39% vs 32%) — the inverse of R1 (69% vs 3%). Each model performs best on its training distribution.
- **One-shot example hurts p2_think.** The example shows genuine CoT reasoning, but the model learned to skip CoT. The example pushes it toward a generation pattern it wasn't optimized for.
- **Zero-shot accuracy (39%) comparable to Base+ZS200 (35%)** from Part I, despite 1200 steps vs 200. The <think> mention in the prompt and extra tokens (160 vs 160) didn't significantly help — the model just ignored <think>.
- **CoT remains unused for 3-number problems** regardless of whether it's mentioned in the prompt. The model rationally skips reasoning tokens since they don't improve accuracy enough to justify the cost.

