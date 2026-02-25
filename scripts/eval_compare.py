"""Compare base model vs RL-trained model, with and without one-shot example."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo.config import OUTPUTS_DIR
from grpo.prompts import make_prompt
from grpo.validation import check_answer
from grpo.datasets import generate_problems

MAX_NEW_TOKENS = 160


def evaluate_model(model_path, tokenizer, problems, oneshot, device, batch_size=16):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    correct = 0
    for i in range(0, len(problems), batch_size):
        batch = problems[i:i+batch_size]
        prompts = [make_prompt(p, tokenizer, oneshot) for p in batch]

        tokenizer.padding_side = 'left'
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        completions = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        for comp, prob in zip(completions, batch):
            if check_answer(comp, prob):
                correct += 1

    del model
    torch.cuda.empty_cache()
    return correct, len(problems)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_count", type=int, default=3, help="Number of integers per problem")
    parser.add_argument("--num_problems", type=int, default=100)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--models", nargs="+", required=True,
                        help="label:run_name pairs (e.g. 'R1:r1'); use label:hf://ID for HuggingFace models (e.g. 'Base:hf://Qwen/Qwen2.5-1.5B-Instruct')")
    parser.add_argument("--oneshot_only", action="store_true", help="Only run one-shot eval")
    parser.add_argument("--zeroshot_only", action="store_true", help="Only run zero-shot eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problems = generate_problems(args.num_problems, args.seed, args.num_count)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

    # Parse model configs: label:run_name or label:hf://model_id
    model_entries = []
    for m in args.models:
        label, ref = m.split(":", 1)
        if ref.startswith("hf://"):
            path = ref[5:]
        else:
            path = os.path.join(OUTPUTS_DIR, "checkpoints", ref, "policy")
        model_entries.append((label, path))

    # Determine which prompt modes to run
    modes = []
    if not args.zeroshot_only:
        modes.append(("one-shot", True))
    if not args.oneshot_only:
        modes.append(("zero-shot", False))

    results = []
    for label, path in model_entries:
        for shot_label, oneshot in modes:
            print(f"Evaluating: {label} ({shot_label}, {args.num_count} nums)...")
            correct, total = evaluate_model(path, tokenizer, problems, oneshot, device)
            acc = correct / total
            results.append((label, shot_label, correct, total, acc))
            print(f"  {correct}/{total} = {acc:.1%}")

    print(f"\n{'='*60}")
    print(f"{'Model':<20} {'Prompt':<12} {'Nums':>4} {'Correct':>8} {'Accuracy':>10}")
    print(f"{'-'*60}")
    for label, shot, correct, total, acc in results:
        print(f"{label:<20} {shot:<12} {args.num_count:>4} {correct:>5}/{total:<3} {acc:>9.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
