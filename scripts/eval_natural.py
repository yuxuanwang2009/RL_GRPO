"""Evaluate models on countdown task with natural language prompts (no format tags)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo.config import OUTPUTS_DIR
from grpo.prompts import make_natural_prompt
from grpo.validation import check_answer_natural
from grpo.datasets import generate_problems

MAX_NEW_TOKENS = 160


def evaluate_model(model_path, tokenizer, problems, device, batch_size=16):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    correct = 0
    for i in range(0, len(problems), batch_size):
        batch = problems[i:i+batch_size]
        prompts = [make_natural_prompt(p, tokenizer) for p in batch]

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
            if check_answer_natural(comp, prob):
                correct += 1

    del model
    torch.cuda.empty_cache()
    return correct, len(problems)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_count", type=int, default=3)
    parser.add_argument("--num_problems", type=int, default=100)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--models", nargs="+", required=True,
                        help="label:run_name pairs (e.g. 'ZS200:zs200'); use label:hf://ID for HuggingFace models (e.g. 'Base:hf://Qwen/Qwen2.5-1.5B-Instruct')")
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

    results = []
    for label, path in model_entries:
        print(f"Evaluating: {label} (natural prompt, {args.num_count} nums)...")
        correct, total = evaluate_model(path, tokenizer, problems, device)
        acc = correct / total
        results.append((label, correct, total, acc))
        print(f"  {correct}/{total} = {acc:.1%}")

    print(f"\n{'='*55}")
    print(f"{'Model':<20} {'Prompt':<10} {'Nums':>4} {'Correct':>8} {'Accuracy':>10}")
    print(f"{'-'*55}")
    for label, correct, total, acc in results:
        print(f"{label:<20} {'natural':<10} {args.num_count:>4} {correct:>5}/{total:<3} {acc:>9.1%}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
