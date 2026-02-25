"""Compare base model vs RL-trained model, with and without one-shot example."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import re
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo.config import OUTPUTS_DIR
from grpo.prompts import make_prompt
from grpo.validation import check_answer, safe_eval, extract_numbers
from grpo.datasets import generate_problems

MAX_NEW_TOKENS = 160


def evaluate_model(model_path, tokenizer, problems, oneshot, device, batch_size=16,
                    verbose=False, verbose_file=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    correct = 0
    no_answer_tag = 0
    wrong_result = 0
    wrong_numbers = 0
    empty_gen = 0

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
        for j, (comp, prob) in enumerate(zip(completions, batch)):
            is_correct = check_answer(comp, prob)
            if is_correct:
                correct += 1

            if verbose:
                idx = i + j
                tag = "CORRECT" if is_correct else "WRONG"
                failure_reason = ""
                if not is_correct:
                    match = re.search(r'<answer>(.*?)</answer>', comp)
                    if not match:
                        if comp.strip() == "":
                            failure_reason = " [EMPTY generation]"
                            empty_gen += 1
                        else:
                            failure_reason = " [no <answer> tag]"
                            no_answer_tag += 1
                    else:
                        expr = match.group(1).strip()
                        result = safe_eval(expr)
                        used = sorted(extract_numbers(expr))
                        expected = sorted(prob["nums"])
                        if result is None:
                            failure_reason = f" [parse error: {expr!r}]"
                            wrong_result += 1
                        elif result != prob["target"]:
                            failure_reason = f" [wrong value: {expr} = {result}, need {prob['target']}]"
                            wrong_result += 1
                        elif used != expected:
                            failure_reason = f" [wrong nums: used {used}, need {expected}]"
                            wrong_numbers += 1
                        else:
                            failure_reason = " [unknown]"
                out = verbose_file or sys.stdout
                print(f"\n{'='*70}", file=out)
                print(f"Problem {idx+1}: nums={prob['nums']}  target={prob['target']}  [{tag}]{failure_reason}", file=out)
                print(f"{'-'*70}", file=out)
                print(comp, file=out)

    if verbose:
        out = verbose_file or sys.stdout
        total_wrong = len(problems) - correct
        print(f"\n{'='*70}", file=out)
        print(f"Failure breakdown ({total_wrong} wrong):", file=out)
        print(f"  No <answer> tag : {no_answer_tag}", file=out)
        print(f"  Empty generation: {empty_gen}", file=out)
        print(f"  Wrong result    : {wrong_result}", file=out)
        print(f"  Wrong numbers   : {wrong_numbers}", file=out)

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
    parser.add_argument("--verbose", action="store_true", help="Print every completion with failure diagnosis")
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

    # Open verbose output file if requested
    verbose_file = None
    if args.verbose:
        verbose_path = os.path.join(OUTPUTS_DIR, "verbose_eval.txt")
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        verbose_file = open(verbose_path, "w")
        print(f"Verbose output will be written to {verbose_path}")

    results = []
    for label, path in model_entries:
        for shot_label, oneshot in modes:
            print(f"Evaluating: {label} ({shot_label}, {args.num_count} nums)...")
            correct, total = evaluate_model(path, tokenizer, problems, oneshot, device,
                                              verbose=args.verbose, verbose_file=verbose_file)
            acc = correct / total
            results.append((label, shot_label, correct, total, acc))
            print(f"  {correct}/{total} = {acc:.1%}")

    if verbose_file:
        verbose_file.close()

    print(f"\n{'='*60}")
    print(f"{'Model':<20} {'Prompt':<12} {'Nums':>4} {'Correct':>8} {'Accuracy':>10}")
    print(f"{'-'*60}")
    for label, shot, correct, total, acc in results:
        print(f"{label:<20} {shot:<12} {args.num_count:>4} {correct:>5}/{total:<3} {acc:>9.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
