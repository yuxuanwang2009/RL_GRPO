"""Compare base model vs RL-trained model, with and without one-shot example."""

import torch
import json
import random
import re
import ast
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 160

_EXAMPLE_Q = (
    "Using the numbers [3, 5, 2], create an expression that equals 13. "
    "You must use all 3 numbers, each exactly once. Available operations: +, -, * (no division). "
    "Show your reasoning in <think>...</think>, then write only the bare expression in <answer>...</answer>."
)
_EXAMPLE_A = (
    "<think>\n"
    "I need to reach 13 using [3, 5, 2].\n"
    "Try 3 + 5 + 2 = 10. Too small, need multiplication.\n"
    "Try 3 * 5 + 2 = 17. Too large.\n"
    "Try 5 * 2 + 3 = 13. Yes!\n"
    "</think>\n"
    "<answer>5 * 2 + 3</answer>"
)


def generate_problems(n, seed, num_count=3):
    random.seed(seed)
    problems = []
    ops = ['+', '-', '*']
    for _ in range(n):
        while True:
            nums = [random.randint(1, 10) for _ in range(num_count)]
            chosen_ops = [random.choice(ops) for _ in range(num_count - 1)]
            # Build left-associative expression: ((a op1 b) op2 c) op3 d ...
            expr = str(nums[0])
            for i, op in enumerate(chosen_ops):
                expr = f"({expr} {op} {nums[i+1]})"
            target = eval(expr)
            if 1 <= target <= 100 and target not in nums:
                break
        random.shuffle(nums)
        problems.append({"target": target, "nums": nums})
    return problems


def make_prompt(problem, tokenizer, oneshot):
    num_count = len(problem["nums"])
    question = (
        f"Using the numbers {problem['nums']}, create an expression that equals {problem['target']}. "
        f"You must use all {num_count} numbers, each exactly once. Available operations: +, -, * (no division). "
        f"Show your reasoning in <think>...</think>, then write only the bare expression in <answer>...</answer>."
    )
    messages = [{"role": "system", "content": "You are a mathematical puzzle solver."}]
    if oneshot:
        messages.append({"role": "user", "content": _EXAMPLE_Q})
        messages.append({"role": "assistant", "content": _EXAMPLE_A})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def safe_eval(expr_str):
    try:
        tree = ast.parse(expr_str.strip(), mode='eval')
    except SyntaxError:
        return None
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.USub)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            return None
    try:
        return eval(compile(tree, '<string>', 'eval'))
    except Exception:
        return None


def check_answer(completion, problem):
    target = problem["target"]
    available = sorted(problem["nums"])
    match = re.search(r'<answer>(.*?)</answer>', completion)
    if not match:
        return False
    expr = match.group(1).strip()
    result = safe_eval(expr)
    if result is None or result != target:
        return False
    used = sorted(int(x) for x in re.findall(r'\d+', expr))
    pool = available.copy()
    for n in used:
        if n in pool:
            pool.remove(n)
        else:
            return False
    return len(pool) == 0


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
    parser.add_argument("--models", nargs="+", default=None,
                        help="model_label:path pairs, e.g. 'Base:Qwen/Qwen2.5-1.5B-Instruct' 'R1:saved_model'")
    parser.add_argument("--oneshot_only", action="store_true", help="Only run one-shot eval")
    parser.add_argument("--zeroshot_only", action="store_true", help="Only run zero-shot eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problems = generate_problems(args.num_problems, args.seed, args.num_count)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

    # Parse model configs
    if args.models:
        model_entries = []
        for m in args.models:
            label, path = m.split(":", 1)
            model_entries.append((label, path))
    else:
        model_entries = [
            ("Base model", "Qwen/Qwen2.5-1.5B-Instruct"),
            ("RL-trained model", "saved_model"),
        ]

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
