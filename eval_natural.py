"""Evaluate models on countdown task with natural language prompts (no format tags)."""

import torch
import random
import re
import ast
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 160


def generate_problems(n, seed, num_count=3):
    random.seed(seed)
    problems = []
    ops = ['+', '-', '*']
    for _ in range(n):
        while True:
            nums = [random.randint(1, 10) for _ in range(num_count)]
            chosen_ops = [random.choice(ops) for _ in range(num_count - 1)]
            expr = str(nums[0])
            for i, op in enumerate(chosen_ops):
                expr = f"({expr} {op} {nums[i+1]})"
            target = eval(expr)
            if 1 <= target <= 100 and target not in nums:
                break
        random.shuffle(nums)
        problems.append({"target": target, "nums": nums})
    return problems


def make_prompt(problem, tokenizer):
    num_count = len(problem["nums"])
    question = (
        f"Using the numbers {problem['nums']}, create an expression that equals {problem['target']}. "
        f"You must use all {num_count} numbers, each exactly once. "
        f"Available operations: +, -, * (no division)."
    )
    messages = [
        {"role": "system", "content": "You are a mathematical puzzle solver."},
        {"role": "user", "content": question},
    ]
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


def extract_last_expression(completion):
    """Find the last valid math expression in the completion."""
    candidates = re.findall(r'[\d][\d\s\+\-\*\(\)]+[\d\)]', completion)
    # Try candidates in reverse order, return first that parses
    for candidate in reversed(candidates):
        candidate = candidate.strip()
        if safe_eval(candidate) is not None:
            return candidate
    return None


def check_answer(completion, problem):
    target = problem["target"]
    available = sorted(problem["nums"])

    expr = extract_last_expression(completion)
    if expr is None:
        return False

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


def evaluate_model(model_path, tokenizer, problems, device, batch_size=16):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    correct = 0
    for i in range(0, len(problems), batch_size):
        batch = problems[i:i+batch_size]
        prompts = [make_prompt(p, tokenizer) for p in batch]

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
    parser.add_argument("--num_count", type=int, default=3)
    parser.add_argument("--num_problems", type=int, default=100)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--models", nargs="+", default=None,
                        help="model_label:path pairs, e.g. 'Base:Qwen/Qwen2.5-1.5B-Instruct'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problems = generate_problems(args.num_problems, args.seed, args.num_count)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

    if args.models:
        model_entries = []
        for m in args.models:
            label, path = m.split(":", 1)
            model_entries.append((label, path))
    else:
        model_entries = [
            ("Base", "Qwen/Qwen2.5-1.5B-Instruct"),
            ("Base+ZS200", "checkpoints/base_zs3/policy"),
        ]

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
