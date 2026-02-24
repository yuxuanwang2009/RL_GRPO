import json
import random


EXAMPLE_Q = (
    "Using the numbers [3, 5, 2], create an expression that equals 13. "
    "You must use all 3 numbers, each exactly once. Available operations: +, -, * (no division). "
    "Show your reasoning in <think>...</think>, then write only the bare expression in <answer>...</answer>."
)
EXAMPLE_A = (
    "<think>\n"
    "I need to reach 13 using [3, 5, 2].\n"
    "Try 3 + 5 + 2 = 10. Too small, need multiplication.\n"
    "Try 3 * 5 + 2 = 17. Too large.\n"
    "Try 5 * 2 + 3 = 13. Yes!\n"
    "</think>\n"
    "<answer>5 * 2 + 3</answer>"
)


def get_prompt(split="train", tokenizer=None, num_count=3, oneshot=False):
    """Generate a random countdown problem prompt.

    Hidden split rule:
      train -> at least one of the numbers is odd
      val   -> all numbers are even
    """
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    while True:
        nums = [random.randint(1, 10) for _ in range(num_count)]

        all_even = all(n % 2 == 0 for n in nums)
        if split == "train" and all_even:
            continue
        if split == "val" and not all_even:
            continue

        # Generate a solvable target using all numbers (left-associative)
        ops = ['+', '-', '*']
        chosen_ops = [random.choice(ops) for _ in range(num_count - 1)]
        expr = str(nums[0])
        for i, op in enumerate(chosen_ops):
            expr = f"({expr} {op} {nums[i+1]})"
        target = eval(expr)

        # Filter for reasonable positive targets; exclude trivial cases where target is already in nums
        if 1 <= target <= 100 and target not in nums:
            break

    random.shuffle(nums)
    answer_data = json.dumps({"target": target, "nums": nums})
    question = (
        f"Using the numbers {nums}, create an expression that equals {target}. "
        f"You must use all {num_count} numbers, each exactly once. Available operations: +, -, * (no division). "
        f"Show your reasoning in <think>...</think>, then write your final expression in <answer>...</answer>."
    )
    if tokenizer is not None:
        messages = [
            {"role": "system", "content": "You are a mathematical puzzle solver."},
        ]
        if oneshot:
            messages.append({"role": "user", "content": EXAMPLE_Q})
            messages.append({"role": "assistant", "content": EXAMPLE_A})
        messages.append({"role": "user", "content": question})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"User: {question}\nAssistant:"
    return text, answer_data


def make_prompt(problem, tokenizer, oneshot):
    """Build a chat-formatted prompt from a problem dict (for evaluation)."""
    num_count = len(problem["nums"])
    question = (
        f"Using the numbers {problem['nums']}, create an expression that equals {problem['target']}. "
        f"You must use all {num_count} numbers, each exactly once. Available operations: +, -, * (no division). "
        f"Show your reasoning in <think>...</think>, then write only the bare expression in <answer>...</answer>."
    )
    messages = [{"role": "system", "content": "You are a mathematical puzzle solver."}]
    if oneshot:
        messages.append({"role": "user", "content": EXAMPLE_Q})
        messages.append({"role": "assistant", "content": EXAMPLE_A})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_natural_prompt(problem, tokenizer):
    """Build a chat-formatted prompt with no format tags (for natural language eval)."""
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
