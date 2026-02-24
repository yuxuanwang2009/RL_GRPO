import random


def generate_problems(n, seed, num_count=3):
    """Generate a fixed set of countdown problems for evaluation.

    Uses the given seed for reproducibility.
    """
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
