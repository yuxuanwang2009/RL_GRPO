import ast
import json
import re


def safe_eval(expr_str):
    """Safely evaluate a mathematical expression containing only +, -, *."""
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


def extract_numbers(expr_str):
    """Extract all integer literals from an expression string."""
    return [int(x) for x in re.findall(r'\d+', expr_str)]


def _verify_numbers(expr, available):
    """Check that the expression uses exactly the available numbers, each once."""
    used = sorted(extract_numbers(expr))
    pool = available.copy()
    for n in used:
        if n in pool:
            pool.remove(n)
        else:
            return False
    return len(pool) == 0


def check_answer(completion, problem):
    """Check a tagged completion (<answer>...</answer>) against a problem dict."""
    target = problem["target"]
    available = sorted(problem["nums"])
    match = re.search(r'<answer>(.*?)</answer>', completion)
    if not match:
        return False
    expr = match.group(1).strip()
    result = safe_eval(expr)
    if result is None or result != target:
        return False
    return _verify_numbers(expr, available)


def extract_last_expression(completion):
    """Find the last valid math expression in a completion (for natural language eval)."""
    candidates = re.findall(r'[\d][\d\s\+\-\*\(\)]+[\d\)]', completion)
    for candidate in reversed(candidates):
        candidate = candidate.strip()
        if safe_eval(candidate) is not None:
            return candidate
    return None


def check_answer_natural(completion, problem):
    """Check a natural-language completion (no tags) against a problem dict."""
    target = problem["target"]
    available = sorted(problem["nums"])

    expr = extract_last_expression(completion)
    if expr is None:
        return False

    result = safe_eval(expr)
    if result is None or result != target:
        return False
    return _verify_numbers(expr, available)


def reward_function(completions, answers):
    """Compute rewards and accuracies for a batch of completions.

    Args:
        completions: list of completion strings
        answers: list of JSON strings with {"target": int, "nums": list}

    Returns:
        (rewards, accuracies) — both lists of floats
    """
    rewards = []
    accuracies = []

    for completion, answer_json in zip(completions, answers):
        r = 0.0
        acc = 0.0
        info = json.loads(answer_json)
        target = info["target"]
        available = sorted(info["nums"])

        # Format reward: +0.1 for <think> tag, +0.1 for <answer> tag
        has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
        has_answer = bool(re.search(r'<answer>.*?</answer>', completion, re.DOTALL))
        r += 0.1 * has_think
        r += 0.1 * has_answer

        if match := re.search(r'<answer>(.*?)</answer>', completion):
            expr = match.group(1).strip()
            result = safe_eval(expr)

            if result is not None and result == target:
                if _verify_numbers(expr, available):
                    r = 1.0  # Full reward overwrites format partial credit
                    acc = 1.0

        rewards.append(r)
        accuracies.append(acc)

    return rewards, accuracies
