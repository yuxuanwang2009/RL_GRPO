import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


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


def main():
    parser = argparse.ArgumentParser(description="Generate countdown answers using trained GRPO model.")
    parser.add_argument("question", type=str,
                        help="e.g. 'Using the numbers [4, 7, 3], create an expression that equals 25.'")
    parser.add_argument("--model_path", type=str, default="saved_model")
    parser.add_argument("--no_oneshot", action="store_true", help="Skip one-shot example")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    messages = [{"role": "system", "content": "You are a mathematical puzzle solver."}]
    if not args.no_oneshot:
        messages.append({"role": "user", "content": _EXAMPLE_Q})
        messages.append({"role": "assistant", "content": _EXAMPLE_A})
    messages.append({"role": "user", "content": args.question})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    completion = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    print(f"Question: {args.question}")
    print(f"Generated: {completion}")

if __name__ == "__main__":
    main()
