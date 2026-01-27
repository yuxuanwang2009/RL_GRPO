import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def main(math=True):
    parser = argparse.ArgumentParser(description="Generate math answers using trained GRPO model.")
    parser.add_argument("question", type=str, help="The math question to answer (e.g., 'What is 5 + 3?')")
    args = parser.parse_args()

    # Load model and tokenizer
    model_path = "saved_model"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Format prompt
    if math:
        prompt = f"User: {args.question} Answer in <answer>...</answer>. \nAssistant:"
    else:
        prompt = f"User: {args.question} \nAssistant:"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode and extract answer
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = generated[len(prompt):].strip()

    print(f"Question: {args.question}")
    print(f"Generated: {completion}")

if __name__ == "__main__":
    main(math=False)