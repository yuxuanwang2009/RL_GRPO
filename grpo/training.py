import torch
import torch.nn.functional as F

from grpo.validation import reward_function


def get_batch_logprobs(model, input_ids, attention_mask, prompt_len):
    """Compute log probs of generated tokens.

    Uses F.cross_entropy in a fused kernel to avoid the full
    [batch, T, vocab_size] log_softmax allocation.
    Caller is responsible for batching (e.g. via gradient accumulation).
    """
    logits = model(input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    chosen_log_probs = -F.cross_entropy(
        logits.transpose(1, 2),
        labels,
        reduction='none'
    )
    result = chosen_log_probs[:, prompt_len-1:]
    del logits, labels, chosen_log_probs
    return result


def evaluate_accuracy(model, tokenizer, prompts, answers, device, max_new_tokens):
    """Run greedy inference on prompts and compute accuracy."""
    tokenizer.padding_side = 'left'
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    prompt_len = inputs.input_ids.shape[1]

    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    with torch.no_grad():
        sequences = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    model.config.use_cache = False
    model.train()
    model.gradient_checkpointing_enable()

    completions_text = tokenizer.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True)
    _, accuracy_list = reward_function(completions_text, answers)
    return sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0.0
