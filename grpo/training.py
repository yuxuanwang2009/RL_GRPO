import torch
import torch.nn.functional as F

from grpo.validation import reward_function


def get_batch_logprobs(model, input_ids, attention_mask, prompt_len, chunk_size=4):
    """Compute log probs of generated tokens in chunks to avoid OOM.

    Processes sequences in chunks of chunk_size, computing log probs only for
    tokens after prompt_len. Uses F.cross_entropy in a fused kernel to avoid
    the full [chunk, T, vocab_size] log_softmax allocation.
    """
    all_logprobs = []
    for i in range(0, input_ids.shape[0], chunk_size):
        chunk_ids = input_ids[i:i+chunk_size]
        chunk_mask = attention_mask[i:i+chunk_size]
        logits = model(chunk_ids, attention_mask=chunk_mask).logits[:, :-1, :]
        labels = chunk_ids[:, 1:]                       # [chunk, T-1]
        # F.cross_entropy computes -log_softmax(logits)[label] in a fused kernel,
        # avoiding the full [chunk, T, 152K] log_softmax allocation.
        # Negating gives us the log prob of each generated token.
        chosen_log_probs = -F.cross_entropy(
            logits.transpose(1, 2),                     # [chunk, vocab, T-1]
            labels,                                     # [chunk, T-1]
            reduction='none'                            # [chunk, T-1]
        )
        all_logprobs.append(chosen_log_probs[:, prompt_len-1:])
        del logits, labels, chosen_log_probs
    return torch.cat(all_logprobs, dim=0)


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
