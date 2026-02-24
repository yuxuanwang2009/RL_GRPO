"""Quick memory profiling to find the OOM bottleneck."""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def mem_mb():
    return torch.cuda.memory_allocated() / 1024**2

def report(label):
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"  {label}: allocated={alloc:.2f} GB, reserved={reserved:.2f} GB")

print("=== Memory Profiling ===\n")

# 1. Baseline
torch.cuda.reset_peak_memory_stats()
report("Empty GPU")

# 2. Load policy
policy = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.bfloat16, trust_remote_code=True
).to("cuda")
policy.config.use_cache = False
policy.gradient_checkpointing_enable()
policy.train()
report("After loading policy")

# 3. Fake input (simulating 64 sequences of length 460)
B, T, V = 64, 460, policy.config.vocab_size
fake_ids = torch.randint(0, V, (B, T), device="cuda")
fake_mask = torch.ones(B, T, dtype=torch.long, device="cuda")
report("After creating fake input")

# 4. Single chunk forward WITHOUT gradients
chunk = 8
with torch.no_grad():
    logits = policy(fake_ids[:chunk], attention_mask=fake_mask[:chunk]).logits
    report(f"After no_grad forward (chunk={chunk})")
    print(f"    Logits tensor: {logits.shape}, {logits.element_size() * logits.nelement() / 1024**3:.2f} GB")
    del logits
    torch.cuda.empty_cache()
    report("After cleanup")

# 5. Single chunk forward WITH gradients
logits = policy(fake_ids[:chunk], attention_mask=fake_mask[:chunk]).logits
report(f"After grad forward 1 chunk ({chunk} seqs)")
logits_size = logits.element_size() * logits.nelement() / 1024**3
print(f"    Logits tensor: {logits.shape}, {logits_size:.2f} GB")
# Compute loss and backward to see gradient memory
labels = fake_ids[:chunk, 1:]
loss = F.cross_entropy(logits[:, :-1].transpose(1, 2), labels)
report("After computing loss (graph still alive)")
loss.backward()
report("After backward (graph freed, grads allocated)")
del logits, labels, loss
torch.cuda.empty_cache()
report("After cleanup")

# 6. Accumulate ALL chunks with gradients (simulating current code)
policy.zero_grad(set_to_none=True)
torch.cuda.empty_cache()
report("Before accumulated forward")

all_logprobs = []
for i in range(0, B, chunk):
    chunk_ids = fake_ids[i:i+chunk]
    chunk_mask = fake_mask[i:i+chunk]
    logits = policy(chunk_ids, attention_mask=chunk_mask).logits[:, :-1, :]
    lp = -F.cross_entropy(logits.transpose(1, 2), chunk_ids[:, 1:], reduction='none')
    all_logprobs.append(lp)
    del logits
    n_chunks_done = (i // chunk) + 1
    report(f"After accumulating chunk {n_chunks_done}/{B//chunk} (graph held)")

print(f"\n  Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print(f"  GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
