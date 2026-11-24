"""Check 7c: disabling enable_gqa (so SDPA can use the memory-efficient kernel on sm_75) must not change model outputs."""
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers.integrations.sdpa_attention as sa
import transformers.utils.logging as tl; tl.disable_progress_bar(); tl.set_verbosity_error()
ORIG = sa.use_gqa_in_sdpa

M = "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, device_map="cuda", dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
model.lm_head.weight = model.model.embed_tokens.weight
model.eval(); gc.collect(); torch.cuda.empty_cache()

QA = ["What is the capital of France? Answer in one word.", "What is 17 multiplied by 23? Give only the number.",
      "Which is the largest planet in our solar system? Answer in one word.", "Complete the sequence: 2, 4, 8, 16, ... Give only the next number."]
long_ids = tok(("The grass is green. The sky is blue. " * 300), return_tensors="pt").input_ids[:, :1000].cuda()

def run(tag):
    torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    with torch.no_grad():
        logits = model(input_ids=long_ids, logits_to_keep=1).logits[0, -1].float()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    answers = []
    for q in QA:
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            answers.append(model.generate(**ids, max_new_tokens=40, do_sample=False)[0, ids.input_ids.shape[1]:].tolist())
    print(f"[{tag}] 1000-token forward: extra peak GPU {peak:.0f} MiB")
    return logits, answers

sa.use_gqa_in_sdpa = ORIG
la, aa = run("original (enable_gqa -> math kernel)")
sa.use_gqa_in_sdpa = lambda attention_mask, key: False
lb, ab = run("fixed    (repeat_kv -> mem-efficient kernel)")

d = (la - lb).abs()
print(f"next-token logits: max abs diff {d.max().item():.4f}, mean abs diff {d.mean().item():.5f}, "
      f"top-1 token same: {la.argmax().item() == lb.argmax().item()}, top-5 same: {la.topk(5).indices.tolist() == lb.topk(5).indices.tolist()}")
print("4 answers token-identical:", [x == y for x, y in zip(aa, ab)])
for q, a in zip(QA, ab): print(f"   {q[:45]:45s} -> {tok.decode(a, skip_special_tokens=True).strip()[:60]!r}")
