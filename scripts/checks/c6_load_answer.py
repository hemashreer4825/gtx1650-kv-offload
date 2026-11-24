"""Check 6: load Qwen3-1.7B in 4-bit (bitsandbytes NF4, fp16 compute) and check text correctness."""
import time, torch, subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

M = "Qwen/Qwen3-1.7B"
torch.cuda.reset_peak_memory_stats()
base = torch.cuda.memory_allocated()
t = time.perf_counter()
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, device_map="cuda", dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.float16))
model.eval()
print(f"load time: {time.perf_counter()-t:.1f} s")
print(f"weights on GPU (torch allocated): {(torch.cuda.memory_allocated()-base)/2**20:.0f} MiB")
print(f"attention implementation: {model.config._attn_implementation}")
print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip(), "(nvidia-smi: used, total)")

QA = [("What is the capital of France? Answer in one word.", "paris"),
      ("What is 17 multiplied by 23? Give only the number.", "391"),
      ("Which is the largest planet in our solar system? Answer in one word.", "jupiter"),
      ("Complete the sequence: 2, 4, 8, 16, ... Give only the next number.", "32")]
ok = 0
for q, exp in QA:
    text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=30, do_sample=False)
    ans = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
    good = exp in ans.lower(); ok += good
    print(f"[{'PASS' if good else 'FAIL'}] Q: {q}\n        A: {ans!r}  (expected '{exp}')")
print(f"\ncorrect: {ok}/{len(QA)}")
print(f"peak torch GPU memory during check: {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
