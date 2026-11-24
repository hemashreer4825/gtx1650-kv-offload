"""Check 6d: tie lm_head to embed_tokens (bit-identical in checkpoint) and prove outputs are unchanged."""
import torch, subprocess, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers.utils.logging as tl; tl.disable_progress_bar(); tl.set_verbosity_error()

def smi():
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
M = "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, device_map="cuda", dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
model.eval()

QA = [("What is the capital of France? Answer in one word.", "paris"),
      ("What is 17 multiplied by 23? Give only the number.", "391"),
      ("Which is the largest planet in our solar system? Answer in one word.", "jupiter"),
      ("Complete the sequence: 2, 4, 8, 16, ... Give only the next number.", "32")]
def run(tag):
    outs = []
    for q, exp in QA:
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.generate(**ids, max_new_tokens=80, do_sample=False)[0, ids.input_ids.shape[1]:].tolist()
        ans = tok.decode(o, skip_special_tokens=True).strip()
        print(f"  [{tag}] [{'PASS' if exp in ans.lower() else 'FAIL'}] {q}\n            -> {ans!r}")
        outs.append(o)
    return outs

torch.cuda.empty_cache()
print(f"UNTIED: allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB, nvidia-smi {smi()}")
a = run("untied")

model.lm_head.weight = model.model.embed_tokens.weight   # share the one table
gc.collect(); torch.cuda.empty_cache()
print(f"\nTIED  : allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB, nvidia-smi {smi()}")
print("lm_head shares memory with embed_tokens?", model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr())
b = run("tied")
print("\noutput tokens identical untied vs tied, per question:", [x == y for x, y in zip(a, b)])
