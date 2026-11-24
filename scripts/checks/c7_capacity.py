"""Check 7: maximum prompt length for (a) full GPU cache and (b) CPU-offloaded cache (DynamicCache(offloading=True)).
Qwen3-1.7B, NF4 weights, fp16 compute, lm_head tied to embed_tokens. Greedy, 16 new tokens. Stops a mode at first failure."""
import sys, time, gc, torch, subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
import transformers.utils.logging as tl; tl.disable_progress_bar(); tl.set_verbosity_error()
import transformers.integrations.sdpa_attention as sa
if "--gqa-fix" in sys.argv:   # check 7b/7c: repeat_kv so SDPA uses the memory-efficient kernel on sm_75
    sa.use_gqa_in_sdpa = lambda attention_mask, key: False
    sys.argv.remove("--gqa-fix"); print("attention fix: ON (repeat_kv -> memory-efficient SDPA kernel)")
print("GPU total (torch):", torch.cuda.get_device_properties(0).total_memory // 2**20, "MiB")

def mem():
    f = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.split("\n")
    m, s = f[1].split(), f[2].split()
    return f"RAM used {m[2]} MiB, avail {m[6]} MiB, swap used {s[2]} MiB"

M = "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, device_map="cuda", dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
model.lm_head.weight = model.model.embed_tokens.weight
model.eval(); gc.collect(); torch.cuda.empty_cache()
print(f"model loaded: allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB | {mem()}", flush=True)

filler = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 4000)
base_ids = tok(filler, return_tensors="pt").input_ids[0]
LENGTHS = [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768]
NEW = 16

for mode in sys.argv[1:]:
    print(f"\n===== mode: {mode}", flush=True)
    for L in LENGTHS:
        ids = base_ids[:L].unsqueeze(0).to("cuda")
        cache = DynamicCache(config=model.config, offloading=(mode == "offload"))
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                torch.cuda.synchronize(); t0 = time.perf_counter()
                out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                torch.cuda.synchronize(); t_pre = time.perf_counter() - t0
                nxt = out.logits[:, -1:].argmax(-1)
                t1 = time.perf_counter()
                for i in range(NEW - 1):
                    out = model(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True)
                    nxt = out.logits[:, -1:].argmax(-1)
                torch.cuda.synchronize(); t_dec = time.perf_counter() - t1
            kv_dev = cache.layers[0].keys.device
            print(f"  {L:6d} tok: OK   prefill {t_pre:6.1f} s ({L/t_pre:6.0f} tok/s) | decode {(NEW-1)/t_dec:6.2f} tok/s | "
                  f"peak GPU {torch.cuda.max_memory_allocated()/2**20:5.0f} MiB | cache on {kv_dev} | {mem()}", flush=True)
            del out, cache
        except Exception as e:
            print(f"  {L:6d} tok: FAILED  {type(e).__name__}: {str(e).splitlines()[0][:140]} | {mem()}", flush=True)
            del cache; gc.collect(); torch.cuda.empty_cache()
            break
