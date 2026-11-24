"""Check 9b: maximum prompt length, KV cache on GPU vs offloaded to CPU RAM, with all fixes (kvsetup.py).
Greedy, 16 new tokens. On first failure, also tries 2048 tokens lower to narrow the limit."""
import sys, os, time, gc, torch
sys.path.insert(0, os.path.dirname(__file__))
from kvsetup import load_model, ram, swap_mib, MiB
from transformers import DynamicCache

model, tok, limit = load_model()
filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 4000
base_ids = tok(filler, return_tensors="pt").input_ids[0]
NEW = 16
swap0 = swap_mib()

def trial(L, offload):
    ids = base_ids[:L].unsqueeze(0).to("cuda")
    cache = DynamicCache(config=model.config, offloading=offload)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize(); tp = time.perf_counter() - t0
            nxt = out.logits[:, -1:].argmax(-1); t1 = time.perf_counter()
            for _ in range(NEW - 1):
                out = model(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True)
                nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize(); td = time.perf_counter() - t1
        sw = swap_mib() - swap0
        flag = "" if sw < 256 else f"  <-- INVALID: swap grew {sw} MiB"
        print(f"  {L:6d} tok: OK      prefill {tp:6.1f} s ({L/tp:5.0f} tok/s) | decode {(NEW-1)/td:5.2f} tok/s | "
              f"peak GPU {torch.cuda.max_memory_allocated()/MiB:5.0f} MiB | {ram()}{flag}", flush=True)
        del out, cache; return sw < 256
    except torch.OutOfMemoryError:
        print(f"  {L:6d} tok: FAILED  GPU out of memory (limit {limit/MiB:.0f} MiB) | {ram()}", flush=True)
    except Exception as e:
        print(f"  {L:6d} tok: FAILED  {type(e).__name__}: {str(e).splitlines()[0][:120]} | {ram()}", flush=True)
    del cache; gc.collect(); torch.cuda.empty_cache(); return False

for mode in ("gpu", "offload"):
    print(f"\n===== KV cache: {'on GPU' if mode == 'gpu' else 'offloaded to CPU RAM'}", flush=True)
    for L in range(4096, 32768 + 1, 4096):
        if not trial(L, mode == "offload"):
            trial(L - 2048, mode == "offload")
            break
