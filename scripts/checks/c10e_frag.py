"""Check 10e: does removing fragmentation raise the GPU-only limit?
usage: c10e_frag.py <dynamic|static>   (run with/without PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
MLP chunking on. First, 4096-token correctness: static vs dynamic tokens must be identical. Then 10k..16k."""
import sys, os, time, gc, torch
sys.path.insert(0, os.path.dirname(__file__))
from kvsetup import load_model, MiB
from transformers import DynamicCache, StaticCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

kind = sys.argv[1]; NEW = 16
print("PYTORCH_CUDA_ALLOC_CONF =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "| cache:", kind)
model, tok, limit = load_model()
ORIG = Qwen3MLP.forward
Qwen3MLP.forward = lambda self, x: ORIG(self, x) if x.shape[1] <= 1024 else torch.cat([ORIG(self, x[:, i:i+1024]) for i in range(0, x.shape[1], 1024)], 1)
base = tok("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 4000, return_tensors="pt").input_ids[0]

def run(L, k):
    ids = base[:L].unsqueeze(0).cuda()
    cache = StaticCache(config=model.config, max_cache_len=L + NEW + 1) if k == "static" else DynamicCache(config=model.config)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        t = time.perf_counter(); out = model(input_ids=ids, past_key_values=cache, logits_to_keep=1); torch.cuda.synchronize(); tp = time.perf_counter() - t
        nxt = out.logits[:, -1:].argmax(-1); toks = []; t = time.perf_counter()
        for _ in range(NEW):
            toks.append(nxt.item()); out = model(input_ids=nxt, past_key_values=cache); nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize(); td = time.perf_counter() - t
    r = (torch.cuda.max_memory_allocated() / MiB, torch.cuda.max_memory_reserved() / MiB, tp, NEW / td, toks)
    del out, cache; gc.collect(); torch.cuda.empty_cache(); return r

if kind == "static":
    a = run(4096, "dynamic"); b = run(4096, "static")
    print(f"4096-token check: static tokens identical to dynamic: {a[4] == b[4]} | decode dynamic {a[3]:.2f} vs static {b[3]:.2f} tok/s")
for L in (10240, 12288, 14336, 16384):
    try:
        pa, pr, tp, dps, _ = run(L, kind)
        print(f"  {L:6d} tok: OK      peak allocated {pa:5.0f} MiB, peak reserved {pr:5.0f} MiB | prefill {tp:5.1f} s | decode {dps:.2f} tok/s", flush=True)
    except torch.OutOfMemoryError as e:
        print(f"  {L:6d} tok: FAILED  out of memory (limit {limit/MiB:.0f} MiB)", flush=True); break
    except Exception as e:
        print(f"  {L:6d} tok: FAILED  {type(e).__name__}: {str(e).splitlines()[0][:150]}", flush=True); break
