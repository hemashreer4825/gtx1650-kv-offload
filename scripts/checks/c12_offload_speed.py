"""Check 12: token-major (contiguous) offload layout - tokens still identical to GPU-only? decode speed at 4k/8k/16k?
Also the pure transfer bound: time to copy one full KV cache CPU->GPU at that length."""
import sys, os, time, gc, torch
sys.path.insert(0, os.path.dirname(__file__))
from kvsetup import load_model, ram, MiB
from kvoffload import PinnedOffloadCache
from transformers import DynamicCache

model, tok, limit = load_model()
base = tok("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 4000,
           return_tensors="pt").input_ids[0]
NEW = 16
def run(L, kind):
    ids = base[:L].unsqueeze(0).cuda(); gc.collect(); torch.cuda.empty_cache()
    cache = PinnedOffloadCache(model.config, L + NEW + 8) if kind == "offload" else DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=cache, logits_to_keep=1)
        lg = out.logits[0, -1].float().cpu(); nxt = out.logits[:, -1:].argmax(-1); toks = []
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(NEW):
            toks.append(nxt.item()); out = model(input_ids=nxt, past_key_values=cache); nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize(); td = time.perf_counter() - t
    if kind == "offload": cache.close()
    del cache, out; gc.collect(); torch.cuda.empty_cache()
    return toks, lg, NEW / td

def transfer_bound(L):
    n = L * 114688 // 2
    src = torch.empty(n, dtype=torch.float16).pin_memory(); dst = torch.empty(n, dtype=torch.float16, device="cuda")
    dst.copy_(src, non_blocking=True); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(3): dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize(); s = (time.perf_counter() - t) / 3
    del src, dst; gc.collect(); torch.cuda.empty_cache(); return s

g, gl, gs = run(4096, "gpu"); o, ol, os_ = run(4096, "offload")
print(f"4096 tok correctness: tokens identical to GPU-only: {g == o} | first-token logits max abs diff {(gl-ol).abs().max():.6f}")
print(f"\n{'tokens':>7} | GPU-only decode | offload decode | old layout (check 11) | pure KV transfer time per token")
old = {4096: 0.75, 8192: 1.09, 16384: 0.23}
for L in (4096, 8192, 16384):
    gs_ = run(L, "gpu")[2] if L <= 8192 else float("nan")
    os2 = os_ if L == 4096 else run(L, "offload")[2]
    tb = transfer_bound(L)
    print(f"{L:7d} | {gs_:9.2f} tok/s | {os2:8.2f} tok/s | {old[L]:8.2f} tok/s         | {tb:.3f} s  (=> at most {1/tb:.1f} tok/s from transfer alone)", flush=True)
print(ram())
