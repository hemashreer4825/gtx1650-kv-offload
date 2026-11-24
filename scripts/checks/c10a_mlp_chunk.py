"""Check 10a (Fix A): process the MLP in 1024-token pieces during prefill. MLP is per-token, so the maths is unchanged.
(1) 6144 tokens without vs with fix: peak GPU memory, last-token logits, 16 greedy tokens must match.
(2) with fix: increase length until the GPU (3040 MiB PyTorch limit) is really full."""
import sys, os, time, gc, torch
sys.path.insert(0, os.path.dirname(__file__))
from kvsetup import load_model, MiB
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

model, tok, limit = load_model()
base_ids = tok("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 4000,
               return_tensors="pt").input_ids[0]
ORIG_FWD = Qwen3MLP.forward
CH = 1024
def chunked_fwd(self, x):
    if x.shape[1] <= CH:
        return ORIG_FWD(self, x)
    return torch.cat([ORIG_FWD(self, x[:, i:i + CH]) for i in range(0, x.shape[1], CH)], dim=1)

def run(L, new=16):
    ids = base_ids[:L].unsqueeze(0).cuda()
    cache = DynamicCache(config=model.config)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        t = time.perf_counter()
        out = model(input_ids=ids, past_key_values=cache, logits_to_keep=1); torch.cuda.synchronize()
        tp = time.perf_counter() - t
        logits = out.logits[0, -1].float().cpu(); toks = []
        nxt = out.logits[:, -1:].argmax(-1); t = time.perf_counter()
        for _ in range(new):
            toks.append(nxt.item())
            out = model(input_ids=nxt, past_key_values=out.past_key_values); nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize(); td = time.perf_counter() - t
    peak = torch.cuda.max_memory_allocated() / MiB
    del out, cache; gc.collect(); torch.cuda.empty_cache()
    return peak, tp, new / td, logits, toks

print("\n(1) same 6144-token input, without vs with MLP chunking")
Qwen3MLP.forward = ORIG_FWD; p0, tp0, d0, l0, t0 = run(6144)
Qwen3MLP.forward = chunked_fwd; p1, tp1, d1, l1, t1 = run(6144)
print(f"  without fix: peak GPU {p0:5.0f} MiB | prefill {tp0:5.1f} s | decode {d0:.2f} tok/s")
print(f"  with fix   : peak GPU {p1:5.0f} MiB | prefill {tp1:5.1f} s | decode {d1:.2f} tok/s")
d = (l0 - l1).abs()
print(f"  last-token logits: max abs diff {d.max():.4f}, top-1 same {l0.argmax().item() == l1.argmax().item()}")
print(f"  16 generated tokens identical: {t0 == t1}")

print("\n(2) with fix: how long can the KV cache on GPU get?")
for L in range(8192, 32768 + 1, 2048):
    try:
        p, tp, dps, _, _ = run(L)
        print(f"  {L:6d} tok: OK      peak GPU {p:5.0f} MiB | prefill {tp:6.1f} s ({L/tp:4.0f} tok/s) | decode {dps:.2f} tok/s", flush=True)
    except torch.OutOfMemoryError:
        print(f"  {L:6d} tok: FAILED  GPU out of memory (PyTorch limit {limit/MiB:.0f} MiB)", flush=True)
        break
