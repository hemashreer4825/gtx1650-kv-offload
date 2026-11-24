"""Check 10c (Fix B): why does offloading use so much RAM, and which variant fixes it?
usage: c10c_offload_ram.py <dynamic|dynamic_empty|static> [L]
Runs GPU-only reference first (same process), then the offload variant; same input, 16 greedy tokens.
Reports pinned host memory held by PyTorch, system RAM, decode speed, and whether tokens match GPU-only exactly."""
import sys, os, time, gc, torch
sys.path.insert(0, os.path.dirname(__file__))
from kvsetup import load_model, ram, MiB
from transformers import DynamicCache, StaticCache

variant = sys.argv[1]; L = int(sys.argv[2]) if len(sys.argv) > 2 else 4096; NEW = 16
model, tok, limit = load_model()
ids = tok("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 4000,
          return_tensors="pt").input_ids[:, :L].cuda()

def host_mib():
    s = torch.cuda.host_memory_stats()
    a = s.get("allocated_bytes.current", s.get("allocated_bytes", {}).get("current", 0) if isinstance(s.get("allocated_bytes"), dict) else 0)
    r = s.get("reserved_bytes.current", s.get("reserved_bytes", {}).get("current", 0) if isinstance(s.get("reserved_bytes"), dict) else 0)
    return a / MiB, r / MiB

def gen(cache, empty=False):
    gc.collect(); torch.cuda.empty_cache()
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=cache, logits_to_keep=1)
        if empty: torch._C._host_emptyCache()
        after_prefill = (host_mib(), ram())
        nxt = out.logits[:, -1:].argmax(-1); toks = []; torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(NEW):
            toks.append(nxt.item())
            out = model(input_ids=nxt, past_key_values=cache); nxt = out.logits[:, -1:].argmax(-1)
            if empty: torch._C._host_emptyCache()
        torch.cuda.synchronize(); td = time.perf_counter() - t
    return toks, NEW / td, after_prefill, (host_mib(), ram())

print(f"KV cache size for {L} tokens (fp16): {L*114688/MiB:.0f} MiB")
print("host_memory_stats keys:", sorted(k for k in torch.cuda.host_memory_stats() if "bytes" in k)[:8])
ref, ref_tps, _, _ = gen(DynamicCache(config=model.config))
print(f"GPU-only reference: decode {ref_tps:.2f} tok/s | {ram()}")
torch._C._host_emptyCache(); gc.collect()
print(f"before offload run: pinned held {host_mib()} MiB (allocated, reserved) | {ram()}")

if variant == "static":
    cache = StaticCache(config=model.config, max_cache_len=L + NEW + 1, offloading=True)
else:
    cache = DynamicCache(config=model.config, offloading=True)
toks, tps, (hp, rp), (hd, rd) = gen(cache, empty=(variant == "dynamic_empty"))
print(f"[{variant}] after prefill : pinned (allocated, reserved) = ({hp[0]:.0f}, {hp[1]:.0f}) MiB | {rp}")
print(f"[{variant}] after decode  : pinned (allocated, reserved) = ({hd[0]:.0f}, {hd[1]:.0f}) MiB | {rd}")
print(f"[{variant}] decode {tps:.2f} tok/s (GPU-only {ref_tps:.2f}) | tokens identical to GPU-only: {toks == ref}")
