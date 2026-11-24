"""Check 11: own offloaded KV cache (kvoffload.py) - correctness, RAM use, max length; plus final GPU-only limit.
All fixes from kvsetup.py on (VRAM limit, attention kernel, FP32 compute, tied lm_head, MLP pieces; expandable_segments)."""
import sys, os, time, gc, torch
sys.path.insert(0, os.path.dirname(__file__))
from kvsetup import load_model, ram, swap_mib, MiB
from kvoffload import PinnedOffloadCache
from transformers import DynamicCache

model, tok, limit = load_model()
base = tok("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 4000,
           return_tensors="pt").input_ids[0]
NEW = 16
swap0 = swap_mib()

def avail():
    import subprocess
    return int(subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.split("\n")[1].split()[6])

def run(L, kind):
    ids = base[:L].unsqueeze(0).cuda()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    a0 = avail()
    cache = PinnedOffloadCache(model.config, L + NEW + 8) if kind == "offload" else DynamicCache(config=model.config)
    try:
        with torch.no_grad():
            torch.cuda.synchronize(); t = time.perf_counter()
            out = model(input_ids=ids, past_key_values=cache, logits_to_keep=1); torch.cuda.synchronize(); tp = time.perf_counter() - t
            logits = out.logits[0, -1].float().cpu(); nxt = out.logits[:, -1:].argmax(-1); toks = []; t = time.perf_counter()
            for _ in range(NEW):
                toks.append(nxt.item()); out = model(input_ids=nxt, past_key_values=cache); nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize(); td = time.perf_counter() - t
        used = a0 - avail()
        return dict(ok=True, tp=tp, dps=NEW / td, peak=torch.cuda.max_memory_allocated() / MiB, ram_used=used,
                    logits=logits, toks=toks, swap=swap_mib() - swap0)
    except torch.OutOfMemoryError:
        return dict(ok=False, why=f"GPU out of memory (limit {limit/MiB:.0f} MiB)")
    except Exception as e:
        return dict(ok=False, why=f"{type(e).__name__}: {str(e).splitlines()[0][:150]}")
    finally:
        if kind == "offload": cache.close()
        del cache; gc.collect(); torch.cuda.empty_cache()

def line(L, r):
    if not r["ok"]: return f"  {L:6d} tok: FAILED  {r['why']} | {ram()}"
    flag = "" if r["swap"] < 256 else f"  <-- INVALID: swap grew {r['swap']} MiB"
    return (f"  {L:6d} tok: OK      prefill {r['tp']:6.1f} s ({L/r['tp']:4.0f} tok/s) | decode {r['dps']:5.2f} tok/s | "
            f"peak GPU {r['peak']:5.0f} MiB | RAM taken {r['ram_used']:5d} MiB (KV size {L*114688/MiB:.0f}) | {ram()}{flag}")

print("\n(1) correctness at 4096 tokens: own offloaded cache vs GPU-only")
g = run(4096, "gpu"); o = run(4096, "offload")
print("  GPU-only :", line(4096, g)); print("  offloaded:", line(4096, o))
if g["ok"] and o["ok"]:
    print(f"  16 generated tokens identical: {g['toks'] == o['toks']} | first-token logits max abs diff {(g['logits']-o['logits']).abs().max():.6f}")
    print(f"  GPU-only text : {tok.decode(g['toks'])!r}\n  offloaded text: {tok.decode(o['toks'])!r}")

print("\n(2) own offloaded cache: maximum length")
for L in range(8192, 32768 + 1, 4096):
    r = run(L, "offload"); print(line(L, r), flush=True)
    if not r["ok"] or r["swap"] >= 256: break

print("\n(3) GPU-only with all fixes: final limit")
for L in (10240, 11264, 12288):
    r = run(L, "gpu"); print(line(L, r), flush=True)
    if not r["ok"]: break
