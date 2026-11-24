"""Check 4: CPU<->GPU transfer behaviour under WSL2.
(a) Does tensor.to('cpu', non_blocking=True) -- what transformers' offloading uses -- give pinned memory?
(b) Copy bandwidth host->device and device->host, pageable vs pinned, 128 MiB blocks.
(c) How much pinned memory can be allocated (1, 2, 3 GiB, freed after each)."""
import torch, time, gc

dev = "cuda"
MiB = 1 << 20
n = 128 * MiB // 2  # fp16 elements in 128 MiB

g = torch.randn(n, device=dev, dtype=torch.float16)
c = g.to("cpu", non_blocking=True); torch.cuda.synchronize()
print(f"(a) .to('cpu', non_blocking=True) result pinned? {c.is_pinned()}")
c2 = g.to("cpu"); print(f"    .to('cpu') (blocking) result pinned? {c2.is_pinned()}")
del c, c2

def bw(src, dst, reps=10):
    for _ in range(2): dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps): dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    return reps * src.numel() * 2 / (time.perf_counter() - t) / 1e9

pageable = torch.empty(n, dtype=torch.float16)
pinned = torch.empty(n, dtype=torch.float16, pin_memory=True)
print("(b) bandwidth, 128 MiB copies, GB/s:")
print(f"    CPU->GPU pageable: {bw(pageable, g):6.2f}")
print(f"    CPU->GPU pinned  : {bw(pinned, g):6.2f}")
print(f"    GPU->CPU pageable: {bw(g, pageable):6.2f}")
print(f"    GPU->CPU pinned  : {bw(g, pinned):6.2f}")
del pageable, pinned; gc.collect()

print("(c) pinned allocation test:")
for gib in (1, 2, 3):
    try:
        t0 = time.perf_counter()
        p = torch.empty(gib * 1024 * MiB, dtype=torch.uint8, pin_memory=True)
        p.fill_(1)
        print(f"    {gib} GiB pinned: OK  ({time.perf_counter()-t0:.1f}s)")
        del p; gc.collect()
    except Exception as e:
        print(f"    {gib} GiB pinned: FAILED  {type(e).__name__}: {str(e)[:150]}")
        break
