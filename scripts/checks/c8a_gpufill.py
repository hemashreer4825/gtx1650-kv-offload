"""Check 8a: is CUDA sysmem fallback really off? Fill the GPU in 64 MiB chunks; time reads of first vs last chunk."""
import torch, time, subprocess
MiB = 1 << 20
free0, total = torch.cuda.mem_get_info()
print(f"before: driver says free {free0/MiB:.0f} MiB of {total/MiB:.0f} MiB")
chunks = []
try:
    while True:
        chunks.append(torch.ones(64 * MiB // 2, dtype=torch.float16, device="cuda"))
        if len(chunks) * 64 > 8000: print("passed 8000 MiB - stopping"); break
except Exception as e:
    print("allocation stopped by:", type(e).__name__)
n = len(chunks); print(f"allocated {n*64} MiB in {n} chunks (physical card: {total/MiB:.0f} MiB, free at start {free0/MiB:.0f} MiB)")
def rd(t):
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(20): t.sum()
    torch.cuda.synchronize(); return 20 * 64 / 1024 / (time.perf_counter() - s)
print(f"read speed first chunk: {rd(chunks[0]):7.1f} GB/s   last chunk: {rd(chunks[-1]):7.1f} GB/s")
print("verdict:", "SPILLING INTO RAM (fallback ON)" if n * 64 > free0 / MiB + 64 else "no spill: allocations stayed within free VRAM (fallback OFF)")
