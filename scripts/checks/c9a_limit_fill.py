"""Check 9a: with the PyTorch hard limit on, does filling the GPU stop at the real free VRAM (no spill)?"""
import sys, os, time, torch
sys.path.insert(0, os.path.dirname(__file__))
from kvsetup import set_vram_limit, MiB
free0, total = torch.cuda.mem_get_info()
limit = set_vram_limit()
chunks = []
try:
    while True:
        chunks.append(torch.ones(64 * MiB // 2, dtype=torch.float16, device="cuda"))
except torch.OutOfMemoryError as e:
    print("allocation stopped by: OutOfMemoryError (PyTorch limit)")
got = len(chunks) * 64
free_end, _ = torch.cuda.mem_get_info()
print(f"allocated {got} MiB | limit {limit/MiB:.0f} MiB | free VRAM at start {free0/MiB:.0f} MiB | driver free now {free_end/MiB:.0f} MiB")
def rd(t):
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(20): t.sum()
    torch.cuda.synchronize(); return 20 * 64 / 1024 / (time.perf_counter() - s)
print(f"read speed first chunk {rd(chunks[0]):.1f} GB/s, last chunk {rd(chunks[-1]):.1f} GB/s")
print("verdict:", "PASS - stayed inside real VRAM, no spill" if got * MiB <= free0 and free_end > 0 else "FAIL - spilled beyond free VRAM")
