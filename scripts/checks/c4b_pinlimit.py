"""Check 4b: cumulative pinned-memory limit, allocated in 64 MiB chunks (like a per-layer KV cache)."""
import torch, subprocess
MiB = 1 << 20
chunks = []
try:
    while True:
        t = torch.empty(64 * MiB, dtype=torch.uint8, pin_memory=True); t.fill_(1); chunks.append(t)
except Exception as e:
    err = f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"
total = len(chunks) * 64
print(f"pinned total before failure: {total} MiB = {total/1024:.2f} GiB   (stopped by: {err})")
print(subprocess.run(["free", "-m"], capture_output=True, text=True).stdout)
del chunks
