"""Check 6c: are lm_head.weight and embed_tokens.weight identical in the Qwen3-1.7B checkpoint?"""
import json, os, torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
p = os.path.expanduser("~/kv-offload/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
idx = json.load(open(os.path.join(p, "model.safetensors.index.json")))["weight_map"]
names = ["model.embed_tokens.weight", "lm_head.weight"]
t = {}
for n in names:
    f = idx[n]; print(f"{n:28s} stored in {f}")
    with safe_open(os.path.join(p, f), "pt") as h: t[n] = h.get_tensor(n)
a, b = t[names[0]], t[names[1]]
print("shapes:", tuple(a.shape), tuple(b.shape), " dtypes:", a.dtype, b.dtype)
print("bit-identical:", torch.equal(a, b))
if not torch.equal(a, b):
    d = (a.float() - b.float()).abs()
    print(f"differing elements: {(d > 0).sum().item()} of {d.numel()}   max abs diff {d.max().item():.3e}")
