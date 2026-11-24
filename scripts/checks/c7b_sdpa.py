"""Check 7b: which SDPA attention kernels work on sm_75 for Qwen3 shapes (16 q-heads, 8 kv-heads, head_dim 128)?"""
import torch, inspect
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
L = 4096
q = torch.randn(1, 16, L, 128, device="cuda", dtype=torch.float16)
k = torch.randn(1, 8, L, 128, device="cuda", dtype=torch.float16)
v = torch.randn_like(k)
k2, v2 = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)   # GQA expanded to 16 heads
for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH):
    for gqa in (True, False):
        torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        try:
            with sdpa_kernel(be):
                o = F.scaled_dot_product_attention(q, k if gqa else k2, v if gqa else v2, is_causal=True, enable_gqa=gqa)
            torch.cuda.synchronize()
            print(f"{be.name:20s} enable_gqa={gqa!s:5s}: OK   extra peak {(torch.cuda.max_memory_allocated()-base)/2**20:6.0f} MiB")
        except Exception as e:
            print(f"{be.name:20s} enable_gqa={gqa!s:5s}: NOT AVAILABLE ({str(e).splitlines()[0][:90]})")

import transformers.integrations.sdpa_attention as sa
print("\n--- transformers sdpa_attention.py: when does it use enable_gqa?")
src = inspect.getsource(sa)
for i, line in enumerate(src.splitlines()):
    if any(s in line for s in ("def use_gqa_in_sdpa", "enable_gqa", "is_torch_greater", "attention_mask is None", "return ", "is_causal")):
        print(f"  {i+1:4d}: {line}")
