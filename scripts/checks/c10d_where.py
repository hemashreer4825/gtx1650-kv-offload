"""Check 10d: during prefill (with MLP chunking), which step inside a layer needs the most temporary GPU memory?
Measures, for layer 0 and layer 27 at L tokens, the peak memory above the starting point for each sub-module.
Also reports KV-cache dtype and allocated vs reserved memory after prefill (fragmentation)."""
import sys, os, gc, torch
sys.path.insert(0, os.path.dirname(__file__))
from kvsetup import load_model, MiB
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

L = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
print("PYTORCH_CUDA_ALLOC_CONF =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
model, tok, limit = load_model()
ORIG = Qwen3MLP.forward
Qwen3MLP.forward = lambda self, x: ORIG(self, x) if x.shape[1] <= 1024 else torch.cat([ORIG(self, x[:, i:i+1024]) for i in range(0, x.shape[1], 1024)], 1)

rows = []
def pre(name):
    def f(m, a, kw=None):
        torch.cuda.synchronize(); m._start = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    return f
def post(name):
    def f(m, a, o, kw=None):
        torch.cuda.synchronize()
        rows.append((name, (torch.cuda.max_memory_allocated() - m._start) / MiB, (torch.cuda.memory_allocated() - m._start) / MiB))
    return f
for li in (0, 27):
    layer = model.model.layers[li]
    for n, m in [("input_layernorm", layer.input_layernorm), ("self_attn", layer.self_attn),
                 ("post_attention_layernorm", layer.post_attention_layernorm), ("mlp", layer.mlp)]:
        m.register_forward_pre_hook(pre(n)); m.register_forward_hook(post(f"layer{li}.{n}"))

ids = tok("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 4000,
          return_tensors="pt").input_ids[:, :L].cuda()
cache = DynamicCache(config=model.config)
gc.collect(); torch.cuda.empty_cache()
start = torch.cuda.memory_allocated()
try:
    with torch.no_grad():
        model.model(input_ids=ids, past_key_values=cache)
    print(f"prefill {L} tok: OK")
except torch.OutOfMemoryError as e:
    print(f"prefill {L} tok: OUT OF MEMORY: {str(e).splitlines()[0][:200]}")
print(f"\npeak temporary memory per step (MiB above memory at step start) | memory kept after step:")
for n, pk, kept in rows: print(f"  {n:34s} peak +{pk:7.0f}   kept +{kept:6.0f}")
if cache.layers and cache.layers[0].keys is not None:
    print(f"\nKV cache dtype: {cache.layers[0].keys.dtype}, tokens {cache.get_seq_length()}, "
          f"expected size {cache.get_seq_length()*114688/MiB:.0f} MiB")
print(f"after prefill: allocated {torch.cuda.memory_allocated()/MiB:.0f} MiB, reserved {torch.cuda.memory_reserved()/MiB:.0f} MiB "
      f"(gap = memory held but unused: {(torch.cuda.memory_reserved()-torch.cuda.memory_allocated())/MiB:.0f} MiB)")
