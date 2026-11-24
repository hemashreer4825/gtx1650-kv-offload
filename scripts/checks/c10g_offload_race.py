"""Check 10g: why do offloaded-cache outputs differ from GPU-only? Hypothesis: race between the non-blocking
GPU->CPU offload copy and the next prefetch (no synchronisation between them).
Test at 4096 tokens: GPU-only vs offload (as-is) vs offload with a full synchronize after every offload/prefetch.
If the synchronised version matches GPU-only exactly, the cause is a timing (race) bug."""
import sys, os, gc, torch
sys.path.insert(0, os.path.dirname(__file__))
from kvsetup import load_model
from transformers import DynamicCache
from transformers import cache_utils as cu

model, tok, limit = load_model()
ids = tok("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 4000,
          return_tensors="pt").input_ids[:, :4096].cuda()
NEW = 16
def gen(cache):
    gc.collect(); torch.cuda.empty_cache()
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=cache, logits_to_keep=1)
        first_logits = out.logits[0, -1].float().cpu()
        nxt = out.logits[:, -1:].argmax(-1); toks = []
        for _ in range(NEW):
            toks.append(nxt.item()); out = model(input_ids=nxt, past_key_values=cache); nxt = out.logits[:, -1:].argmax(-1)
    return toks, first_logits

def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)

ref, rl = gen(DynamicCache(config=model.config))
asis, al = gen(DynamicCache(config=model.config, offloading=True))

orig_off, orig_pre = cu.CacheLayerMixin.offload, cu.CacheLayerMixin.prefetch
def off_sync(self): orig_off(self); torch.cuda.synchronize()
def pre_sync(self): orig_pre(self); torch.cuda.synchronize()
cu.CacheLayerMixin.offload, cu.CacheLayerMixin.prefetch = off_sync, pre_sync
synced, sl = gen(DynamicCache(config=model.config, offloading=True))
cu.CacheLayerMixin.offload, cu.CacheLayerMixin.prefetch = orig_off, orig_pre

print(f"GPU-only tokens        : {ref}")
print(f"offload as-is          : {asis}")
print(f"offload + synchronize  : {synced}")
print(f"prefill (1st token) logits max diff vs GPU-only: as-is {(al-rl).abs().max():.4f}, synced {(sl-rl).abs().max():.4f}")
print(f"as-is  identical: {asis == ref}   first differing generated token index: {first_diff(asis, ref)}")
print(f"synced identical: {synced == ref}   first differing generated token index: {first_diff(synced, ref)}")
print("decoded GPU-only :", repr(tok.decode(ref))); print("decoded as-is    :", repr(tok.decode(asis)))
