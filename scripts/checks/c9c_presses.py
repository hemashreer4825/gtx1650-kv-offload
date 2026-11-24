"""Check 9c: do StreamingLLM and SnapKV (kvpress) run correctly on Qwen3-1.7B with our setup?
~4k-token context, 3 facts at 10% / 50% / 90% depth, 3 questions. Measures tokens actually kept in the KV cache."""
import sys, os, time, gc, torch
sys.path.insert(0, os.path.dirname(__file__))
from kvsetup import load_model, MiB
from transformers import pipeline, DynamicCache
from kvpress import StreamingLLMPress, SnapKVPress, KeyRerotationPress

model, tok, limit = load_model()
pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tok)

FACTS = [("The secret code for the blue door is 7419.", "What is the secret code for the blue door?", "7419"),
         ("The name of the captain's cat is Biscuit.", "What is the name of the captain's cat?", "biscuit"),
         ("The treasure is buried under the old oak tree.", "Where is the treasure buried?", "oak")]
filler = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ").split(". ")
sent = [s.strip() + "." for s in filler if s.strip()] * 400
n = 0; parts = []
target = 4000
for i, s in enumerate(sent):
    parts.append(s)
    if len(tok(" ".join(parts)).input_ids) > target: break
for frac, (fact, _, _) in zip((0.9, 0.5, 0.1), reversed(FACTS)):   # insert from the back so indices stay valid
    parts.insert(int(len(parts) * frac), fact)
context = " ".join(parts)
questions = [q + " Answer in a few words." for _, q, _ in FACTS]
print(f"context: {len(tok(context).input_ids)} tokens; facts at 10% / 50% / 90% depth")

SETUPS = [("no press (full KV cache)", None),
          ("StreamingLLM keep 50%", StreamingLLMPress(compression_ratio=0.5)),
          ("StreamingLLM+rerotation keep 50%", KeyRerotationPress(press=StreamingLLMPress(compression_ratio=0.5))),
          ("SnapKV keep 50%", SnapKVPress(compression_ratio=0.5)),
          ("SnapKV keep 25%", SnapKVPress(compression_ratio=0.75))]
for name, press in SETUPS:
    cache = DynamicCache()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        t = time.perf_counter()
        out = pipe(context, questions=questions, press=press, cache=cache, max_new_tokens=24)
        dt = time.perf_counter() - t
        kept = cache.get_seq_length()
        res = ["OK " if exp in a.lower() else "MISS" for a, (_, _, exp) in zip(out["answers"], FACTS)]
        print(f"\n[{name}] ran OK in {dt:.1f} s | KV tokens kept after compression: {kept} | peak GPU {torch.cuda.max_memory_allocated()/MiB:.0f} MiB")
        for pos, r, a in zip(("10% (start) ", "50% (middle)", "90% (end)   "), res, out["answers"]):
            print(f"    fact at {pos}: {r}  answer: {a.strip()[:70]!r}")
    except Exception as e:
        print(f"\n[{name}] FAILED: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    del cache
