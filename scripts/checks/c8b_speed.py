"""Check 8b: is FP16 compute the reason for slow speed on GTX 1650? Compare FP16 vs FP32 compute.
(a) raw GEMM with the model's MLP shape; (b) model prefill 2048 tok + 32-token decode, bnb compute fp16 vs fp32; (c) answers in fp32."""
import sys, time, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
import transformers.integrations.sdpa_attention as sa
import transformers.utils.logging as tl; tl.disable_progress_bar(); tl.set_verbosity_error()
sa.use_gqa_in_sdpa = lambda attention_mask, key: False   # check 7c fix

print("(a) raw GEMM 2048x2048 @ 2048x6144 (one MLP projection over 2048 tokens):")
for dt in (torch.float16, torch.float32):
    a = torch.randn(2048, 2048, device="cuda", dtype=dt); b = torch.randn(2048, 6144, device="cuda", dtype=dt)
    for _ in range(3): a @ b
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10): a @ b
    torch.cuda.synchronize(); ms = (time.perf_counter() - t) / 10 * 1000
    print(f"    {str(dt):14s} {ms:7.2f} ms  ({2*2048*2048*6144/ms/1e9:6.2f} TFLOPS)")
    del a, b
torch.cuda.empty_cache()

M = "Qwen/Qwen3-1.7B"; tok = AutoTokenizer.from_pretrained(M)
ids = tok(("The grass is green. The sky is blue. The sun is yellow. Here we go. " * 600), return_tensors="pt").input_ids[:, :2048].cuda()
QA = [("What is the capital of France? Answer in one word.", "paris"), ("What is 17 multiplied by 23? Give only the number.", "391"),
      ("Which is the largest planet in our solar system? Answer in one word.", "jupiter")]
print("(b)/(c) model, NF4 weights:")
for cdt in (torch.float16, torch.float32):
    model = AutoModelForCausalLM.from_pretrained(M, device_map="cuda", dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=cdt))
    model.lm_head.weight = model.model.embed_tokens.weight; model.eval(); gc.collect(); torch.cuda.empty_cache()
    with torch.no_grad():
        model(input_ids=ids[:, :128])  # warm-up
        cache = DynamicCache(config=model.config)
        torch.cuda.synchronize(); t = time.perf_counter()
        out = model(input_ids=ids, past_key_values=cache, logits_to_keep=1)
        torch.cuda.synchronize(); tp = time.perf_counter() - t
        nxt = out.logits[:, -1:].argmax(-1); t = time.perf_counter()
        for _ in range(32):
            out = model(input_ids=nxt, past_key_values=out.past_key_values); nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize(); td = time.perf_counter() - t
        ok = 0
        for q, e in QA:
            x = tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="pt").to("cuda")
            a = tok.decode(model.generate(**x, max_new_tokens=30, do_sample=False)[0, x.input_ids.shape[1]:], skip_special_tokens=True)
            ok += e in a.lower()
    print(f"    compute {str(cdt):14s}: prefill 2048 tok {tp:6.1f} s ({2048/tp:6.0f} tok/s) | decode {32/td:6.2f} tok/s | answers correct {ok}/3")
    del model, out, cache; gc.collect(); torch.cuda.empty_cache()
