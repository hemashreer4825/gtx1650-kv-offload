"""Check 6b: where does the 4-bit model's GPU memory go?"""
import torch, subprocess, collections
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers.utils.logging as tl; tl.disable_progress_bar()

def smi():
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print("nvidia-smi before load:", smi())
M = "Qwen/Qwen3-1.7B"
model = AutoModelForCausalLM.from_pretrained(
    M, device_map="cuda", dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
print(f"after load : allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB, reserved {torch.cuda.memory_reserved()/2**20:.0f} MiB, nvidia-smi {smi()}")
torch.cuda.empty_cache()
print(f"after empty_cache: allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB, reserved {torch.cuda.memory_reserved()/2**20:.0f} MiB, nvidia-smi {smi()}")

emb, head = model.model.embed_tokens.weight, model.lm_head.weight
print(f"\nembed_tokens: {tuple(emb.shape)} {emb.dtype}   lm_head: {tuple(head.shape)} {head.dtype} {type(model.lm_head).__name__}")
print("lm_head shares memory with embed_tokens (tied)?", emb.data_ptr() == head.data_ptr())

by = collections.Counter(); seen = set()
for name, t in list(model.named_parameters()) + list(model.named_buffers()):
    if t.data_ptr() in seen: continue
    seen.add(t.data_ptr())
    kind = ("embed_tokens" if "embed_tokens" in name else "lm_head" if "lm_head" in name else
            "attn/mlp 4-bit weights" if t.dtype == torch.uint8 else f"other ({t.dtype})")
    by[kind] += t.numel() * t.element_size()
print("\nGPU memory by part:")
for k, v in by.most_common(): print(f"  {k:28s} {v/2**20:8.0f} MiB")
print(f"  {'TOTAL unique tensors':28s} {sum(by.values())/2**20:8.0f} MiB")
