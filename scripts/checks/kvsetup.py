"""Shared setup for experiment 2. Every script imports this so all methods run with identical settings.

Fixes applied (each verified in a check):
  1. VRAM hard limit  - torch.cuda.set_per_process_memory_fraction: PyTorch may not allocate more than the VRAM that is
                        actually free, so nothing spills into system RAM (check 8a showed WSL spills otherwise).
  2. Attention kernel - transformers' enable_gqa path forces SDPA's math kernel on sm_75; repeat_kv lets the
                        memory-efficient kernel run (checks 7b, 7c: outputs unchanged).
  3. FP32 compute     - bitsandbytes NF4 weights, matmuls in FP32: ~3x faster prefill on GTX 1650, answers unchanged (check 8b).
  4. Tied lm_head     - checkpoint lm_head is bit-identical to embed_tokens; share it (check 6c/6d), saves ~585 MiB.
  5. MLP in pieces    - during prefill the MLP runs on 1024-token pieces (per-token maths, unchanged): less
                        temporary memory; 16 generated tokens identical (check 10a).
  6. expandable_segments (set in run_py.sh) - removes allocator fragmentation; GPU-only 8k -> 10k (check 10e).
"""
MLP_CHUNK = 1024


def apply_mlp_chunking():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
    if getattr(Qwen3MLP, "_chunked", False):
        return
    orig = Qwen3MLP.forward
    def fwd(self, x):
        if x.shape[1] <= MLP_CHUNK:
            return orig(self, x)
        return torch.cat([orig(self, x[:, i:i + MLP_CHUNK]) for i in range(0, x.shape[1], MLP_CHUNK)], dim=1)
    Qwen3MLP.forward = fwd; Qwen3MLP._chunked = True
import gc, subprocess, torch

MODEL = "Qwen/Qwen3-1.7B"
MiB = 1 << 20
SAFETY_MIB = 256   # headroom for allocations outside PyTorch's allocator (cuBLAS/bnb workspaces)


def set_vram_limit():
    free, total = torch.cuda.mem_get_info()
    limit = free - SAFETY_MIB * MiB
    torch.cuda.set_per_process_memory_fraction(limit / total, 0)
    print(f"[setup] VRAM: card {total/MiB:.0f} MiB, free at start {free/MiB:.0f} MiB -> PyTorch hard limit {limit/MiB:.0f} MiB")
    return limit


def apply_attention_fix():
    import transformers.integrations.sdpa_attention as sa
    sa.use_gqa_in_sdpa = lambda attention_mask, key: False


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import transformers.utils.logging as tl
    tl.disable_progress_bar(); tl.set_verbosity_error()
    apply_attention_fix()
    apply_mlp_chunking()
    # Fix 8 (main run 1, 2026-11-20): transformers 5.2 materialises weights on the GPU with a 4-thread pool before
    # 4-bit quantisation; once, unquantised tensors piled up to 2.95 GiB and loading hit the 3040 MiB limit (17 earlier
    # loads were fine). One worker makes loading sequential; the loaded weights are the same.
    import transformers.core_model_loading as cml
    cml.GLOBAL_WORKERS = 1
    limit = set_vram_limit()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="cuda", dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float32))
    model.lm_head.weight = model.model.embed_tokens.weight
    model.eval(); gc.collect(); torch.cuda.empty_cache()
    print(f"[setup] model loaded: {torch.cuda.memory_allocated()/MiB:.0f} MiB on GPU | {ram()}")
    return model, tok, limit


def ram():
    f = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.split("\n")
    m, s = f[1].split(), f[2].split()
    return f"RAM used {m[2]} MiB, avail {m[6]} MiB, swap used {s[2]} MiB"


def swap_mib():
    return int(subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.split("\n")[2].split()[2])
