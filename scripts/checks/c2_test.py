import importlib, torch
for m in ["torch", "transformers", "accelerate", "bitsandbytes", "kvpress"]:
    try:
        mod = importlib.import_module(m); print(f"{m:14s}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"{m:14s} IMPORT FAILED: {e!r}")
print("torch CUDA build:", torch.version.cuda)
print("GPU seen by torch:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("GPU capability:", torch.cuda.get_device_capability(0))
print("archs compiled into torch:", torch.cuda.get_arch_list())
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
print("fp16 matmul on GPU OK:", bool((x @ x).float().abs().mean() > 0))
import bitsandbytes as bnb
w = bnb.nn.Linear4bit(256, 256, bias=False, compute_dtype=torch.float16, quant_type="nf4").cuda()
y = w(torch.randn(4, 256, device="cuda", dtype=torch.float16))
print("bitsandbytes 4-bit layer on GPU OK:", tuple(y.shape), "finite:", bool(torch.isfinite(y).all()))
