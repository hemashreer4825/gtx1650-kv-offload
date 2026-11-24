#!/bin/bash
set -e
mkdir -p ~/kv-offload/logs/setup
cd ~/kv-offload
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu128
pip install -q transformers accelerate bitsandbytes kvpress
echo "=== versions"
python - <<'EOF'
import importlib, torch
for m in ["torch","transformers","accelerate","bitsandbytes","kvpress"]:
    try:
        mod = importlib.import_module(m); print(f"{m:14s}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"{m:14s} IMPORT FAILED: {e!r}")
print("torch CUDA build:", torch.version.cuda)
print("GPU seen by torch:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("GPU capability:", torch.cuda.get_device_capability(0))
print("archs compiled into torch:", torch.cuda.get_arch_list())
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
print("fp16 matmul on GPU OK:", float((x @ x).float().abs().mean()) > 0)
EOF
echo "=== pip check (dependency conflicts)"; pip check || true
du -sh .venv
