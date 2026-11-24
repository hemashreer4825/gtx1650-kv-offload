#!/bin/bash
echo "### WSL kernel messages about the GPU (dxg) in this boot, last 15"
sudo -n dmesg 2>/dev/null | grep -i "dxg\|nvidia\|gpu" | tail -15 || dmesg 2>&1 | grep -i "dxg\|nvidia\|gpu" | tail -15
echo "### processes"; ps -eo pid,args | grep "[r]un_ruler" | cut -c1-60
echo "### GPU health test: 2 GB of matmuls + compare with CPU"
cd ~/kv-offload && . .venv/bin/activate
python - <<'EOF'
import torch, time
a = torch.randn(2048, 2048, device="cuda"); b = torch.randn(2048, 2048, device="cuda")
t = time.time()
for _ in range(200): c = a @ b
torch.cuda.synchronize(); print(f"200 matmuls OK in {time.time()-t:.2f}s")
ref = (a.cpu() @ b.cpu()); print("max diff vs CPU:", (c.cpu() - ref).abs().max().item())
x = torch.empty(int(2.5 * 2**30) // 4, device="cuda"); x.fill_(1.0); torch.cuda.synchronize(); print("2.5 GiB alloc+fill OK, sum", x[:10].sum().item())
EOF
nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader
