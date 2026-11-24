#!/bin/bash
F=~/kv-offload/.venv/lib/python3.10/site-packages/transformers/cache_utils.py
echo "######## StaticLayer"; sed -n 254,330p $F
echo "######## StaticCache.__init__"; sed -n 1015,1092p $F | grep -vE '^\s*$'
~/kv-offload/.venv/bin/python - <<'EOF'
import torch
print("torch._C._host_emptyCache exists:", hasattr(torch._C, "_host_emptyCache"))
print("torch.cuda.host_memory_stats exists:", hasattr(torch.cuda, "host_memory_stats"))
EOF
