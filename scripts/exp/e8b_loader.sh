#!/bin/bash
pkill -f "scripts/exp/run_ruler.py"; sleep 3
echo "after kill:"; ps -eo pid,args | grep "[r]un_ruler" | cut -c1-80; nvidia-smi --query-gpu=memory.used --format=csv,noheader
F=~/kv-offload/.venv/lib/python3.10/site-packages/transformers/core_model_loading.py
echo "#### thread pool / workers / env vars in core_model_loading.py"
grep -n "ThreadPoolExecutor\|max_workers\|GLOBAL_WORKERS\|environ\|PARALLEL\|def spawn_materialize\|def _job" $F | head -30
echo "#### modeling_utils: parallel loading switches"
grep -n "HF_ENABLE_PARALLEL_LOADING\|HF_PARALLEL_LOADING_WORKERS\|ThreadPoolExecutor" ~/kv-offload/.venv/lib/python3.10/site-packages/transformers/modeling_utils.py | head
echo "#### previous successful loads (model loaded lines) in logs"
grep -h "model loaded" ~/kv-offload/logs/setup/*.log ~/kv-offload/logs/exp/*.log | sort | uniq -c
