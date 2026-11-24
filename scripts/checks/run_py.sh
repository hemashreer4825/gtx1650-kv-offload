#!/bin/bash
# usage: run_py.sh <script.py> <logname> [args...]  -- runs inside the kv-offload venv, logs to ~/kv-offload/logs/setup/<logname>.log
# Fix (check 10e): expandable_segments removes allocator fragmentation; GPU-only limit 8k -> 10k. On by default.
S=$1; L=$2; shift 2
cd ~/kv-offload && . .venv/bin/activate
export HF_HOME=~/kv-offload/models/hf HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}
python "$S" "$@" 2>&1 | tee ~/kv-offload/logs/setup/$L.log
