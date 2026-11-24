#!/bin/bash
# usage: run_exp.sh <logname> [run_ruler.py args...]  -- same environment as run_py.sh (fixes incl. expandable_segments)
# Restarts run_ruler.py (which resumes from its output file) after a crash, up to 5 attempts, 60 s apart.
N=$1; shift
cd ~/kv-offload && . .venv/bin/activate
export HF_HOME=~/kv-offload/models/hf HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}
LOG=~/kv-offload/logs/exp/$N.log
echo "[run_exp] $(date '+%F %T') args: $*" >> $LOG
for attempt in 1 2 3 4 5; do
  echo "[run_exp] attempt $attempt start $(date '+%F %T')" >> $LOG
  nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader >> $LOG
  free -m >> $LOG
  python scripts/exp/run_ruler.py "$@" 2>&1 | grep --line-buffered -v "HF_TOKEN" >> $LOG
  rc=${PIPESTATUS[0]}
  echo "[run_exp] attempt $attempt exit $rc $(date '+%F %T')" >> $LOG
  [ $rc -eq 0 ] && break
  sleep 60
done
echo "[run_exp] finished $(date '+%F %T') exit $rc" >> $LOG
