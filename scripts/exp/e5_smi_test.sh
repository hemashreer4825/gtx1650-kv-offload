#!/bin/bash
which nvidia-smi
timeout 3 nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader,nounits -lms 500
echo "exit $?"
cd ~/kv-offload && . .venv/bin/activate && python -c "import pynvml; print('pynvml present')" 2>&1 | tail -1
