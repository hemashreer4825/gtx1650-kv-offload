#!/bin/bash
S=/mnt/c/Users/hittu/AppData/Local/Temp/claude/93ce209826d37306/2a22c400-4693-43dd-8c5a-6951183ed74f/scratchpad
cd ~/kv-offload
cp $S/run_ruler.py scripts/exp/; cp $S/run_exp.sh scripts/exp/; cp $S/e10_diag.sh $S/e11_relaunch.sh scripts/exp/
. .venv/bin/activate && python -m py_compile scripts/exp/run_ruler.py && echo "syntax OK"
mv logs/exp/main.log logs/exp/main_attempt2_cuda_unknown.log
ls results/exp/; cat results/exp/main.jsonl 2>/dev/null | head -c 300
