#!/bin/bash
S=/mnt/c/Users/hittu/AppData/Local/Temp/claude/93ce209826d37306/2a22c400-4693-43dd-8c5a-6951183ed74f/scratchpad
cd ~/kv-offload
cp $S/kvsetup.py scripts/checks/kvsetup.py
. .venv/bin/activate && python -m py_compile scripts/checks/kvsetup.py && echo "kvsetup syntax OK"
mv logs/exp/main.log logs/exp/main_attempt1_load_oom.log
ls -la results/exp/main.jsonl 2>&1
cp $S/e8b_loader.sh $S/e9_relaunch.sh scripts/exp/
nohup bash scripts/exp/run_exp.sh main --out /home/hittu/kv-offload/results/exp/main.jsonl --n 1 > /dev/null 2>&1 &
echo "launched pid $!"
