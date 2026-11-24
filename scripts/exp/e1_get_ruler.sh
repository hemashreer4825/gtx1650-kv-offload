#!/bin/bash
# Experiment step 1a: get official RULER code (pinned commit recorded) and inspect data-generation scripts.
set -e
mkdir -p ~/kv-offload/third_party ~/kv-offload/logs/exp
cd ~/kv-offload/third_party
if [ ! -d RULER ]; then git clone --depth 1 https://github.com/NVIDIA/RULER.git; fi
cd RULER
echo "RULER commit: $(git rev-parse HEAD)  date: $(git log -1 --format=%cd)"
echo "#### files"; ls scripts/data scripts/data/synthetic scripts/data/synthetic/json 2>/dev/null
echo "#### requirements in repo"; cat requirements.txt 2>/dev/null | head -40 || true
echo "#### niah.py args"; grep -n "add_argument" scripts/data/synthetic/niah.py
echo "#### variable_tracking.py args"; grep -n "add_argument" scripts/data/synthetic/variable_tracking.py
echo "#### imports"; grep -hn "^import\|^from" scripts/data/synthetic/niah.py scripts/data/synthetic/variable_tracking.py | sort -u
echo "#### synthetic.yaml niah/vt entries"; sed -n 1,200p scripts/synthetic.yaml | grep -nA12 -E "^(niah_single_2|niah_multikey_1|niah_multivalue|vt):"
echo "#### essay download script"; cat scripts/data/synthetic/json/download_paulgraham_essay.py 2>/dev/null | head -60
