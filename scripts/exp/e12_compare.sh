#!/bin/bash
cd ~/kv-offload/results/exp
python3 - <<'EOF'
import json
p = {(d["method"], d["task"], d["length"], d["index"]): d for d in map(json.loads, open("pilot2.jsonl"))}
for d in map(json.loads, open("main.jsonl")):
    k = (d["method"], d["task"], d["length"], d["index"])
    if k in p and d["status"] == "ok":
        print(k, "full output identical to pilot 2:", d["output"] == p[k]["output"], "| tokens", d["gen_tokens"], p[k]["gen_tokens"])
EOF
