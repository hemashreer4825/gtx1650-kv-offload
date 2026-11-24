#!/bin/bash
cd ~/kv-offload/results/exp
python3 - <<'EOF'
import json, collections
rows = [json.loads(l) for l in open("main.jsonl")]
c = collections.Counter(r["status"] for r in rows)
print("records:", len(rows), dict(c))
for r in rows:
    print(f'{r["time"]} {r["length"]:6d} {r["task"]:16s} {r["method"]:12s} {r["status"]:8s} score {r.get("score","-")} | decode {r.get("decode_tok_s","-")} | temp max {r.get("gpu_temp_max")} | {r.get("error","")[:60]}')
EOF
