#!/bin/bash
# Experiment step 3: generate RULER data with RULER's own prepare.py (unmodified), Qwen3 tokenizer (hf), base template.
L=~/kv-offload/logs/exp/e3_gen_ruler.log
TOK=~/kv-offload/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
{
cd ~/kv-offload && . .venv/bin/activate
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python -c "import tenacity, yaml; print('tenacity', tenacity.__version__ if hasattr(tenacity,'__version__') else 'ok', '| yaml', yaml.__version__)" 2>&1 || pip install --no-input tenacity pyyaml 2>&1 | tail -1
echo "RULER commit: $(git -C third_party/RULER rev-parse HEAD)"
cd third_party/RULER/scripts/data
for LEN in 4096 8192 16384 20480; do
  for T in niah_single_2 niah_multikey_1 niah_multivalue vt; do
    t0=$(date +%s)
    python prepare.py --save_dir ~/kv-offload/data/ruler/$LEN --benchmark synthetic --task $T \
      --tokenizer_path $TOK --tokenizer_type hf --max_seq_length $LEN --model_template_type base --num_samples 5 \
      > ~/kv-offload/logs/exp/prep_${T}_${LEN}.out 2>&1
    rc=$?
    echo "$LEN $T: exit $rc, $(( $(date +%s)-t0 )) s, $(wc -l < ~/kv-offload/data/ruler/$LEN/$T/validation.jsonl 2>/dev/null) samples"
    [ $rc -ne 0 ] && tail -20 ~/kv-offload/logs/exp/prep_${T}_${LEN}.out
  done
done
echo "### sample check (token counts with the Qwen3 tokenizer)"
cd ~/kv-offload
python - <<'EOF'
import json, os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/kv-offload/models/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"))
for L in (4096, 8192, 16384, 20480):
    for T in ("niah_single_2", "niah_multikey_1", "niah_multivalue", "vt"):
        p = os.path.expanduser(f"~/kv-offload/data/ruler/{L}/{T}/validation.jsonl")
        if not os.path.exists(p): print(L, T, "MISSING"); continue
        s = [json.loads(l) for l in open(p)]
        n = [len(tok.encode(x["input"])) for x in s]
        print(f"{L:6d} {T:16s} n={len(s)} input tokens {min(n)}-{max(n)} | answers e.g. {s[0]['outputs']} | in text: {all(a in x['input'] for x in s for a in x['outputs'])}")
s = json.loads(open(os.path.expanduser("~/kv-offload/data/ruler/4096/niah_single_2/validation.jsonl")).readline())
print("--- example niah_single_2 start:", repr(s["input"][:300])); print("--- end:", repr(s["input"][-300:])); print("--- answer_prefix:", repr(s["answer_prefix"]))
s = json.loads(open(os.path.expanduser("~/kv-offload/data/ruler/4096/vt/validation.jsonl")).readline())
print("--- example vt end:", repr(s["input"][-250:])); print("--- answer_prefix:", repr(s["answer_prefix"]), "| outputs", s["outputs"])
EOF
du -sh ~/kv-offload/data/ruler
} > $L 2>&1
cat $L
