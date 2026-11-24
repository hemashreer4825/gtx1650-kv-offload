#!/bin/bash
# Experiment step 2: install RULER data-generation dependencies into the kv-offload venv, download essays (haystack text).
mkdir -p ~/kv-offload/logs/exp ~/kv-offload/scripts/exp
L=~/kv-offload/logs/exp/e2_ruler_deps.log
{
cd ~/kv-offload && . .venv/bin/activate
echo "### torch/transformers before:"; pip show torch transformers 2>/dev/null | grep -E "^(Name|Version)"
pip install --no-input nltk wonderwords html2text beautifulsoup4 2>&1 | tail -3
echo "### installed:"; pip show nltk wonderwords html2text beautifulsoup4 | grep -E "^(Name|Version)"
echo "### torch/transformers after (must be unchanged):"; pip show torch transformers | grep -E "^(Name|Version)"
python -c "import nltk; [print(p, nltk.download(p, quiet=True)) for p in ('punkt','punkt_tab')]"
echo "### essay download (217 URLs)"
cd ~/kv-offload/third_party/RULER/scripts/data/synthetic/json
python download_paulgraham_essay.py 2>&1 | grep -v "it/s\]" | tail -15
ls -la PaulGrahamEssays.json
python -c "import json; t=json.load(open('PaulGrahamEssays.json'))['text']; print('essay chars', len(t), 'words', len(t.split()))"
} > $L 2>&1
cat $L
