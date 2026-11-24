#!/bin/bash
P=~/kv-offload/.venv/lib/python3.10/site-packages/kvpress/pipeline.py
echo "######## kvpress pipeline 105-175 (preprocess)"; sed -n 105,175p $P
echo "######## kvpress pipeline 200-end"; sed -n 200,400p $P
echo "######## RULER eval string match"; grep -n -B2 -A12 "def string_match" ~/kv-offload/third_party/RULER/scripts/eval/synthetic/constants.py
echo "######## RULER eval postprocess"; grep -n -A12 "def postprocess_pred" ~/kv-offload/third_party/RULER/scripts/eval/evaluate.py
echo "######## essay url count"; wc -l ~/kv-offload/third_party/RULER/scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt; head -3 ~/kv-offload/third_party/RULER/scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt
echo "######## nltk / wonderwords present?"; cd ~/kv-offload && . .venv/bin/activate && python -c "import nltk" 2>&1 | tail -1; python -c "import wonderwords" 2>&1 | tail -1; python -c "import html2text, bs4" 2>&1 | tail -1
