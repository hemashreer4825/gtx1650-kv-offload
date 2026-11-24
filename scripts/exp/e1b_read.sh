#!/bin/bash
R=~/kv-offload/third_party/RULER/scripts/data
echo "######## constants.py"; cat $R/synthetic/constants.py | grep -v "^#"
echo "######## download script tail"; sed -n 60,200p $R/synthetic/json/download_paulgraham_essay.py
echo "######## tokenizer.py hf"; grep -n -A12 "class HFTokenizer" $R/tokenizer.py
echo "######## niah.py 74-end"; sed -n 74,400p $R/synthetic/niah.py
echo "######## vt.py generation"; grep -n "def \|template\|answer_prefix\|length\|outputs" $R/synthetic/variable_tracking.py | head -60
echo "######## prepare.py template use"; grep -n "template\|answer_prefix\|remove_newline\|subprocess\|command" $R/prepare.py | head -40
echo "######## template.py qwen/base"; grep -n -i -B1 -A3 "base\|qwen\|meta-chat" $R/template.py | head -40
