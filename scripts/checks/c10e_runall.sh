#!/bin/bash
C=~/kv-offload/scripts/checks
F='Loading weights|Warning: You are|UserWarning|scaled_dot'
echo "################ A: dynamic cache + expandable_segments"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 bash $C/run_py.sh $C/c10e_frag.py c10e_dynamic_expandable dynamic 2>&1 | grep -vE "$F"
echo "################ B: static cache (no expandable_segments)"
PYTHONUNBUFFERED=1 bash $C/run_py.sh $C/c10e_frag.py c10e_static static 2>&1 | grep -vE "$F"
