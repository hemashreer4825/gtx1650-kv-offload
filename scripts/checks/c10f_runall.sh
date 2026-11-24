#!/bin/bash
C=~/kv-offload/scripts/checks
F='Loading weights|Warning: You are|UserWarning|scaled_dot'
echo "################ static cache + expandable_segments"
bash $C/run_py.sh $C/c10e_frag.py c10e_static_expandable static 2>&1 | grep -vE "$F"
for v in dynamic dynamic_empty static; do
  echo "################ Fix B offload variant: $v (4096 tokens)"
  bash $C/run_py.sh $C/c10c_offload_ram.py c10c_offload_$v $v 4096 2>&1 | grep -vE "$F"
done
