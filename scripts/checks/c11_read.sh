#!/bin/bash
F=~/kv-offload/.venv/lib/python3.10/site-packages/transformers/cache_utils.py
echo "######## DynamicLayer rest (119-167)"; sed -n 119,167p $F
echo "######## Cache class methods (682-906)"; awk 'NR>682 && NR<906 && /    def /{print NR": "$0}' $F
echo "######## Cache.get_seq_length / get_mask_sizes / __len__"; awk 'NR>682 && NR<906' $F | grep -nA12 -E "def (get_seq_length|get_mask_sizes|__len__)\(" | head -60
