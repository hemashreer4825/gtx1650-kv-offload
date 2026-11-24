#!/bin/bash
cd ~/kv-offload && . .venv/bin/activate
SP=$(python -c "import site;print(site.getsitepackages()[0])")
echo "######## cache_utils.py 27-84 (layer offload / prefetch)"; sed -n 27,84p $SP/transformers/cache_utils.py
echo "######## cache_utils.py 740-800 (Cache prefetch/offload/update)"; sed -n 740,800p $SP/transformers/cache_utils.py
echo "######## pin_memory anywhere in cache_utils?"; grep -n "pin_memory\|non_blocking\|Stream" $SP/transformers/cache_utils.py
echo "######## streaming_llm_press.py"; grep -vE '^\s*#|^\s*$' $SP/kvpress/presses/streaming_llm_press.py | head -60
echo "######## snapkv_press.py"; grep -vE '^\s*$' $SP/kvpress/presses/snapkv_press.py | head -110
