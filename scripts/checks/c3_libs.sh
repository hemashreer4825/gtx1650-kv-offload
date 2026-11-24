#!/bin/bash
cd ~/kv-offload && . .venv/bin/activate
SP=$(python -c "import site;print(site.getsitepackages()[0])")
echo "=== transformers: cache classes defined in cache_utils.py"
grep -nE "^class " $SP/transformers/cache_utils.py
echo "=== transformers: 'offload' mentions in cache_utils.py (first 40)"
grep -nE "offload" $SP/transformers/cache_utils.py | head -40
echo "=== transformers: cache_implementation names accepted by generate()"
grep -nE "ALL_CACHE_IMPLEMENTATIONS|STATIC_CACHE_IMPLEMENTATIONS|DYNAMIC_CACHE_IMPLEMENTATIONS|DEPRECATED" $SP/transformers/generation/configuration_utils.py | head -15
echo "=== transformers: Qwen3 model present?"
ls $SP/transformers/models/qwen3/ 2>&1 | head
echo "=== kvpress: package files"
ls $SP/kvpress $SP/kvpress/presses | head -80
echo "=== kvpress: press classes exported"
python -c "import kvpress; print(sorted(n for n in dir(kvpress) if 'Press' in n))"
echo "=== kvpress: supported model classes (grep)"
grep -rnE "Qwen3|Qwen2|Llama|Mistral|Phi3|Gemma" $SP/kvpress/*.py | grep -iE "import|supported|isinstance|Attention" | head -30
