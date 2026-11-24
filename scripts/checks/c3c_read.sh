#!/bin/bash
cd ~/kv-offload && . .venv/bin/activate
SP=$(python -c "import site;print(site.getsitepackages()[0])")
echo "######## DynamicLayer (cache_utils.py 85-125)"; sed -n 85,125p $SP/transformers/cache_utils.py
echo "######## base_press.py forward_hook"; grep -nE "def forward_hook|q_len|cache_position|prefill|decod|return output|def __call__|register_forward_hook" $SP/kvpress/presses/base_press.py | head -30
echo "######## pipeline.py: question handling"; grep -nE "question|def _forward|def preprocess|context_ids|press\(|max_new_tokens|generate_answer|def postprocess" $SP/kvpress/pipeline.py | head -50
