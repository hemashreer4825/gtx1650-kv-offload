#!/bin/bash
ls -la ~/kv-offload-install.log
ls ~/kv-offload ~/kv-offload/.venv/bin | head -20
du -sh ~/kv-offload/.venv
~/kv-offload/.venv/bin/pip list 2>/dev/null | grep -iE "^(torch|transformers|accelerate|bitsandbytes|kvpress) "
