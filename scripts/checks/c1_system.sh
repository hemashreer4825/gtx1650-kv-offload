#!/bin/bash
echo "=== GPU / driver"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,compute_cap --format=csv
nvidia-smi | grep -i "cuda version"
echo "=== Python"
which python3; python3 --version; python3 -m venv --help >/dev/null 2>&1 && echo "venv module: OK" || echo "venv module: MISSING"
echo "=== CUDA toolkit (nvcc) installed system-wide?"
which nvcc || ls -d /usr/local/cuda* 2>/dev/null || echo "no nvcc"
echo "=== RAM (MiB)"; free -m
echo "=== Disk"; df -h / | tail -1
echo "=== CPU"; lscpu | grep -E "Model name|^CPU\(s\)"
echo "=== Pinned (locked) memory limit"; ulimit -l
