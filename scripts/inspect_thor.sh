#!/usr/bin/env bash
# Read-only environment inventory; no workloads, installs or clock changes.
set -u
hostname
uname -m
cat /etc/os-release
for tool in sbatch srun sinfo scontrol ncu nsys nvcc python3 docker; do
    command -v "$tool" || true
done
if command -v sinfo >/dev/null; then
    sinfo -o '%P %a %l %D %G'
    squeue -u "$(id -un)"
    scontrol show partition
fi
if command -v ncu >/dev/null; then ncu --version; fi
if command -v nvcc >/dev/null; then nvcc --version; fi
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
fi
if command -v python3 >/dev/null; then
    python3 - <<'PY'
import importlib.metadata
import platform
print('Python:', platform.python_version())
for name in ('torch', 'vllm', 'vllm-omni', 'triton', 'transformers', 'huggingface-hub'):
    try:
        print(name, importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:
        print(name, 'not installed in this Python environment')
PY
fi
for directory in /fact_data /scratch; do
    if test -d "$directory"; then ls -ld "$directory"; df -h "$directory"; fi
done
