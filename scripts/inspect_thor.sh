#!/usr/bin/env bash
# Read-only environment inventory; no workloads, installs or clock changes.
set -u
export PYTHONDONTWRITEBYTECODE=1
VLA_INSPECT_PYTHON="${VLA_INSPECT_PYTHON:-python3}"
hostname
uname -m
cat /etc/os-release
free -h
for tool in sbatch srun sinfo scontrol ncu nsys nvcc python3 docker; do
    command -v "$tool" || true
done
if command -v sinfo >/dev/null; then
    sinfo -o '%P %a %l %D %G'
    squeue -u "$(id -un)"
    scontrol show partition
fi
if command -v ncu >/dev/null; then
    ncu --version
else
    # One known vendor directory level; no broad filesystem search.
    for executable in /opt/nvidia/nsight-compute/*/ncu; do
        if test -x "$executable"; then
            printf 'Installed NCU: %s\n' "$executable"
            "$executable" --version
        fi
    done
fi
if command -v nvcc >/dev/null; then
    nvcc --version
elif test -x /usr/local/cuda/bin/nvcc; then
    /usr/local/cuda/bin/nvcc --version
fi
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
fi
if command -v "$VLA_INSPECT_PYTHON" >/dev/null; then
    "$VLA_INSPECT_PYTHON" -B - <<'PY'
import importlib.metadata
import platform
import sys
print('Python:', platform.python_version(), sys.executable)
for name in ('torch', 'torchvision', 'vllm', 'vllm-omni', 'triton', 'transformers', 'huggingface-hub', 'safetensors', 'numpy', 'matplotlib', 'lerobot'):
    try:
        print(name, importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:
        print(name, 'not installed in this Python environment')
PY
fi
if command -v docker >/dev/null; then
    if test -r /var/run/docker.sock && test -w /var/run/docker.sock; then
        docker ps -a --format '{{.Names}} | {{.Image}} | {{.Status}}' || true
        docker images --format '{{.Repository}}:{{.Tag}} | {{.Size}}' || true
    else
        echo 'Docker inventory unavailable with current socket permissions; sudo is not attempted.'
    fi
fi
for directory in /fact_data /scratch; do
    if test -d "$directory"; then ls -ld "$directory"; df -h "$directory"; fi
done
