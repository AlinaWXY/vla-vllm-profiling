#!/usr/bin/env bash
# Source from a shell whose project directory and Python environment are ready.
# Use user-owned scratch for disposable caches; results belong in /fact_data.
VLA_SCRATCH_ROOT="${VLA_SCRATCH_ROOT:-/scratch/$(id -un)/vla-vllm-profiling}"
case "$VLA_SCRATCH_ROOT" in
    /scratch/*) ;;
    *) echo 'VLA_SCRATCH_ROOT must be below /scratch.' >&2; return 1 2>/dev/null || exit 1 ;;
esac
mkdir -p "$VLA_SCRATCH_ROOT"/{tmp,huggingface,pip,xdg,triton,torch,inductor,uv,matplotlib}
export TMPDIR="$VLA_SCRATCH_ROOT/tmp"
export HF_HOME="$VLA_SCRATCH_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR="$VLA_SCRATCH_ROOT/pip"
export XDG_CACHE_HOME="$VLA_SCRATCH_ROOT/xdg"
export TRITON_CACHE_DIR="$VLA_SCRATCH_ROOT/triton"
export TORCH_HOME="$VLA_SCRATCH_ROOT/torch"
export TORCHINDUCTOR_CACHE_DIR="$VLA_SCRATCH_ROOT/inductor"
export UV_CACHE_DIR="$VLA_SCRATCH_ROOT/uv"
export MPLCONFIGDIR="$VLA_SCRATCH_ROOT/matplotlib"
export TOKENIZERS_PARALLELISM=false
