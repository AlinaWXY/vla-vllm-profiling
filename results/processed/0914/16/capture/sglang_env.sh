#!/usr/bin/env bash
# Source inside a Slurm allocation after selecting the isolated SGLang runtime.
export VLA_SCRATCH_ROOT=/scratch/xinyaowang/vla-vllm-profiling
export TMPDIR="$VLA_SCRATCH_ROOT/sglang-tmp-${SLURM_JOB_ID:-setup}"
export SGLANG_CACHE_DIR="$VLA_SCRATCH_ROOT/sglang-cache"
export SGLANG_JIT_CACHE_DIR="$SGLANG_CACHE_DIR/jit"
export TRITON_CACHE_DIR="$SGLANG_CACHE_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$SGLANG_CACHE_DIR/inductor"
export TORCH_EXTENSIONS_DIR="$SGLANG_CACHE_DIR/torch-extensions"
export CUDA_HOME=/fact_data/xinyaowang/vla_vllm_profiling/.venv-sglang-pi05/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MAX_JOBS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH=/fact_data/xinyaowang/sglang-pi05-profiling/python:/fact_data/xinyaowang/vla_vllm_profiling
source /fact_data/xinyaowang/vla_vllm_profiling/scripts/cache_env.sh
