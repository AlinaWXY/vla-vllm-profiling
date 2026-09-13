#!/usr/bin/env bash
#SBATCH --job-name=pi05-full-numerics
#SBATCH --partition=dev
#SBATCH --qos=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/fact_data/xinyaowang/vla_vllm_profiling/results/processed/0913/21/slurm_%j.log
set -euo pipefail
cd /fact_data/xinyaowang/vla_vllm_profiling
export TMPDIR=/scratch/xinyaowang/vla-vllm-profiling/slurm-tmp-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
source scripts/cache_env.sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/fact_data/xinyaowang/vllm-omni-pi05-numerical-fixes:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MAX_JOBS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_CACHE_DIR=/scratch/xinyaowang/vla-vllm-profiling/triton-numerics-x86
export TORCHINDUCTOR_CACHE_DIR=/scratch/xinyaowang/vla-vllm-profiling/inductor-numerics-x86
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
 .venv-numerics-x86-cu130/bin/python /fact_data/xinyaowang/vllm-omni-pi05-numerical-fixes/tests/pi05/test_realtime_numerical_contracts.py
.venv-numerics-x86-cu130/bin/python -u -m profiling.validate_framework \
  --checkpoint /scratch/xinyaowang/vla-vllm-profiling/assets/pi05_base/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba \
  --tokenizer /scratch/xinyaowang/vla-vllm-profiling/assets/paligemma_tokenizer \
  --output results/processed/0913/21 \
  --diagnostic results/processed/0913/15 \
  --baseline-source vllm-omni/vllm_omni/diffusion/models/pi05/realtime_triton.py
