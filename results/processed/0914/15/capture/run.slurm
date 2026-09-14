#!/usr/bin/env bash
#SBATCH --job-name=pi05-l20-coarse
#SBATCH --partition=dev
#SBATCH --qos=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/fact_data/xinyaowang/vla_vllm_profiling/results/processed/0914/07/slurm_%j.log
set -euo pipefail
project_dir=/fact_data/xinyaowang/vla_vllm_profiling
output_dir="$project_dir/results/processed/0914/07"
export TMPDIR=/scratch/xinyaowang/vla-vllm-profiling/slurm-tmp-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
source "$project_dir/scripts/cache_env.sh"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PI05_PROFILE_SOURCE="$output_dir/harness"
export PYTHONPATH="/fact_data/xinyaowang/vllm-omni-pi05-numerical-fixes:$PI05_PROFILE_SOURCE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MAX_JOBS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_CACHE_DIR=/scratch/xinyaowang/vla-vllm-profiling/triton-numerics-x86
export TORCHINDUCTOR_CACHE_DIR=/scratch/xinyaowang/vla-vllm-profiling/inductor-numerics-x86
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
cd "$output_dir/harness"
exec "$project_dir/.venv-numerics-x86-cu130/bin/python" -u "$output_dir/run_suite.py"
