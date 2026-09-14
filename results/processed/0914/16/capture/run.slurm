#!/usr/bin/env bash
#SBATCH --job-name=pi05-sglang-roofline
#SBATCH --partition=dev
#SBATCH --qos=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/fact_data/xinyaowang/vla_vllm_profiling/results/processed/0914/12/slurm_%j.log
set -euo pipefail
project_dir=/fact_data/xinyaowang/vla_vllm_profiling
output_dir="$project_dir/results/processed/0914/12"
source "$output_dir/sglang_env.sh"
export PYTHONPATH="/fact_data/xinyaowang/sglang-pi05-profiling/python:$output_dir/harness"
cd "$output_dir/harness"
exec "$project_dir/.venv-sglang-pi05/bin/python" -u -m profiling.sglang_suite --output "$output_dir"
