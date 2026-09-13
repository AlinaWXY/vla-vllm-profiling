#!/usr/bin/env bash
set -euo pipefail
cd /fact_data/xinyaowang/vla_vllm_profiling
export PI05_PROFILE_SOURCE=/fact_data/xinyaowang/vla_vllm_profiling/results/processed/0913/27/harness
export PI05_OMNI_SOURCE=/fact_data/xinyaowang/vllm-omni-pi05-numerical-fixes
bash scripts/thor_python.sh -u -m profiling.bench_pi05 --checkpoint /scratch/xinyaowang/vla-vllm-profiling/assets/pi05_base/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba --tokenizer /scratch/xinyaowang/vla-vllm-profiling/assets/paligemma_tokenizer --output results/processed/0913/27/workload --cameras 3 --steps 10 --warmup 3 --iterations 20 --scope all --buffered-original
