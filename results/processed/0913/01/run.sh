#!/usr/bin/env bash
set -euo pipefail
cd /fact_data/xinyaowang/vla_vllm_profiling
bash scripts/thor_python.sh -u -m profiling.ncu --ncu /opt/nvidia/nsight-compute/2026.1.1/ncu capture --contract results/processed/0913/01/metrics.json --output results/processed/0913/01/ncu --cache-control all --launch-count 168 -- bash scripts/thor_python.sh -u -m profiling.bench_pi05 --checkpoint /scratch/xinyaowang/vla-vllm-profiling/assets/pi05_base/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba --tokenizer /scratch/xinyaowang/vla-vllm-profiling/assets/paligemma_tokenizer --output results/processed/0913/01/workload --cameras 3 --steps 10 --warmup 3 --profile
