#!/usr/bin/env bash
set -euo pipefail
cd /fact_data/xinyaowang/vla_vllm_profiling
if [[ -f results/processed/0913/29/CANCELLED.json ]]; then
  echo "Experiment 0913/29 was cancelled; allocate a new experiment for future captures." >&2
  exit 1
fi
export PI05_PROFILE_SOURCE=/fact_data/xinyaowang/vla_vllm_profiling/results/processed/0913/29/harness
export PI05_OMNI_SOURCE=/fact_data/xinyaowang/vllm-omni-pi05-numerical-fixes
bash scripts/thor_python.sh -u -m profiling.ncu --ncu /opt/nvidia/nsight-compute/2026.1.1/ncu capture --report-name vla --contract results/processed/0913/29/metrics.json --output results/processed/0913/29/ncu --replay-mode kernel --graph-profiling graph --cache-control none --nvtx-scope '' --scope-description 'VLM + Action Expert; all 3 camera masks valid; full GPU computation, static conditioning/preprocessing excluded' -- bash scripts/thor_python.sh -u -m profiling.bench_pi05 --checkpoint /scratch/xinyaowang/vla-vllm-profiling/assets/pi05_base/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba --tokenizer /scratch/xinyaowang/vla-vllm-profiling/assets/paligemma_tokenizer --output results/processed/0913/29/workload --buffered-original --expected-fingerprint results/processed/0913/27/workload/full_scopes.json --cameras 3 --steps 10 --warmup 3 --scope all --profile --profile-kind graphs
