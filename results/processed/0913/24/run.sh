#!/usr/bin/env bash
set -euo pipefail
cd /fact_data/xinyaowang/vla_vllm_profiling
if /opt/nvidia/nsight-compute/2026.1.1/ncu --target-processes all --nvtx --replay-mode kernel --profile-from-start off --graph-profiling graph --cache-control none --clock-control none --metrics gpu__time_duration.sum,lts__t_bytes.sum,sm__ops_path_tensor_src_bf16_dst_fp32.sum --export results/processed/0913/24/range_probe/range bash scripts/thor_python.sh -u -m profiling.range_probe --output results/processed/0913/24/range_probe/workload.json > results/processed/0913/24/range_probe/ncu.log 2>&1; then
  /opt/nvidia/nsight-compute/2026.1.1/ncu --import results/processed/0913/24/range_probe/range.ncu-rep --page raw --csv --print-units base --print-fp > results/processed/0913/24/range_probe/raw.csv
else
  echo "Range probe failed; raw log retained. Baseline timing can proceed independently."
fi
