#!/usr/bin/env bash
set -euo pipefail
cd /fact_data/xinyaowang/vla_vllm_profiling
if /opt/nvidia/nsight-compute/2026.1.1/ncu --target-processes all --profile-from-start off --nvtx --replay-mode range --cache-control none --clock-control none --metrics gpu__time_duration.sum,lts__t_bytes.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__ops_path_tensor_src_bf16_dst_fp32.sum --export results/processed/0913/08/range_probe/range bash scripts/thor_python.sh -u -m profiling.range_probe --output results/processed/0913/08/range_probe/workload.json > results/processed/0913/08/range_probe/ncu.log 2>&1; then
  /opt/nvidia/nsight-compute/2026.1.1/ncu --import results/processed/0913/08/range_probe/range.ncu-rep --page raw --csv --print-units base --print-fp > results/processed/0913/08/range_probe/raw.csv
else
  echo "Range probe failed; raw log retained. Baseline timing can proceed independently."
fi
bash scripts/thor_python.sh -u -m profiling.bench_pi05 --checkpoint /scratch/xinyaowang/vla-vllm-profiling/assets/pi05_base/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba --tokenizer /scratch/xinyaowang/vla-vllm-profiling/assets/paligemma_tokenizer --output results/processed/0913/08/workload --cameras 3 --steps 10 --warmup 3 --iterations 20 --scope all
