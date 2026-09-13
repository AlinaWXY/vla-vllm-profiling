# Thor reference validation — 2026-09-13

These are calibration and counter checks, not VLA inference measurements.
Both 8192 × 8192 dense GEMMs were run in the shared Torch 2.11/CUDA 13 environment.
All outputs matched the known result. The installed NCU 2026.1.1 collected three
application-replay passes, using the same six metrics as the Action Expert L2
contract.

| Check | Expected | NCU measured |
| --- | ---: | ---: |
| BF16 Tensor GEMM operations | 1,099,511,627,776 | 1,099,511,627,776 |
| FP32 SIMT GEMM + output multiply | 1,099,578,736,640 | 1,099,578,736,640 |
| BF16 Tensor operations in FP32 GEMM | 0 | 0 |

The FP32 kernel is `cutlass_80_simt_sgemm_256x128_8x4_nn_align1`; its output
stage adds N² multiplies beyond 2N³ GEMM FLOPs. The BF16 kernel is
`nvjet_sm110_tst_256x256_64x4_2x1_2cta_v_bz_NNT`.
The original `.ncu-rep`, raw CSV and SHA256 are included here.

A separate run without NCU measured these empirical references:

| Reference | Achieved median rate |
| --- | ---: |
| BF16 GEMM | 114.088 TFLOP/s |
| FP32 GEMM, TF32 disabled | 6.375 TFLOP/s |
| L2 copy | 954.716 GB/s |

`ceilings_l2.json` includes samples and methodology. Power and temperature were
queried before and after this calibration. The installed `nvidia-smi` reports
clock frequencies as N/A; clocks/power settings were unchanged and unlocked.
These achieved rates are references, not theoretical hardware peaks or a DRAM roof.

A separate cache-warm, single-pass NCU check measured 8,425,056 L2 bytes for the
4 MiB copy's 8,388,608 logical read+write bytes (0.434% more). Its original report,
CSV and `l2_validation.json` are included. The bandwidth reference uses logical
copy bytes; this small counter difference is not silently folded into the rate.

For reproduction, run `profiling.validate_compute_counters` under NCU with
`--replay-mode application --profile-from-start off --nvtx
--nvtx-include compute_reference/ --clock-control none --cache-control all` and
the metrics in `../environment_sm110a/l2_metrics.json`. Export using `--page raw
--csv --print-units base --print-fp`. Run `profiling.calibrate --memory-level l2`
separately for rates. Full-size FP32 software-instruction counting can take
minutes per pass; NCU's long-launch warning alone does not establish a hang.
