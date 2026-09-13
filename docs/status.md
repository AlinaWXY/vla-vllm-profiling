# Execution status — 2026-09-13

## 本轮 NCU 已取消（2026-09-13T20:20:50+08:00）

用户要求停止耗时过长的本轮实验。[0913/28](../results/processed/0913/28/README.md)
逐算子采集在取消前已报错退出，保留部分日志但不作为完整结果；
[0913/29](../results/processed/0913/29/README.md) 整体采集未启动，已取消排队。
已核实 Thor 无残留 GPU 任务，不自动重试。
[0913/27](../results/processed/0913/27/README.md) 的有效基线保留：VLM GPU p50 69.054 ms、
Action Expert GPU p50 41.949 ms、VLA GPU p50 111.005 ms、原始 pipeline wall p50 113.410 ms。
新框架的完整 roofline 图尚未产生；下文采集计划仅作为历史记录。

## Completed locally

- Cloned vLLM into the requested workspace.
- Fetched fixed π0.5 optimized and functional-reference vLLM-Omni revisions.
- Registered vLLM and both Omni revisions as Git submodules, visible in the
  repository root and available through a recursive clone.
- Downloaded all 14,467,165,872 checkpoint bytes off Thor and verified the pinned SHA256.
  All 812 tensors are F32, totaling 3,616,757,520 parameters.
- Downloaded the official public OpenPI tokenizer, verified its identity with the
  pinned Google SentencePiece file, and passed 14 token-ID/mask/padding cases on Thor.
- Implemented the real-checkpoint pipeline benchmark, isolated expert graph/direct
  measurements, NVTX attribution, target-specific metric discovery, NCU collection,
  CSV analysis and roofline plotting.
- Implemented scratch cache configuration, exact source reconstruction, a read-only
  Thor inventory and indicative roof calibration.
- The prior 18 CPU tests passed on Thor's existing Python 3.12, including CSV/math,
  offline assets, cache preservation, memory guards and the Thor execution rule.
  Python syntax and shell syntax checks pass.
  PNG/PDF rendering passed using disposable synthetic test input, which was
  removed after the check. Real-weight inference subsequently completed on Thor.
  The expanded 22-test CPU suite, including exact launch-coverage and
  precision-duration de-duplication checks, and operator-group rendering pass locally.

## Shared ARM64 runtime ready

The user explicitly authorized off-host ARM64 environment preparation for Thor.
`deep-space` prepared 209 isolated Python 3.12 ARM64 packages in the shared
project directory. Version and native ELF architecture audits passed. Thor
validated BF16 matmul, a Triton add with `.target sm_110a`, native vLLM extension
loading, and the optimized Pi05Pipeline / Pi05RealtimeTritonDecoder imports. No packages are installed or upgraded on Thor.
The vLLM source pin is now v0.22.0, matching the optimized Omni Docker base.

The complete runtime validation is recorded in `runtime.json`; the standalone
L2 calibration entry point also ran successfully and produced provisional
reference rates in `ceilings_l2.json`. Its pre-run clock query was unavailable,
so these values are not final model-run ceilings.

An x86-built `sm_110a` cubin has passed on Thor: all 1024 float outputs matched,
with 4096 bytes of explicit device data allocation. Build and execution records
are in `results/processed/environment_sm110a/`. This is a deployment probe, not
pi0.5 inference or a performance measurement.

## Counter validation and real-model results

`results/processed/compute_counter_validation/` contains complete NCU application
replay results. BF16 Tensor operations exactly match 2×8192³; FP32 SIMT operations
match 2×8192³+8192², including the output multiply, with zero BF16 Tensor operations.
A separate calibration reports 114.088 TFLOP/s BF16, 6.375 TFLOP/s FP32 and
954.716 GB/s logical L2 copy bandwidth. Cache-warm NCU L2 traffic differs from
logical copy bytes by 0.434%. These are empirical references, not hardware maxima.

The first real-weight run hit its 30-minute startup limit while streaming
checkpoint tensors over Thor's NFS mount. It did not reach GPU inference or
report an OOM. Memory samples retained over 110 GiB host headroom. A retry uses
an extended two-hour limit and completed all weights, warmup and measurement.
All 813 parameter names are initialized (812 checkpoint tensors plus a tied alias),
with no missing parameters. Loading and transfer times are excluded from latency.

The historical real-weight run provided three cameras but masked them because of the key bug; its batch-one, ten-step workload records 20 timed samples:
safe pipeline p50 237.790 ms, optimized pipeline p50 72.456 ms, optimized expert
CUDA Graph p50 34.725 ms and direct-launch expert p50 35.594 ms. Graph/direct
expert outputs agree exactly for this input. Peak Torch allocated memory is
14.48 GiB; the sampled minimum host headroom is 99.11 GiB.

**Large numerical discrepancy:** safe versus optimized max absolute error is
0.191869 and relative RMSE is 1.03704. These normalized-space synthetic-input
outputs are not a robot success test. The historical run masked all three cameras
because its observation keys did not match the processor. The numerical causes
are now investigated below; the latency ratio is not a validated equivalent-model speedup. Outputs, individual timing
samples and memory observations are in `results/processed/thor_pi05_20260913/`.

NCU completed 1,654 kernel invocations: 1,650 instrumented Triton launches exactly
match the source manifest, plus four framework fill/copy/conversion kernels.
All ten steps have 165 Triton launches, and all 18 decoder layers are covered.
All six requested counters are present. The 16 operator groups, raw and annotated
NCU CSVs, original 35,045,925-byte report and PNG/PDF figures are published in the
same result directory. Baseline and NCU workload optimized outputs agree exactly.

The NCU duration sum is 57.699 ms, separate from uninstrumented expert latency.
FFN gate/up and down account for 49.114% of this replay duration. L2 traffic totals
17.822 GB; counted operations are 417.155 GFLOP BF16 Tensor and 1.251 GFLOP FP32 SIMT.
The matching empirical L2 roof is explicitly labeled; no DRAM roof is inferred.
NCU-phase memory sampling retained at least 79.65 GiB host headroom.

## Complete-model numerical investigation on L20

Slurm job 1490549 completed successfully on 2026-09-13. Experiment
[0913/15](../results/processed/0913/15/README.md) compares complete
`Pi05Pipeline.forward` outputs, including vision, prefix, ten denoising steps
and the production CUDA Graph paths. With three valid cameras, original relative
RMSE is 166.084%; correcting decoder RoPE layout and matching the model frequency
reduces it to 3.113%. Saving each camera's graph output before the next replay
reduces it to 1.007%. A control with eager image encoding gives 0.888%.

Two semantic bugs are confirmed: decoder RoPE pairs adjacent dimensions instead
of Gemma's two halves, and same-shaped camera calls return a shared graph output
that is overwritten before prefix concatenation. Full source evidence and control
results are in [findings](../results/processed/0913/15/findings.md). All 813 parameter
names loaded; safe repeats and optimized graph repeats agree exactly for these
inputs. Corrections are process-local diagnostics, not changes to pinned upstream
source. Residuals and the lack of external-oracle validation remain explicit.

The corrected framework commit `6bdbf97e357232b8aa3b6caf5a6c8b0fed713c25`
was subsequently tested without numerical substitutions in Slurm job 1490577,
[0913/21](../results/processed/0913/21/README.md). Three-camera final-action
relative RMSE is 1.0047%. Actual source and a same-process corrected original
decoder control agree exactly. Three framework regression tests and 24 project
CPU tests pass. Cross-process diagnostic outputs have a small recorded difference;
the source regression does not claim cross-process bitwise reproducibility.

## Follow-up beyond this initial profiling result

1. Validate remaining numerical residuals against an external oracle.
2. If optimizing further, remeasure after numerical validation and use additional
   utilization/stall counters to investigate the measured FFN hotspots.

## Claims not yet supported

No numerically equivalent speedup, definitive DRAM bottleneck, robot success rate
or external numerical-parity result is claimed. Measured hotspots and L2 rooflines
describe this experimental implementation under the documented replay conditions.
Environment probes and calibration remain distinct from real-model measurements.


## Corrected source performance and complete scopes

Experiment [0913/27](../results/processed/0913/27/README.md) uses actual corrected
Omni 6bdbf97 and three valid cameras (prefix 918). Twenty samples give GPU
scope-graph p50 values of 69.054 ms VLM, 41.949 ms Action Expert and 111.005 ms
combined VLA. Original complete pipeline wall p50 is 113.410 ms. Both direct
scoped and recaptured-graph outputs match the original pipeline exactly for this
input. All 813 actual parameter names are loaded and fingerprinted.

Whole-graph NCU hardware counters passed the known-work probe in 0913/24.
Per-kernel and whole-graph model captures are running in 0913/28 and 0913/29.
Original-checkpoint buffered loading took about four seconds once the OS page
cache was warm; this is a startup I/O observation, not a GPU inference speedup.
No new packages were installed on Thor.
