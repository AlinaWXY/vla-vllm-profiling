# Execution status — 2026-09-13

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
  removed after the check. The VLA runner has started, but model loading has not completed.
  The expanded 19-test CPU suite and operator-group PNG/PDF rendering also pass locally.

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

## New counter validation and loading status

`results/processed/compute_counter_validation/` contains complete NCU application
replay results. BF16 Tensor operations exactly match 2×8192³; FP32 SIMT operations
match 2×8192³+8192², including the output multiply, with zero BF16 Tensor operations.
A separate calibration reports 114.088 TFLOP/s BF16, 6.375 TFLOP/s FP32 and
954.716 GB/s logical L2 copy bandwidth. Cache-warm NCU L2 traffic differs from
logical copy bytes by 0.434%. These are empirical references, not hardware maxima.

The first real-weight run hit its 30-minute startup limit while streaming
checkpoint tensors over Thor's NFS mount. It did not reach GPU inference or
report an OOM. Memory samples retained over 110 GiB host headroom. A retry uses
an extended two-hour limit and reports loaded tensor/byte progress. Loading
and network transfer times are excluded from kernel/deployment latency claims.

## Remaining work

1. Complete real-checkpoint loading on Thor; local assets are now ready.
2. Run inference and numerical checks using one process/configuration.
3. Collect uninstrumented timing, NCU operator metrics and documented ceilings.
4. Review counter coverage and publish measured rooflines, commands and limits.

## Claims not yet supported

No successful pi0.5 deployment, VLA latency, speedup, Action Expert NCU result,
kernel bottleneck, VLA roofline point or external numerical-parity result is
claimed. The successful toy-kernel deployment and NCU access checks are separately
labeled environment validation.
