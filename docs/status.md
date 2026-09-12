# Execution status — 2026-09-12

## Completed locally

- Cloned vLLM into the requested workspace.
- Fetched fixed π0.5 optimized and functional-reference vLLM-Omni revisions.
- Registered vLLM and both Omni revisions as Git submodules, visible in the
  repository root and available through a recursive clone.
- Verified the model repository revision and availability using its public API;
  weights have not yet been downloaded.
- Implemented the real-checkpoint pipeline benchmark, isolated expert graph/direct
  measurements, NVTX attribution, target-specific metric discovery, NCU collection,
  CSV analysis and roofline plotting.
- Implemented scratch cache configuration, exact source reconstruction, a read-only
  Thor inventory and indicative roof calibration.
- All 18 CPU tests pass on Thor's existing Python 3.12, including CSV/math,
  offline assets, cache preservation, memory guards and the Thor execution rule.
  Python syntax and shell syntax checks pass.
  PNG/PDF rendering passed using disposable synthetic test input, which was
  removed after the check. The VLA GPU runner has not been executed.

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

## Remaining work

1. Identify existing local checkpoint/tokenizer files; runtime loading is offline.
2. Run real-weight inference and numerical checks using one process/configuration.
3. Collect uninstrumented timing, NCU operator metrics and documented ceilings.
4. Review counter coverage and publish measured rooflines, commands and limits.

## Claims not yet supported

No successful pi0.5 deployment, VLA latency, speedup, Action Expert NCU result,
kernel bottleneck, VLA roofline point or external numerical-parity result is
claimed. The successful toy-kernel deployment and NCU access checks are separately
labeled environment validation.
