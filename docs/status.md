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
- All 16 CPU tests pass on Thor's existing Python 3.12, including CSV/math,
  offline assets, cache preservation, memory guards and the Thor execution rule.
  Python syntax and shell syntax checks pass.
  PNG/PDF rendering passed using disposable synthetic test input, which was
  removed after the check. No GPU runner has been executed.

## Current blocker

Thor SSH is restored. CUDA 13.2 and NCU 2026.1.1 are already installed, and shared
framework sources are readable. The default Python has no PyTorch/vLLM package
metadata; the known shared `chia_env` is x86-64, incompatible with Thor's aarch64
CPU. Docker inventory is inaccessible to the current user; sudo is unavailable.
An existing, accessible ARM runtime and local model/tokenizer paths remain to be
identified. No dependency install, model download or GPU workload has been run.

## Next steps using the existing environment

1. Locate an existing accessible aarch64 Python/PyTorch/vLLM environment and local
   checkpoint/tokenizer. Do not install or upgrade dependencies or download assets.
2. Validate existing versions against the pinned experimental source; retain any
   incompatibility as a finding instead of automatically replacing packages.
3. Run directly on Thor, as explicitly authorized on 2026-09-12; other hosts still
   require Slurm. Keep the initial workload to one process and one configuration.
4. Run real-weight inference and numerical checks; fix compatibility defects.
5. Collect uninstrumented timing, NCU operator metrics and documented ceilings.
6. Inspect every category's counter coverage, draw measured rooflines and write a
   concise report with limitations and bottleneck candidates.
7. Add reviewed results and the measured environment to the `vla-vllm-profiling`
   repository; retain the original NCU reports with checksums.

## Claims not yet supported

No successful Thor deployment, latency value, speedup, NCU counter, kernel
bottleneck, roofline point or external numerical-parity result is claimed.
