# Execution status — 2026-09-11

## Completed locally

- Cloned vLLM into the requested workspace.
- Fetched fixed π0.5 optimized and functional-reference vLLM-Omni revisions.
- Verified the model repository revision and availability using its public API;
  weights have not yet been downloaded.
- Implemented the real-checkpoint pipeline benchmark, isolated expert graph/direct
  measurements, NVTX attribution, target-specific metric discovery, NCU collection,
  CSV analysis and roofline plotting.
- Implemented scratch cache configuration, exact source reconstruction, a read-only
  Thor inventory and indicative roof calibration.
- All 10 CPU CSV/math tests pass; Python syntax and shell syntax checks pass.
  PNG/PDF rendering passed using disposable synthetic test input, which was
  removed after the check. No GPU runner has been executed.

## Current blocker

Thor is unavailable: the user reported that the server crashed. Hardware work is
paused until the server and SSH access are restored. This repository is a
preparation snapshot published at the user's request, before hardware results
exist. Available L20/H20 resources cannot stand in for a Thor measurement.

## Next steps after Thor recovers

1. Inspect Thor, its storage mounts, Slurm resources and NCU installation.
2. Determine compatible ARM/CUDA/PyTorch/Triton/vLLM versions; create a project
   environment and acquire the pinned checkpoint/tokenizer in scratch.
3. Present a concrete Slurm resource plan from live cluster information, respecting
   the applicable confirmation rules; then submit authorized jobs.
4. Run real-weight inference and numerical checks; fix compatibility defects.
5. Collect uninstrumented timing, NCU operator metrics and documented ceilings.
6. Inspect every category's counter coverage, draw measured rooflines and write a
   concise report with limitations and bottleneck candidates.
7. Add reviewed results and the measured environment to the `vla-vllm-profiling`
   repository; retain the original NCU reports with checksums.

## Claims not yet supported

No successful Thor deployment, latency value, speedup, NCU counter, kernel
bottleneck, roofline point or external numerical-parity result is claimed.
