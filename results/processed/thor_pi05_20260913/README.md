# Thor π0.5 real-weight measurements — 2026-09-13

This is a measurement of the pinned experimental vLLM-Omni PR 4419 implementation.
**Safe/optimized numerical equivalence is not established: relative RMSE is
103.7%. Do not interpret the latency ratio as a validated equivalent-model speedup.**
Action Expert NCU collection is in progress; a model roofline is not yet included.

## Workload and latency

Real `lerobot/pi05_base` weights, revision
`b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`; BF16 blanket cast, batch 1,
three synthetic 224×224 RGB cameras, zero state, fixed noise/seed 17,
ten denoising steps, output `[50,32]`. Cross-request prefix caches are disabled.
All 813 expected parameter names loaded; no missing parameters.

Each row uses 3 warmups and 20 measurements without NCU. Pipeline time includes
preprocessing, prefix computation, denoising and output copy, excluding service
transport/scheduling. Expert device time excludes prefix and AdaRMS precomputation.

| Measurement | p50 (ms) | p95 (ms) |
| --- | ---: | ---: |
| Safe pipeline wall time | 237.790 | 241.429 |
| Optimized pipeline wall time | 72.456 | 73.762 |
| Optimized expert, CUDA Graph | 34.725 | 34.811 |
| Optimized expert, direct launches | 35.594 | 35.824 |

`benchmark.json` preserves all individual samples, versions and configuration.
`actions.npz` preserves safe/optimized normalized-space actions and input noise.
`numerical_difference.json` is computed from those saved arrays: maximum absolute
error 0.191869, mean absolute error 0.043668, error RMS 0.059375, safe RMS 0.057254.
Expert direct/graph maximum absolute difference was 0.0. This last check only
validates graph replay of the same experimental decoder, not model correctness.
The safe/optimized difference has not been isolated to an individual component.
No external LeRobot oracle or robot task evaluation has been run.

## Memory and platform

NVIDIA Thor GB10B, compute capability 11.0, shared ARM64 runtime, Torch 2.11.0+cu130,
vLLM 0.22.0, Triton 3.6.0, Transformers 5.8.1. Existing CUDA 13.2 `ptxas`
is used for `sm_110a` JIT. No environment packages were installed on Thor.

Torch peak allocated: 15,544,240,640 bytes (14.48 GiB); peak reserved:
15,730,737,152 bytes. Read-only 10-second samples show process HWM 31.66 GiB
and minimum host available memory 99.11 GiB. RSS and CUDA allocations overlap on
UMA and must not be added. Sampling is not a hard memory bound.
Power/clock queries are retained; N/A fields remain unknown and clocks were not locked.

The first startup attempt timed out during NFS weight loading. The completed
retry allowed two hours; loading time is excluded from all latency numbers.

## Reproduce

On Thor, from the project directory, with the shared runtime and downloaded assets:

```bash
bash scripts/thor_python.sh -u -m profiling.bench_pi05 \
  --checkpoint /scratch/xinyaowang/vla-vllm-profiling/assets/pi05_base/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba \
  --tokenizer /scratch/xinyaowang/vla-vllm-profiling/assets/paligemma_tokenizer \
  --output results/raw/new_baseline --cameras 3 --steps 10 --warmup 3 --iterations 20
```

See the root README for source reconstruction, asset download, NCU commands and
measurement definitions. Use a new output directory for each run.

Recompute numerical errors from this directory's saved arrays:

```python
import numpy as np
a = np.load("results/processed/thor_pi05_20260913/actions.npz")
d = a["optimized"] - a["safe"]
print("max_abs", np.abs(d).max(), "mean_abs", np.abs(d).mean())
print("relative_rmse", np.sqrt(np.mean(d**2)) / np.sqrt(np.mean(a["safe"]**2)))
```
