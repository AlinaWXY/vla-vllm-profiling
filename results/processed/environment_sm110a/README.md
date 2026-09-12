# Environment validation, not VLA results

These records validate an ARM64 runtime prepared on `deep-space` and executed on
NVIDIA Thor. No pi0.5 checkpoint was loaded.

- `build.json`, `probe.cubin`, `probe.json`: the off-host PTXAS 13.2.86 `sm_110a`
  build and its 1024-output correctness check on Thor.
- `ncu_probe.csv`, `ncu_summary.json`: actual NCU timing/instruction collection on
  that toy kernel. This CSV also exercises the wide-table parser.
- `ncu_probe_l2.csv`, `l2_metrics.json`, `counter_availability.json`: explicit L2
  and precision counters. No DRAM traffic counters were exposed on this device.
- `packages.json`: 209 package versions/tags and ARM64 native-extension audit.
- `runtime.json`: successful BF16 matmul, Triton add (`.target sm_110a`), vLLM
  native-extension import and optimized pi0.5 entry-point imports on Thor.
- `ceilings_l2.json`: provisional empirical references from dense 8192x8192 GEMM
  and a cache-resident L2 copy. BF16: 112.386 TFLOP/s; FP32 with TF32 disabled:
  6.375 TFLOP/s; L2 copy: 955.477 GB/s. These are achieved rates, not theoretical
  maxima. Kernel instruction-path validation remains necessary before treating
  the compute-domain labels as definitive ceilings.

Clocks/power settings were not changed. The pre-calibration clock query timed out;
that missing observation is explicit in `power_before_calibration.txt`. A post-run
snapshot is in `power_after_calibration.txt`. Recalibrate with clock observations
around the actual model workload before drawing final bottleneck conclusions.

The two small original `.ncu-rep` files are included here, with checksums in
`ncu_summary.json`; working copies remain under `results/raw/`. No plot of this toy kernel is presented as an Action Expert
roofline.
