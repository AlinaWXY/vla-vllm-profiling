# Measurement contract

## Three different timings

- **Pipeline wall time:** synchronized host timer around `Pi05Pipeline.forward()`;
  preprocessing, prefix, all denoising steps and output copy are included.
  Websocket transport, engine scheduling and robot execution are excluded.
- **Action Expert elapsed device time:** CUDA events around the isolated optimized
  decoder, with CUDA Graph enabled and disabled in separate measurements. The
  decoder uses real prefix KV from the warmed pipeline. Time/style preprocessing
  and vision/language prefix computation are outside this interval.
- **NCU duration:** `gpu__time_duration.sum` for each profiled launch. The first
  collection uses kernel replay and flushed caches. This changes cache behavior
  and can serialize kernels, so sums are not the original request latency.

Use a separate invocation without NCU for deployment latency. A request already
contains all configured denoising steps; do not multiply its timing by 10 again.

## What an operator means

The primary result is one measured CUDA/Triton kernel invocation. Fusion means an
operator such as QKV projection + RoPE or down projection + gated residual appears
as one point. It must not be split into fabricated timings for individual PyTorch
operations. Each Triton launch gets its denoising step, decoder layer, kernel name
and source line. Copies and metadata kernels remain visible under the outer scope.

Instrumentation is installed only for diagnostic direct launches, after the
production CUDA Graph is warmed. Measure latency and graph/direct parity in the
separate uninstrumented benchmark; profile mode skips those repeated sweeps and
the additional expert timing graph. The launch arguments and Triton
specializations are unchanged; graph launch overhead is different. The profile
must be labeled accordingly. Validate direct-launch vs decoder-graph action
agreement. A later graph-node collection may corroborate the mapping, but no such
result currently exists.

The export also pools repeated denoising steps/layers by labeled source line in
`operators.csv`. Its rates use total FLOPs / total time and total FLOPs / total
traffic, rather than an unweighted average of individual rates. Precision domains
remain separate. Both figures annotate operator IDs and semantic names directly;
the invocation figure labels operator clusters, while each dot remains an actual
invocation. `operator_legend.csv` maps IDs to displayed names, full source launch
sites and plotted domains. The three AdaRMS sites distinguish pre-attention,
pre-FFN and final normalization. Semantic roles are reviewed against the pinned
PR 4419 source; unknown sites retain their source names. Labels may move to avoid
overlap, but measured point coordinates do not move. Original invocations and
their full labels remain in `kernels.csv`.
`profiling.audit_profile` requires an exact multiset match between the instrumented
Triton manifest and NCU NVTX labels, and checks every requested metric. Its
`hotspots.csv` sums each launch duration once, including launches with zero counted
floating-point arithmetic; `coverage.json` retains unlabelled framework kernels.

## FLOPs and traffic

For a kernel with measured duration `t`, counted floating-point operations `F`
and measured traffic bytes `B` at the selected memory level (L2 on Thor):

```text
Arithmetic intensity = F / B                     [FLOP/byte]
Achieved performance = F / t / 1e12               [TFLOP/s]
Roof(I) = min(compute_TFLOPS, memory_GBPS * I / 1000)
```

BF16 Tensor operations are obtained from a supported hardware operations counter,
not from MMA instruction count times a guessed instruction width. Prefer a
single aggregate counter; never add it to its sparsity children. The selected
counter and exact formula are saved in `metrics.json`.
Executed hardware arithmetic may include padded matrix-tile work; these are
counted operations, not a claim of useful algorithm FLOPs.

FP32 SIMT arithmetic counts `FADD + FMUL + 2*FFMA`. BF16 Tensor and FP32 SIMT
are separate precision domains, even if a fused kernel does both.
Each precision row carries that kernel's full time and traffic; neither can be
summed across precision domains without counting the same launch twice.
Integer arithmetic, loads/stores and transcendental instructions such as exponentials are
not counted as ordinary floating-point add/mul/FMA work in these panels. A softmax
or normalization point therefore describes counted arithmetic, not all issued work.

Missing or unavailable metrics stay unknown. Kernels with zero traffic at the
selected level cannot be assigned a finite arithmetic intensity at that level.
Keep their timing in the table. Preserve all launches,
including repeated names; their cache states and shapes can differ.

## Thor conditions

Record device model, compute capability, CUDA/driver/PyTorch/Triton/NCU versions,
execution host (and Slurm allocation on other hosts), power mode and observed frequencies. Thor uses a different
memory/clock environment from datacenter GPUs. NCU does not support clock control
on Thor in the same way as desktop GPUs. The initial command uses
`--clock-control none`; no automatic system clock or power changes are made.

Empirical GEMM and streaming-copy rates are reference ceilings. They are not
guaranteed maxima and must be recorded under the same power state as the workload.
FP32 calibration disables TF32. Confirm the BF16 and FP32 instruction paths with
NCU before interpreting precision-specific roofs. Copy traffic is read+write bytes. DRAM streaming buffers exceed L2 capacity; the
explicit L2 calibration uses cache-resident buffers and bypasses L1.

## Correctness gates

1. Require `type=pi05` checkpoint and real safetensors; reject missing model weights.
2. Record shape and finiteness of `[50,32]` outputs for fixed input/noise.
3. Record safe vs optimized errors on the same PR. Finiteness alone is not parity.
4. Validate against LeRobot or the PR 6950 oracle in a compatible separate
   environment, with identical preprocessed inputs, weights, dtype and noise.
5. Compare graph and direct optimized decoder outputs before diagnostic profiling.

The PR 4419 pipeline blanket-casts its modules to BF16, while PR 6950 documents
LeRobot's mixed FP32/BF16 layout. Do not assume the two implementations are
numerically identical. Until the external oracle is run, report performance as
experimental and correctness as unverified beyond the observed gates.

## Primary references

- [NCU CLI: replay, NVTX, CSV exports and metric discovery](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [NCU Profiling Guide: roofline, replay/cache control and Thor clock limitations](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Experimental π0.5 optimized implementation, PR 4419](https://github.com/vllm-project/vllm-omni/pull/4419)
- [Functional π0.5 integration and numerical oracle, PR 6950](https://github.com/vllm-project/vllm-omni/pull/6950)
- [π0.5 checkpoint and license](https://huggingface.co/lerobot/pi05_base)


## Thor memory-level finding (2026-09-12)

The installed NCU 2026.1.1 metric inventory for NVIDIA Thor (GB10B) contains no
`dram__bytes*` counters. `lts__t_bytes.sum` is present and was collected on the
small deployment probe, along with timing and FP32 instruction counts. Therefore
Thor collection explicitly uses `--memory-level l2`; it does not substitute L2
bytes into a DRAM arithmetic intensity. The metric contract, output table, axis
label and bandwidth ceiling all retain the selected level. DRAM remains available
for targets that actually expose its counters.

For the L2 reference, `profiling.calibrate --memory-level l2` uses Triton copy
kernels with `.cg` loads to bypass L1 and `.wb` stores. The two buffers together
must fit in half the device-reported L2 cache (the deployment probe reported
32 MiB on this Thor). The achieved copy rate is an empirical reference, not an
absolute hardware peak. Historical provisional results remain in
`results/processed/environment_sm110a/`. The subsequent compute-path and L2
counter checks, power observations and empirical references are in
`results/processed/compute_counter_validation/`. Use that directory's
`ceilings_l2.json` for the real-model run. Historical nvidia-smi frequency telemetry
was unavailable and NCU clock control was disabled; the system's historical
frequency limits were not recorded, so exact frequency matching is not established.
The earlier wording "unlocked clocks" was too strong: on 2026-09-13 the read-only
sysfs observation found GPU GPC min=max=current=1.575 GHz. New collections save
these observations without changing any settings.

## Reading points below the reference

For precision domain d and memory level l, the plotted coordinates are
`P_d = F_d / t` and `AI_l,d = F_d / B_l`. The memory branch is
`BW_l * AI_l,d`. Below that branch, `P_d / roof = (B_l / t) / BW_l`:
this is the fraction of the chosen bandwidth reference achieved, not proof that
this memory level limits the kernel. Another level, dependency latency, too few
blocks or warps, resource limits, and non-counted work can all lower the point.

The current x-axis uses **L2 traffic only**. A DRAM or L1 slope cannot be drawn
against these same intensities and called a matching-level roof. This Thor's
inventory exposes no DRAM/FBPA counter bases. L2 read-miss sectors can diagnose
requests going beyond L2, but `32 * miss sectors` is not a measured DRAM byte count.

The reference compute kernels are large pure GEMMs. A fused BF16/FP32 kernel uses
the full elapsed time in both panels, with only that panel's arithmetic counted.
SFU, integer, reduction, synchronization and memory work are not represented by
the two arithmetic counts. Neither panel is expected to reach its pure GEMM roof.
Tensor FLOPs count executed arithmetic, including tile padding, rather than only
useful model arithmetic.

NCU kernel replay with cache flushing measures isolated cache-cold launches.
The `cache-control all/none` pair changes only flushing and tests sensitivity;
`none` with kernel replay is not application-managed warm-cache execution.
Application/range replay is needed to preserve that context across passes.
Short kernels and multi-pass ratios need extra care; do not label a low hit rate
alone as bandwidth saturation. See the [NCU profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control).

## Dated records

Use `python -m profiling.experiments --purpose '...'` to allocate
`results/processed/MMDD/NN`, with Asia/Hong_Kong dates and exclusive directory
creation. Each figure carries its experiment ID and full plot timestamp; a sidecar
records input hashes. Collection start/end timestamps and observed frequency
limits are separate. Historical data replots retain the original collection dates.
Plotting entry points refuse to overwrite prior images.


## Complete VLM, Action Expert and VLA (corrected Omni source)

The selected source is Omni `6bdbf97e357232b8aa3b6caf5a6c8b0fed713c25`.
The harness follows its corrected Gemma RoPE and per-camera output ownership.
`PI05_OMNI_SOURCE` selects an existing checkout without installing a package.
`profiling.bench_pi05 --scope all` checks all three effective image masks and
compares decomposed direct and captured-graph outputs with the original pipeline
in the same process before collecting measurements.

VLM includes three image encoders, text embedding, concatenation and prefix
Transformer; Action Expert includes action projection, denoising steps and
head/Euler update. The combined VLA scope runs both on the GPU. Preprocessing,
tokenization, output D2H copy and static timestep/AdaRMS preparation are excluded
from these GPU scopes. Independent pipeline wall timing includes preprocessing
and output copy, without websocket transport or serving scheduling.

Per-operator capture uses NCU kernel replay, graph node profiling and cache
control `all`; NVTX and a source-launch manifest map points to stages, layers,
steps, actual kernel names and tensor shapes. Aggregate operator rates use
`sum(F)/sum(t)` and `sum(F)/sum(bytes)`. These sums are not overall latency.

Thor NCU 2026.1.1 rejects FP32 SASS counters in range replay and rejects
`cuGraphLaunch` inside a captured replay range (experiments 17–19). Whole-stage
collection therefore uses `--graph-profiling graph --replay-mode kernel
--cache-control none`, with each complete CUDA Graph as one measured workload.
A known-work probe must verify Tensor counts before accepting full-model data.
Graph profiling retains dependencies and cache reuse between its nodes, unlike
isolated kernel replay; see NVIDIA's [graph profiling documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#graph-profiling).

Whole-graph time, L2 requested bytes and BF16 Tensor operations come from that
graph report. Instruction-level FP32 counters are unavailable in graph mode;
if an FP32 whole-stage point is shown, its numerator is explicitly the sum from
the matching per-kernel workload. Hardware BF16 totals must agree between both
collections before those records are joined. Input, source and output identity
are checked as well. Missing counters are never replaced with zero.

`profiling.full_roofline` renders VLM and expert operator plots and exact point
legends. `profiling.whole_roofline` renders VLM, expert and combined VLA overall
plots. Every plot remains below a new dated experiment directory.


The GPU scope graphs are newly captured from the same compiled vision callable
and prefix/expert kernels, preserving the image input copy and per-camera output
clone. PyTorch rejects nesting an existing graph replay inside a new capture
(experiment 26), so the vision callable is recaptured directly. This preserves
its compiled arithmetic but changes graph packaging. Original Pi05Pipeline wall
time remains a separate measurement. Experiment 27 verifies both direct and
recaptured scoped output equality (maximum absolute difference 0) in-process.

A slow NFS mapping was observed during startup. The benchmark can stream the
original F32 serialization with buffered I/O (`--buffered-original`) instead of
faulting the checkpoint's memory map tensor by tensor. Once the original file
was in the OS page cache, loading took about four seconds. This changes startup
I/O, not model values or timed GPU work. Actual parameters are fingerprinted
before profiling, and captures require the same fingerprint as the baseline.
A separate BF16 cache was prepared off Thor in experiment 25, but the final
27/28/29 runs use the original F32 checkpoint.

Long-running jobs use a frozen copy of the harness in their experiment directory.
`PYTHONSAFEPATH=1` prevents the working directory from shadowing that snapshot;
the benchmark checks its actual import path before loading the model. NCU raw
reports, exact NVTX labels and compiler-generated source are retained. The two
operator views use the same stage-local IDs: one pools repeated calls; the other
shows every invocation and labels its actual operator cluster. Zero-counted-FLOP
copy/conversion kernels remain in tables even though a logarithmic FLOP plot
cannot display zero.
