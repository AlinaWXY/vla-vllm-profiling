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

## FLOPs and traffic

For a kernel with measured duration `t`, counted floating-point operations `F`
and total DRAM read+write bytes `B`:

```text
Arithmetic intensity = F / B                     [FLOP/byte]
Achieved performance = F / t / 1e12               [TFLOP/s]
Roof(I) = min(compute_TFLOPS, memory_GBPS * I / 1000)
```

BF16 Tensor operations are obtained from a supported hardware operations counter,
not from MMA instruction count times a guessed instruction width. Prefer a
single aggregate counter; never add it to its sparsity children. The selected
counter and exact formula are saved in `metrics.json`.

FP32 SIMT arithmetic counts `FADD + FMUL + 2*FFMA`. BF16 Tensor and FP32 SIMT
are separate precision domains, even if a fused kernel does both. Integer index
arithmetic, loads/stores and transcendental instructions such as exponentials are
not counted as ordinary floating-point add/mul/FMA work in these panels. A softmax
or normalization point therefore describes counted arithmetic, not all issued work.

Missing or unavailable metrics stay unknown. Kernels with no DRAM traffic can be
cache resident and cannot be assigned a finite DRAM arithmetic intensity. Keep
their timing in the table and consider a future L2 roofline. Preserve all launches,
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
NCU before interpreting precision-specific roofs. The first copy calibration's
traffic is read+write bytes and its buffers should exceed the target L2 capacity.

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
absolute hardware peak. This calibration ran successfully in the shared runtime. Its provisional results
and missing pre-run clock observation are recorded in
`results/processed/environment_sm110a/`; final model-run ceilings need matching
clock observations and instruction-path checks.
