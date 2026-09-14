# Measurement and interpretation

The retained results cover VLM and Action Expert major operators on one NVIDIA L20. Both jobs used batch 1, three valid synthetic camera images, prefix length 918, ten denoising steps and the same pinned π0.5 base checkpoint.

For each group, arithmetic intensity is `Σ BF16 FLOPs / Σ DRAM bytes`; achieved throughput is `Σ BF16 FLOPs / Σ NCU kernel time`. Repeated layers and denoising steps are pooled by summing counts, bytes and duration. Utilization is weighted by kernel duration. Every invocation is counted once. Unavailable metrics remain missing rather than becoming zero.

The BF16 hardware operation counter was validated against a dense `8192³` GEMM: its count equals `2 × 8192³` FLOPs. FP32 add, multiply and twice-FMA counts are separate; transcendental, integer and FP64 time-encoding work are not included in that FP32 count. BF16 input and FP32 accumulation describe the Tensor math in the main figure, not every model parameter.

The solid roof uses [L20 nominal specifications](https://www.hpe.com/es/es/collaterals/collateral.c04123180.html): 119 TFLOP/s BF16 and 864 GB/s DRAM. The ridge is 137.73 FLOP/byte. Dashed roofs use each job's GEMM/copy reference; these are indicative achieved rates, not hard upper bounds.

“Memory side” and “compute side” identify the tighter ideal roofline constraint. Filled markers additionally require Tensor active ≥75% on the compute side or DRAM utilization ≥70% on the memory side. These are reporting heuristics, not a causal sensitivity experiment. Hollow points do not establish saturation. See the [NVIDIA Roofline guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#roofline-charts).

## Operator boundaries

QKV and output projections are separate. FFN up includes gate/up and activation; FFN down is separate. VLM projection/FFN groups combine vision and prefix work. Vision attention is separate from prefix attention.

Omni exposes separate prefix/expert QK and PV kernels; its vision attention is fused. SGLang uses fused QK + softmax + PV attention. Measured duration and DRAM traffic of a fused kernel cannot be uniquely divided among its mathematical stages. SGLang's standalone residual operations remain in the appendix, whereas some Omni down projections include fused residual work.

## Collection and comparison limits

Both captures used NCU application replay, `cache-control=all` and unlocked clocks. Replay passes matched model parameters, inputs, outputs and launch manifests; Omni additionally froze compiled launch choices. Original collection times are preserved. The decimal-axis figures were plotted later on 2026-09-14, with that time separately printed.

NCU changes cache and scheduling conditions. Per-kernel durations cannot substitute for independent native CUDA Graph or synchronized wall latency. SGLang's NCU dispatch is eager to preserve NVTX attribution; the underlying fused kernels are upstream implementations.

The frameworks use different software versions and fusion boundaries. SGLang runs 18 prefix attention layers while Omni only needs KV from the last prefix layer. SGLang's timed denoising includes time MLP/conditioning projections, whereas Omni prepares static timestep/AdaRMS conditions outside its GPU scopes. Cross-framework numerical equivalence has not been established. Results describe each measured implementation, not an equivalent-model speedup or robot task success rate.
