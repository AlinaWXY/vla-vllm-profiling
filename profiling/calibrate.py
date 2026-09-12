"""Measure indicative roofline reference ceilings; direct on Thor, Slurm elsewhere.

These are achieved GEMM and streaming-copy rates, NOT claims of absolute hardware
peak. Repeat under matching power/clocks. Cache-flushed NCU kernel replay can have
a different memory state from these streaming copy measurements.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics

from profiling.ncu import require_gpu_execution
from profiling.runtime import require_headroom


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--matrix-size", type=int, default=8192)
    p.add_argument("--copy-mib", type=int, help="Default 512 for DRAM or 4 for L2")
    p.add_argument("--memory-level", choices=("dram", "l2"), default="dram")
    p.add_argument("--power-clock-note", required=True,
                   help="Observed power mode/clock state and how it was recorded")
    args = p.parse_args()
    require_gpu_execution()
    require_headroom(4)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.copy_mib is None:
        args.copy_mib = 4 if args.memory_level == "l2" else 512
    if min(args.iterations, args.matrix_size, args.copy_mib) < 1:
        p.error("sizes and iteration count must be positive")
    import torch
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    l2_bytes = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0))
    if args.memory_level == "l2" and (l2_bytes <= 0 or 2 * args.copy_mib * 1024**2 > l2_bytes // 2):
        raise ValueError("L2 copy buffers together must fit within half of the reported L2 cache.")

    def measure(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        durations = []
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(args.iterations):
            start.record()
            fn()
            end.record()
            end.synchronize()
            durations.append(start.elapsed_time(end) / 1000)
        return durations

    compute, samples = {}, {}
    with torch.inference_mode():
        for name, dtype in (("bf16_tensor", torch.bfloat16), ("fp32_simt", torch.float32)):
            n = args.matrix_size
            a = torch.randn((n, n), device="cuda", dtype=dtype)
            b = torch.randn_like(a)
            c = torch.empty_like(a)
            durations = measure(lambda: torch.mm(a, b, out=c))
            compute[name] = 2 * n**3 / statistics.median(durations) / 1e12
            samples[name] = durations
            del a, b, c
        byte_count = args.copy_mib * 1024**2
        a = torch.empty(byte_count, device="cuda", dtype=torch.uint8).fill_(17)
        b = torch.empty_like(a)
        if args.memory_level == "l2":
            from profiling.memory_kernels import copy_l2
            def copy():
                copy_l2[((byte_count + 1023) // 1024,)](a, b, byte_count, 1024)
            copy()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(32):
                    copy()
            durations = [duration / 32 for duration in measure(graph.replay)]
            assert bool((b == 17).all()), "L2 copy correctness check failed"
            del graph
        else:
            durations = measure(lambda: b.copy_(a))
        bandwidth = 2 * byte_count / statistics.median(durations) / 1e9
        samples["copy"] = durations
    output = {
        "title": "π0.5 Action Expert — empirical reference ceilings",
        "label": f"Measured GEMM / {args.memory_level.upper()} copy reference",
        "compute_tflops": compute, "memory_level": args.memory_level,
        f"{args.memory_level}_bandwidth_gbps": bandwidth,
        "provenance": {"host": platform.node(), "gpu": torch.cuda.get_device_name(),
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                       "power_clock_note": args.power_clock_note,
                       "matrix_size": args.matrix_size, "copy_buffer_mib": args.copy_mib,
                       "l2_cache_bytes": l2_bytes,
                       "copy_method": "32-copy CUDA Graph, Triton .cg loads / .wb stores, cache-resident buffers; duration divided by 32" if args.memory_level == "l2" else "Torch D2D copy, streaming buffers",
                       "torch": torch.__version__, "cuda": torch.version.cuda,
                       "method": "Median CUDA-event elapsed time after 5 warmups; dense GEMM, TF32 disabled; copy traffic = read + write",
                       "limitations": "Indicative achieved ceilings; confirm kernel instruction path with NCU. Not theoretical device limits."},
        "samples_seconds": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
