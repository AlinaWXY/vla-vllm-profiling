"""Verify range replay counts native BF16 GEMM and JIT FP32 work in one graph."""
import argparse
import json
from pathlib import Path

from profiling.experiments import timestamp
from profiling.ncu import require_gpu_execution
from profiling.runtime import require_headroom


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require_gpu_execution()
    require_headroom(4)
    started = timestamp()
    import torch
    from profiling.range_probe_kernel import fma_probe
    with torch.inference_mode():
        n, elements = 1024, 1024 * 1024
        a = torch.ones(n, n, device="cuda", dtype=torch.bfloat16)
        b, c = torch.ones_like(a), torch.empty_like(a)
        x = torch.ones(elements, device="cuda", dtype=torch.float32)
        y, z = torch.ones_like(x), torch.empty_like(x)
        def workload():
            torch.mm(a, b, out=c)
            fma_probe[((elements + 255) // 256,)](x, y, z, elements, 256)
        for _ in range(3):
            workload()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            workload()
        graph.replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        try:
            with torch.cuda.nvtx.range("known_work_graph"):
                graph.replay()
                torch.cuda.synchronize()
        finally:
            torch.cuda.cudart().cudaProfilerStop()
        assert bool((c == n).all()) and bool((z == 3).all())
    args.output.write_text(json.dumps({"started_at": started, "finished_at": timestamp(),
        "expected_bf16_tensor_flops": 2 * n**3, "expected_jit_fp32_fma_flops": 2 * elements,
        "all_outputs_correct": True, "scope": "One CUDA Graph: native BF16 GEMM plus JIT FP32 FMA"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
