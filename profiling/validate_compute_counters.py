"""NCU workload to verify the instruction domains of the calibrated GEMMs."""
import argparse
import json
from pathlib import Path

from profiling.ncu import require_gpu_execution
from profiling.runtime import require_headroom


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--matrix-size", type=int, default=8192)
    p.add_argument("--case", choices=("compute", "l2_copy"), default="compute")
    args = p.parse_args()
    require_gpu_execution()
    require_headroom(4)
    # NCU application replay repeats this exact deterministic workload. Permit
    # rewriting its own report, but never replace a report for a different case.
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("case", "compute") != args.case or any(
            case.get("matrix_size", args.matrix_size) != args.matrix_size for case in previous["cases"]
        ):
            raise FileExistsError(args.output)
    if args.matrix_size < 1:
        p.error("matrix-size must be positive")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    n = args.matrix_size
    cases = []
    with torch.inference_mode():
        for name, dtype in ((("bf16_tensor", torch.bfloat16), ("fp32_simt", torch.float32))
                            if args.case == "compute" else ()):
            a = torch.ones((n, n), dtype=dtype, device="cuda")
            b = torch.ones_like(a)
            c = torch.empty_like(a)
            for _ in range(5):
                torch.mm(a, b, out=c)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("compute_reference")
            torch.cuda.nvtx.range_push(name)
            try:
                torch.mm(a, b, out=c)
                torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()
                torch.cuda.cudart().cudaProfilerStop()
            assert bool((c == n).all()), f"{name} GEMM output mismatch"
            cases.append({"domain": name, "expected_matmul_flops": 2 * n**3, "matrix_size": n,
                          "output_correct": True})
            del a, b, c
        if args.case == "l2_copy":
            from profiling.memory_kernels import copy_l2
            byte_count = 4 * 1024**2
            if 2 * byte_count > torch.cuda.get_device_properties(0).L2_cache_size // 2:
                raise ValueError("Copy buffers exceed half of L2")
            a = torch.full((byte_count,), 17, dtype=torch.uint8, device="cuda")
            b = torch.empty_like(a)
            def copy():
                copy_l2[((byte_count + 1023) // 1024,)](a, b, byte_count, 1024)
            for _ in range(5):
                copy()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("memory_reference")
            try:
                copy()
                torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
                torch.cuda.cudart().cudaProfilerStop()
            assert bool((b == 17).all()), "L2 copy output mismatch"
            cases.append({"domain": "l2_copy", "expected_l2_bytes": 2 * byte_count,
                          "buffer_bytes": byte_count, "output_correct": True})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"case": args.case, "cases": cases, "torch": torch.__version__,
        "purpose": "Match NCU counters to known GEMM work or L2 traffic; NCU timing is not a ceiling measurement"},
        indent=2) + "\n")
    print("Compute reference output checks passed", flush=True)


if __name__ == "__main__":
    main()
