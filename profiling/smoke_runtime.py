"""Small Thor runtime checks without downloading assets or loading a VLA model."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import time

from profiling.ncu import require_gpu_execution
from profiling.runtime import require_headroom


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require_gpu_execution()
    require_headroom(2)
    if args.output.exists():
        raise FileExistsError(args.output)
    print("Importing Torch from the shared ARM64 environment...", flush=True)
    import torch
    print("Torch imported:", torch.__version__, flush=True)
    import triton
    import triton.language as tl
    from triton.backends.nvidia.compiler import get_ptxas

    @triton.jit
    def add_one(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(y + i, tl.load(x + i, i < N) + 1.0, i < N)

    assert torch.cuda.get_device_capability() == (11, 0), "Expected Thor SM 11.0"
    print("Torch CUDA device:", torch.cuda.get_device_name(), flush=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        a = torch.ones((128, 128), dtype=torch.bfloat16, device="cuda")
        c = a @ a
        assert bool((c == 128).all())
        x = torch.arange(1024, dtype=torch.float32, device="cuda")
        y = torch.empty_like(x)
        kernel = add_one[(4,)](x, y, 1024, 256)
        torch.cuda.synchronize()
        assert torch.equal(y.cpu(), torch.arange(1, 1025, dtype=torch.float32))
    print("BF16 matmul and Triton add passed", flush=True)
    # Import native vLLM ops and the actual optimized pi0.5 entry points.
    import vllm
    import vllm._C
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline
    from vllm_omni.diffusion.models.pi05.realtime_triton import Pi05RealtimeTritonDecoder
    report = {
        "status": "passed", "timestamp_unix": time.time(), "vla_benchmark": False,
        "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "torch_compiled_architectures": torch.cuda.get_arch_list(),
        "triton": triton.__version__, "ptxas": get_ptxas(110).path,
        "triton_ptx_target": re.search(r"^\.target .+$", kernel.asm["ptx"], re.M).group(0),
        "vllm": vllm.__version__, "vllm_path": vllm.__file__,
        "transformers": importlib.metadata.version("transformers"),
        "pi05_imports": [Pi05Pipeline.__name__, Pi05RealtimeTritonDecoder.__name__],
        "checks": ["128x128 BF16 matmul", "1024-value Triton add", "vLLM native extension import", "optimized pi0.5 imports"],
        "max_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "limitation": "No checkpoint loaded; no VLA latency or correctness claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
