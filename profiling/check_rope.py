"""Numerical repro for the decoder's adjacent-pair versus Gemma half-split RoPE."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

from profiling.experiments import timestamp, file_record, clock_snapshot
from profiling.ncu import require_gpu_execution
from profiling.runtime import require_headroom


def load_kernels(path):
    spec = importlib.util.spec_from_file_location("pi05_checked_kernels", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def error(actual, reference):
    actual, reference = actual.float(), reference.float()
    diff = actual - reference
    rms = float(reference.square().mean().sqrt())
    return {"max_abs": float(diff.abs().max()), "mean_abs": float(diff.abs().mean()),
            "rmse": float(diff.square().mean().sqrt()), "reference_rms": rms,
            "relative_rmse": float(diff.square().mean().sqrt()) / rms if rms else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require_gpu_execution()
    require_headroom(4)
    if (args.output / "rope_check.json").exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    started = timestamp()
    print(started, "Importing Torch/Triton for numerical check only", flush=True)
    import torch
    source = Path("vllm-omni/vllm_omni/diffusion/models/pi05/realtime_triton.py")
    kernels = load_kernels(source)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    with torch.inference_mode():
        n, features, dim, heads = 50, 1024, 256, 8
        x = torch.randn(n, features, dtype=torch.bfloat16, device="cuda")
        w = (torch.randn(features, (heads + 2) * dim, device="cuda") / math.sqrt(features)).bfloat16()
        qkv = torch.mm(x, w)
        raw_q, raw_k, ref_v = qkv.split((heads * dim, dim, dim), dim=-1)
        raw_q = raw_q.reshape(n, heads, dim)
        for base in (0, 150, 918):
            pos = (torch.zeros(n, device="cuda") if base == 0 else
                   torch.arange(base, base + n, device="cuda", dtype=torch.float32))
            inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim))
            phase = pos[:, None] * inv[None, :]
            cos, sin = torch.cat([phase.cos()] * 2, dim=-1).bfloat16(), torch.cat([phase.sin()] * 2, dim=-1).bfloat16()
            table = torch.stack((phase.cos(), phase.sin()), dim=-1).reshape(n, dim).bfloat16()
            def rotate_half(v):
                return torch.cat((-v[..., dim//2:], v[..., :dim//2]), dim=-1)
            # Same eager BF16 arithmetic and half-split layout as installed Gemma.
            ref_q = raw_q * cos[:, None, :] + rotate_half(raw_q) * sin[:, None, :]
            ref_k = raw_k * cos + rotate_half(raw_k) * sin
            q, k, v = torch.empty_like(raw_q), torch.empty_like(raw_k), torch.empty_like(ref_v)
            kernels._matmul_rope_qkv[(128,)](x, n, features, dim, heads, w, table, q, k, v,
                                             block_m=32, block_n=64, block_k=128)
            torch.cuda.synchronize()
            legacy = {name: error(a, b) for name, a, b in (("q",q,ref_q),("k",k,ref_k),("v",v,ref_v))}
            # Existing prefix kernel already implements the Gemma half-split convention.
            kernels._matmul_gemma_rope_qkv[((n + 31)//32, heads + 2)](
                x, n, features, dim, heads, w, cos, sin, q, k, v,
                block_m=32, block_half=128, block_k=128)
            torch.cuda.synchronize()
            gemma = {name: error(a, b) for name, a, b in (("q",q,ref_q),("k",k,ref_k),("v",v,ref_v))}
            records.append({"positions": "all_zero_control" if base == 0 else [base, base + n - 1],
                            "legacy_decoder": legacy, "existing_gemma_kernel": gemma})
            print(timestamp(), json.dumps(records[-1]), flush=True)
    result = {"started_at": started, "finished_at": timestamp(), "clock": clock_snapshot(),
              "source": file_record(source), "torch": torch.__version__,
              "workload": "Synthetic BF16 activations/weights; semantic unit check, not model performance or end-to-end parity",
              "shape": {"tokens": n, "input_features": features, "head_dim": dim, "q_heads": heads},
              "cases": records}
    (args.output / "rope_check.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
