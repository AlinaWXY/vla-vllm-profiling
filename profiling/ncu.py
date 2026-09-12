"""Discover target-specific metrics and collect explicitly scoped NCU reports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess


FP32_METRICS = [f"smsp__sass_thread_inst_executed_op_f{op}_pred_on.sum"
                for op in ("add", "mul", "fma")]
BF16_METRICS = ["sm__ops_path_tensor_src_bf16_dst_fp32.sum",
                "sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum"]


def require_gpu_execution():
    """Direct execution is explicitly authorized on Thor, Slurm elsewhere."""
    if socket.gethostname().split(".")[0] in {"thor0", "fact-thor"}:
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Direct GPU execution is authorized only on thor0/fact-thor; other hosts require Slurm.")


def select_contract(query, memory_level="dram"):
    if memory_level not in {"dram", "l2"}:
        raise ValueError("Memory level must be dram or l2.")
    available = set(re.findall(r"\b(?:gpu|dram|smsp|sm|lts)__[A-Za-z0-9_.]+", query))
    traffic = (["dram__bytes_read.sum", "dram__bytes_write.sum"] if memory_level == "dram"
               else ["lts__t_bytes.sum"])
    required = ["gpu__time_duration.sum", *traffic]
    absent = [name for name in required + FP32_METRICS if name not in available]
    tensor = next((name for name in BF16_METRICS if name in available), None)
    if absent or tensor is None:
        raise ValueError("Target counters need manual review; refusing incomplete roofline. "
                         f"Missing: {absent}; BF16 Tensor candidate: {tensor}. "
                         "Inspect metrics.txt and installed NCU Tensor roofline sections.")
    return {
        "schema_version": 2,
        "note": "Dense BF16 Tensor and FP32 add/mul/FMA are separate roofs; transcendental and integer ops excluded.",
        "metrics": required + FP32_METRICS + [tensor],
        "memory": {"level": memory_level, "metrics": traffic,
                   "note": "L2 requested bytes are not DRAM bytes; use a matching memory-level bandwidth roof."},
        "domains": {
            "bf16_tensor": {"terms": [{"metric": tensor, "weight": 1}]},
            "fp32_simt": {"terms": [{"metric": name, "weight": weight}
                                       for name, weight in zip(FP32_METRICS, (1, 1, 2))]},
        },
    }


def discover(ncu, output, memory_level="dram"):
    require_gpu_execution()
    if (output / "metrics.json").exists():
        raise FileExistsError("Existing metric contract; select a new inventory directory.")
    output.mkdir(parents=True, exist_ok=True)
    # Query base names first, then suffixes only for supported candidates. A full
    # suffix inventory on Thor is tens of MB and is unnecessary for this contract.
    def query(name, args):
        result = subprocess.run([ncu, *args], check=False, capture_output=True, text=True)
        (output / name).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"NCU query failed ({result.returncode}); inspect {output / name}")
        return result.stdout

    query("version.txt", ["--version"])
    query("sections.txt", ["--list-sections"])
    bases = query("metric_bases.txt", ["--query-metrics", "--query-metrics-mode", "base"])
    available = set(re.findall(r"\b(?:gpu|dram|smsp|sm|lts)__[A-Za-z0-9_]+", bases))
    candidates = ["gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
                  "lts__t_bytes.sum", *FP32_METRICS, *BF16_METRICS]
    selected = sorted({metric.split(".")[0] for metric in candidates} & available)
    if not selected:
        raise ValueError("No known metric bases on target; inspect metric_bases.txt")
    metrics = query("metrics.txt", ["--query-metrics-mode", "suffix", "--metrics", ",".join(selected)])
    contract = select_contract(metrics, memory_level)
    (output / "metrics.json").write_text(json.dumps(contract, indent=2) + "\n")


def capture(args):
    require_gpu_execution()
    if not args.command:
        raise ValueError("Pass the benchmark command after --.")
    command = args.command[1:] if args.command[0] == "--" else args.command
    contract = json.loads(args.contract.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    report = args.output / "action_expert"
    cmd = [args.ncu, "--target-processes", "all", "--profile-from-start", "off",
           "--nvtx", "--nvtx-include", "action_expert/",
           "--graph-profiling", "node", "--replay-mode", "kernel",
           "--clock-control", "none", "--cache-control", "all",
           "--metrics", ",".join(contract["metrics"]), "--export", str(report), *command]
    manifest = {
        "command": cmd, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "collection": "Isolated kernel replay, caches flushed; no NCU clock locking on Thor.",
        "scope": "Action Expert diagnostic replay of the optimized Triton kernels; CUDA Graph disabled for attribution.",
        "metrics_contract": contract,
    }
    (args.output / "collection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(shlex.join(cmd), flush=True)
    with (args.output / "ncu.log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    for name, extra in (("raw.csv", []), ("operators.csv", ["--print-nvtx-rename", "kernel"])):
        with (args.output / name).open("w") as csvfile:
            subprocess.run([args.ncu, "--import", str(report) + ".ncu-rep", "--page", "raw",
                            "--csv", "--print-units", "base", "--print-fp", *extra],
                           stdout=csvfile, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ncu", default="ncu")
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--memory-level", choices=("dram", "l2"), default="dram",
                   help="Explicit roof level; L2 traffic cannot substitute for DRAM traffic.")
    c = sub.add_parser("capture")
    c.add_argument("--contract", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.mode == "discover":
        discover(args.ncu, args.output, args.memory_level)
    else:
        capture(args)


if __name__ == "__main__":
    main()
