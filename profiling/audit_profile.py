"""Check measured launch coverage against the instrumented decoder manifest."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

from profiling.roofline import aggregate_operators, analyze, attach_operators, load_ncu


LABEL = re.compile(r"step(-?\d+)\.layer(-?\d+)\.(\w+)\.line(\d+)")


def audit(kernels, benchmark, contract):
    if not benchmark.get("profiled"):
        raise ValueError("Coverage requires the benchmark from the profiled invocation.")
    manifest = benchmark["operator_manifest"]
    if not manifest:
        raise ValueError("Empty launch manifest.")
    expected = Counter(item["operator"] for item in manifest)
    measured = Counter()
    unlabelled = []
    for kernel in kernels:
        label = LABEL.search(kernel.get("operator", ""))
        if label:
            measured[label.group()] += 1
            if label.group(3) not in kernel["kernel"]:
                raise ValueError(f"Kernel name does not match its source label: {kernel['id']}")
        else:
            unlabelled.append({"id": kernel["id"], "kernel": kernel["kernel"]})
    if expected != measured:
        raise ValueError(f"Launch coverage differs: missing={dict(expected - measured)}, "
                         f"extra={dict(measured - expected)}")
    # All requested metrics must be present. Genuine zero counts remain valid data.
    for kernel in kernels:
        for name in contract["metrics"]:
            value = kernel["metrics"].get(name, (None, ""))[0]
            if value is None or value < 0:
                raise ValueError(f"Missing/invalid metric {name} for kernel {kernel['id']}")
    rows = analyze(kernels, contract)
    if any(r["duration_ns"] is None or r["duration_ns"] <= 0 for r in rows):
        raise ValueError("Nonpositive kernel duration.")
    by_step, by_layer = Counter(), Counter()
    for item in manifest:
        match = LABEL.fullmatch(item["operator"])
        if not match:
            raise ValueError(f"Invalid manifest label: {item['operator']}")
        by_step[int(match.group(1))] += 1
        by_layer[int(match.group(2))] += 1
    steps = benchmark["workload"]["steps"]
    if set(by_step) != set(range(steps)):
        raise ValueError("Denoising step coverage differs from the workload.")
    # Durations and traffic are identical across precision rows: select one domain.
    first_domain = next(iter(contract["domains"]))
    unique = [r for r in rows if r["domain"] == first_domain]
    total_ns = sum(r["duration_ns"] for r in unique)
    groups = [r for r in aggregate_operators(rows) if r["domain"] == first_domain]
    hotspots = [{"group_id": r["group_id"], "operator": r["operator"],
                 "invocations": r["invocations"], "ncu_total_ms": r["duration_ns"] / 1e6,
                 "ncu_mean_us": r["mean_duration_ns"] / 1e3,
                 "ncu_time_share_pct": 100 * r["duration_ns"] / total_ns,
                 "memory_level": r["memory_level"], "memory_bytes": r["memory_bytes"]}
                for r in sorted(groups, key=lambda r: r["duration_ns"], reverse=True)]
    report = {
        "coverage": "exact manifest/NCU label multiset match",
        "kernel_invocations": len(kernels), "labelled_triton_invocations": sum(measured.values()),
        "unlabelled_invocations": unlabelled,
        "triton_invocations_by_step": dict(sorted(by_step.items())),
        "triton_invocations_by_layer": dict(sorted(by_layer.items())),
        "all_requested_metrics_present": True,
        "precision_row_status_counts": dict(Counter(r["status"] for r in rows)),
        "ncu_kernel_duration_sum_ms": total_ns / 1e6,
        "memory_level": contract["memory"]["level"],
        "memory_bytes_sum": sum(r["memory_bytes"] for r in unique),
        "counted_flops_by_domain": {domain: sum(r["flops"] for r in rows if r["domain"] == domain)
                                    for domain in contract["domains"]},
        "warning": "NCU replay sums are not production latency. Time/traffic counted once per launch; "
                   "integer and transcendental instructions are excluded from counted FLOPs.",
    }
    return report, hotspots


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--annotated", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    kernels = load_ncu(args.raw)
    attach_operators(kernels, load_ncu(args.annotated))
    report, hotspots = audit(kernels, json.loads(args.benchmark.read_text()),
                             json.loads(args.contract.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "hotspots.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(hotspots[0]))
        writer.writeheader()
        writer.writerows(hotspots)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
