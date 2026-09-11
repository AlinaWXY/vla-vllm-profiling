"""Parse NCU long-form CSV without silently treating missing counters as zero.

Each invocation stays separate. Precision domains stay separate as well: a BF16
Tensor operation and an FP32 instruction do not share a hardware compute roof.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path


def number(value):
    try:
        value = float(str(value).replace(",", "").strip())
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def load_ncu(path):
    # NCU may prepend connection/progress messages and repeat its CSV header.
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    start = next((i for i, line in enumerate(lines)
                  if "Metric Name" in line and "Metric Value" in line
                  and "ID" in next(csv.reader([line]), [])), None)
    if start is None:
        raise ValueError("No NCU raw CSV header; export with --page raw --csv.")
    kernels = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        if not row.get("ID") or row["ID"] == "ID" or not row.get("Metric Name"):
            continue
        key = (row.get("Process ID", ""), row.get("Device", ""),
               row.get("Context", ""), row.get("Stream", ""), row["ID"])
        entry = kernels.setdefault(key, {
            "id": row["ID"], "process_id": key[0], "device": key[1],
            "context": key[2], "stream": key[3],
            "kernel": row.get("Kernel Name", ""),
            "grid": row.get("Grid Size", ""), "block": row.get("Block Size", ""),
            "metrics": {},
        })
        metric = (number(row["Metric Value"]), (row.get("Metric Unit") or "").strip())
        name = row["Metric Name"]
        old = entry["metrics"].get(name)
        if old is not None and old != metric:
            raise ValueError(f"Conflicting duplicate metric {name} for kernel {key}")
        entry["metrics"][name] = metric
    if not kernels:
        raise ValueError("The report contains no kernel metrics.")
    return list(kernels.values())


def metric_value(kernel, name, kind):
    if name not in kernel["metrics"]:
        return None
    value, unit = kernel["metrics"][name]
    if value is None or value < 0:
        return None
    scales = {
        "time": {"second": 1, "s": 1, "msecond": 1e-3, "ms": 1e-3,
                 "usecond": 1e-6, "us": 1e-6, "nsecond": 1e-9, "ns": 1e-9},
        "bytes": {"byte": 1, "bytes": 1, "Kbyte": 1e3, "Mbyte": 1e6,
                  "Gbyte": 1e9, "KiB": 1024, "MiB": 1024**2},
        "count": {"": 1, "inst": 1, "instruction": 1, "instructions": 1,
                  "op": 1, "ops": 1, "sector": 1, "sectors": 1},
    }
    if unit not in scales[kind]:
        raise ValueError(f"Unsupported {kind} unit {unit!r} for {name}; use --print-units base")
    return value * scales[kind][unit]


def analyze(kernels, contract):
    if not contract.get("domains"):
        raise ValueError("At least one explicitly defined precision domain is required.")
    for domain, spec in contract["domains"].items():
        terms = spec.get("terms", [])
        if not terms or len({t["metric"] for t in terms}) != len(terms):
            raise ValueError(f"Empty or duplicate FLOP terms in {domain}")
        if any(number(t["weight"]) is None or t["weight"] <= 0
               or not t["metric"].endswith(".sum") for t in terms):
            raise ValueError(f"Use positive weights and absolute .sum counters in {domain}")
    rows = []
    for kernel in kernels:
        duration = metric_value(kernel, "gpu__time_duration.sum", "time")
        dram = metric_value(kernel, "dram__bytes.sum", "bytes")
        if dram is None:
            read = metric_value(kernel, "dram__bytes_read.sum", "bytes")
            write = metric_value(kernel, "dram__bytes_write.sum", "bytes")
            if read is not None and write is not None:
                dram = read + write
        for domain, spec in contract["domains"].items():
            missing = []
            flops = 0.0
            for term in spec["terms"]:
                count = metric_value(kernel, term["metric"], "count")
                if count is None:
                    missing.append(term["metric"])
                else:
                    flops += count * term["weight"]
            issues = []
            if duration is None or duration <= 0:
                issues.append("missing_or_nonpositive_duration")
            if dram is None or dram <= 0:
                issues.append("missing_or_zero_dram_traffic")
            if missing:
                issues.append("missing_flop_metrics:" + ";".join(missing))
            if not missing and flops == 0:
                issues.append("zero_counted_flops")
            complete_flops = None if missing else flops
            rows.append({
                **{k: v for k, v in kernel.items() if k != "metrics"},
                "domain": domain, "duration_ns": None if duration is None else duration * 1e9,
                "dram_bytes": dram, "flops": complete_flops,
                "ai_flops_per_byte": flops / dram if not issues else None,
                "performance_tflops": flops / duration / 1e12 if not issues else None,
                "status": "ok" if not issues else "|".join(issues),
            })
    return rows


def attach_operators(kernels, annotated):
    def identity(k):
        return tuple(k.get(name, "") for name in ("process_id", "device", "context", "stream", "id"))
    labels = {identity(k): k for k in annotated}
    if set(labels) != {identity(k) for k in kernels}:
        raise ValueError("Operator CSV must be exported from the exact same report (kernel identities differ).")
    for kernel in kernels:
        label = labels[identity(kernel)]
        if label["metrics"] != kernel["metrics"]:
            raise ValueError("Operator CSV metrics differ; reports cannot be joined safely.")
        kernel["operator"] = label["kernel"]


def plot(rows, ceilings, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not ceilings.get("provenance"):
        raise ValueError("Ceilings need provenance (device, power/clock state, method/source).")
    domains = list(ceilings["compute_tflops"])
    if not domains:
        raise ValueError("No compute ceilings supplied.")
    fig, axes = plt.subplots(1, len(domains), figsize=(7 * len(domains), 5), squeeze=False)
    for ax, domain in zip(axes[0], domains):
        points = [r for r in rows if r["domain"] == domain and r["status"] == "ok"]
        peak = ceilings["compute_tflops"][domain]
        bandwidth = ceilings["dram_bandwidth_gbps"]
        if peak <= 0 or bandwidth <= 0:
            raise ValueError("Ceilings must be positive, in TFLOP/s and decimal GB/s.")
        intensities = [r["ai_flops_per_byte"] for r in points]
        ridge = peak * 1000 / bandwidth
        lo = min(intensities + [ridge]) / 4
        hi = max(intensities + [ridge]) * 4
        x = np.logspace(np.log10(lo), np.log10(hi), 400)
        ax.loglog(x, np.minimum(peak, bandwidth * x / 1000), color="#35465c",
                  label=ceilings.get("label", "Documented hardware ceiling"))
        if points:
            scatter = ax.scatter(intensities, [r["performance_tflops"] for r in points],
                                 c=[r["duration_ns"] / 1000 for r in points],
                                 cmap="viridis", s=24, alpha=0.75, rasterized=True)
            fig.colorbar(scatter, ax=ax, label="NCU replay duration (µs)")
        else:
            ax.text(.5, .5, "No valid measured points", transform=ax.transAxes, ha="center")
        ax.set(xlabel="DRAM arithmetic intensity (FLOP/byte)",
               ylabel="Counted arithmetic throughput (TFLOP/s)", title=domain)
        ax.grid(True, which="both", alpha=.18)
        ax.legend(fontsize=8)
    fig.suptitle(ceilings.get("title", "π0.5 Action Expert — NCU kernel invocations"))
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(str(output) + "." + extension, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--operators-csv", type=Path)
    parser.add_argument("--ceilings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kernels = load_ncu(args.csv)
    if args.operators_csv:
        attach_operators(kernels, load_ncu(args.operators_csv))
    contract = json.loads(args.contract.read_text())
    rows = analyze(kernels, contract)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "kernels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"kernel_invocations": len(kernels), "precision_rows": len(rows),
               "valid_precision_rows": sum(r["status"] == "ok" for r in rows),
               "domains": list(contract["domains"]),
               "note": "One kernel can appear in multiple precision domains; do not sum their durations."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.ceilings:
        plot(rows, json.loads(args.ceilings.read_text()), args.output / "roofline")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
