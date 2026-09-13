"""Compare explicitly scoped cache experiments without inventing a DRAM roof."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from profiling.audit_profile import audit
from profiling.experiments import figure_record, file_record, timestamp
from profiling.roofline import (load_ncu, attach_operators, operator_key, operator_label,
                               metric_value, analyze)


PERCENT_METRICS = {
    "l1_hit_pct": "l1tex__t_sector_hit_rate.pct",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "l1_util_pct": "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_util_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm_util_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "tensor_active_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    **{f"stall_{s}_pct": f"smsp__warp_issue_stalled_{s}_per_warp_active.pct"
       for s in ("long_scoreboard", "short_scoreboard", "wait", "math_pipe_throttle", "not_selected")},
}
LAUNCH_METRICS = {
    "theoretical_occupancy_pct": "sm__maximum_warps_per_active_cycle_pct",
    "registers_per_thread": "launch__registers_per_thread",
    "shared_bytes_per_block": "launch__shared_mem_per_block",
    "register_block_limit": "launch__occupancy_limit_registers",
    "shared_block_limit": "launch__occupancy_limit_shared_mem",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "grid_blocks": "launch__grid_size",
    "block_threads": "launch__block_size",
}


def value(kernel, metric, unit=None):
    result, actual_unit = kernel["metrics"].get(metric, (None, ""))
    if result is None or result < 0 or (unit is not None and actual_unit != unit):
        raise ValueError(f"Missing/invalid {metric} {actual_unit!r} in kernel {kernel['id']}")
    return result


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_run(path):
    kernels = load_ncu(path / "ncu/raw.csv")
    attach_operators(kernels, load_ncu(path / "ncu/operators.csv"))
    contract = json.loads((path / "metrics.json").read_text())
    benchmark = json.loads((path / "workload/benchmark.json").read_text())
    coverage, _ = audit(kernels, benchmark, contract, selected_steps=[0])
    if len(kernels) != 168 or coverage["labelled_triton_invocations"] != 165:
        raise ValueError("Expected exactly the first 168 launches, including all 165 step00 kernels")
    rows, groups = [], defaultdict(list)
    for kernel in kernels:
        name = operator_key(kernel)
        if not name.startswith("_"):
            continue
        row = {"id": kernel["id"], "operator": kernel["operator"], "source": name,
               "duration_us": metric_value(kernel, "gpu__time_duration.sum", "time") * 1e6,
               "l1_bytes": metric_value(kernel, "l1tex__t_bytes.sum", "bytes"),
               "l2_bytes": metric_value(kernel, "lts__t_bytes.sum", "bytes"),
               "l2_read_miss_sectors": value(kernel, "lts__t_sectors_op_read_lookup_miss.sum", "sector"),
               "sm_clock_hz": value(kernel, "sm__cycles_elapsed.avg.per_second", "cycle/second"),
               **{key: value(kernel, metric, "%") for key, metric in PERCENT_METRICS.items()},
               **{key: value(kernel, metric) for key, metric in LAUNCH_METRICS.items()}}
        # This is an L2 read-request miss proxy, never an exact DRAM traffic count.
        row["l2_read_miss_request_bytes"] = 32 * row["l2_read_miss_sectors"]
        row["l2_gbps"] = row["l2_bytes"] / row["duration_us"] / 1000
        rows.append(row)
        groups[name].append(row)
    summary = []
    for index, name in enumerate(sorted(groups), 1):
        members = groups[name]
        time_sum = sum(r["duration_us"] for r in members)
        row = {"group_id": index, "operator": name, "label": operator_label(name),
               "invocations": len(members), "total_us": time_sum,
               "mean_us": time_sum / len(members),
               "l2_gbps": sum(r["l2_bytes"] for r in members) / time_sum / 1000,
               "l2_read_miss_request_bytes": sum(r["l2_read_miss_request_bytes"] for r in members)}
        for key in PERCENT_METRICS:
            weight = "l1_bytes" if key == "l1_hit_pct" else "l2_bytes" if key == "l2_hit_pct" else "duration_us"
            denominator = sum(r[weight] for r in members)
            row[key] = sum(r[key] * r[weight] for r in members) / denominator if denominator else None
        for key in LAUNCH_METRICS:
            row[key + "_min"] = min(r[key] for r in members)
            row[key + "_max"] = max(r[key] for r in members)
        row["sm_clock_hz_median"] = statistics.median(r["sm_clock_hz"] for r in members)
        summary.append(row)
    return kernels, coverage, rows, summary, analyze(kernels, contract)


def plot_comparison(cold, warm, output, inputs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    stamp = figure_record(output / "cache_comparison", inputs)
    y = np.arange(len(cold))
    labels = [f"[{r['group_id']}] {r['label']}" for r in cold]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    for ax, key, title in zip(axes, ("mean_us", "l2_hit_pct", "stall_long_scoreboard_pct"),
                              ("Mean kernel duration (µs)", "L2 hit rate (%)", "Long-scoreboard stalls (%)")):
        ax.barh(y - .18, [r[key] for r in cold], .35, label="cache-control all", color="#a66230")
        ax.barh(y + .18, [r[key] for r in warm], .35, label="cache-control none", color="#247f98")
        ax.set(xlabel=title, yticks=y)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=9)
    fig.suptitle("π0.5 Action Expert — first denoising step, same weights and shapes")
    fig.text(.5, .045, "Both use kernel replay; no-flush is a cache sensitivity experiment, not application-cache replay. "
             "Stalls are normalized over active warps.", ha="center", fontsize=9)
    fig.text(.5, .015, stamp, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .96))
    for ext in ("png", "pdf"):
        fig.savefig(output / f"cache_comparison.{ext}", dpi=180)
    plt.close(fig)

    stamp = figure_record(output / "utilization", inputs)
    keys = ["occupancy_pct", "tensor_active_pct", "l1_util_pct", "l2_util_pct", "stall_long_scoreboard_pct"]
    labels_x = ["Occupancy*", "Tensor active", "L1/TEX throughput", "L2 throughput", "Long scoreboard*"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, data, title in zip(axes, (cold, warm), ("Cache flush: all", "Cache flush: none")):
        values = np.array([[r[k] for k in keys] for r in data])
        im = ax.imshow(values, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        ax.set(xticks=np.arange(len(keys)), yticks=y, title=title)
        ax.set_xticklabels(labels_x, rotation=22, ha="right", fontsize=9)
        for i in range(len(data)):
            for j in range(len(keys)):
                ax.text(j, i, f"{values[i,j]:.1f}", ha="center", va="center",
                        color="white" if values[i,j] > 55 else "black", fontsize=9)
    axes[0].set_yticklabels(labels, fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Percent", fraction=.02, pad=.02)
    fig.suptitle("π0.5 Action Expert — resource utilization and memory-dependency waits")
    fig.text(.5, .05, "* Occupancy: active SM cycles; stalls: active warps. Other columns: elapsed cycles. "
             "Columns have different denominators; do not add them.", ha="center", fontsize=8)
    fig.text(.5, .015, stamp, ha="center", fontsize=9)
    fig.subplots_adjust(left=.28, right=.86, bottom=.19, top=.91, wspace=.13)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"utilization.{ext}", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cold", type=Path, required=True)
    p.add_argument("--warm", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs, data, signatures, flop_signatures = [], {}, [], []
    for label, path in (("all", args.cold), ("none", args.warm)):
        collection = json.loads((path / "ncu/collection.json").read_text())
        cmd = collection["command"]
        if cmd[cmd.index("--cache-control") + 1] != label:
            raise ValueError("Cache-policy input mismatch")
        kernels, coverage, rows, summary, precision_rows = read_run(path)
        inputs += [path / "ncu/raw.csv", path / "ncu/operators.csv", path / "ncu/collection.json",
                   path / "workload/benchmark.json", path / "workload/actions.npz"]
        signatures.append([(k["operator"], k["grid"], k["block"]) for k in kernels])
        flop_signatures.append([r["flops"] for r in precision_rows])
        (args.output / f"coverage_{label}.json").write_text(json.dumps(coverage, indent=2) + "\n")
        write_csv(args.output / f"invocations_{label}.csv", rows)
        write_csv(args.output / f"operators_{label}.csv", summary)
        data[label] = summary
    if signatures[0] != signatures[1] or flop_signatures[0] != flop_signatures[1]:
        raise ValueError("Paired captures differ in operator order, shapes, or counted work")
    import numpy as np
    with np.load(args.cold / "workload/actions.npz") as a, np.load(args.warm / "workload/actions.npz") as b:
        if set(a.files) != set(b.files) or not all(np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError("Paired captures differ in input noise or pipeline output")
    report = {"analyzed_at": timestamp(), "sources": [file_record(p) for p in inputs],
              "paired_operator_order_grid_block_and_flops_equal": True,
              "paired_pipeline_output_and_noise_equal": True,
              "aggregation": "Time-weighted ratios except cache hits weighted by same-level bytes; ranges for launch limits.",
              "scope": "step00 only, 168 launches: 165 Triton + 3 framework helpers; full model executes 10 steps",
              "limitations": ["No-flush kernel replay does not preserve application cache state between passes.",
                              "L2 read-miss request bytes are not measured DRAM bytes or bandwidth.",
                              "One capture per policy; short kernels may have cross-pass counter variation.",
                              "Optimized backend numerical equivalence remains unverified."]}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_comparison(data["all"], data["none"], args.output, inputs)
    print(json.dumps({"validated": True, "operator_groups": len(data["all"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
