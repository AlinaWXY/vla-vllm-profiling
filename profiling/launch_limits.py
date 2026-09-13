"""Report resource limits already present in the original NCU export."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from profiling.diagnose import LAUNCH_METRICS, value, write_csv
from profiling.experiments import figure_record, file_record, timestamp
from profiling.roofline import load_ncu, attach_operators, operator_key, operator_label


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--annotated", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    kernels = load_ncu(args.raw)
    attach_operators(kernels, load_ncu(args.annotated))
    groups = defaultdict(list)
    for kernel in kernels:
        name = operator_key(kernel)
        if name.startswith("_"):
            groups[name].append(kernel)
    rows = []
    for group_id, name in enumerate(sorted(groups), 1):
        row = {"group_id": group_id, "operator": name, "label": operator_label(name),
               "invocations": len(groups[name])}
        for key, metric in LAUNCH_METRICS.items():
            values = [value(k, metric) for k in groups[name]]
            row[key + "_min"] = min(values)
            row[key + "_max"] = max(values)
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "launch_limits.csv", rows)
    (args.output / "launch_limits_source.json").write_text(json.dumps({
        "analyzed_at": timestamp(), "inputs": [file_record(args.raw), file_record(args.annotated)],
        "scope": "All 1650 original Triton launches, 10 steps, 18 layers",
        "note": "Occupancy limit is theoretical, not measured active occupancy. Resource limits and launch sizes can coexist with memory stalls."
    }, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    stamp = figure_record(args.output / "launch_limits", [args.raw, args.annotated])
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    for ax, key, scale, xlabel in zip(axes, ("theoretical_occupancy_pct", "registers_per_thread", "shared_bytes_per_block"),
                                    (1, 1, 1024), ("Theoretical occupancy limit (%)", "Registers per thread", "Shared memory per block (KiB)")):
        values = [r[key + "_max"] / scale for r in rows]
        minimums = [r[key + "_min"] / scale for r in rows]
        ax.barh(y, values, color=["#b36128" if r["group_id"] in (9, 10) else "#2b8198" for r in rows])
        ax.errorbar(values, y, xerr=([v - m for v, m in zip(values, minimums)], [0] * len(rows)),
                    fmt="none", ecolor="black", capsize=3)
        for i, v in enumerate(values):
            ax.text(v + max(values) * .02, i, f"{v:g}", va="center", fontsize=9)
        ax.set(xlabel=xlabel, xlim=(0, max(values) * 1.22), yticks=y)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_yticklabels([f"[{r['group_id']}] {r['label']}" for r in rows], fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("π0.5 Action Expert — resource limits from all original NCU launches")
    fig.text(.5, .045, "Theoretical occupancy is an upper bound, not measured utilization. "
             "FFN gate/up and down each allow only 2 resident blocks/SM; block size = 128 threads.", ha="center", fontsize=9)
    fig.text(.5, .015, stamp, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .96))
    for ext in ("png", "pdf"):
        fig.savefig(args.output / f"launch_limits.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
