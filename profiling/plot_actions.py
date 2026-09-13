"""Visualize the saved normalized-space action discrepancy; no parity claim."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from profiling.experiments import figure_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("actions", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    stamp = figure_record(args.output / "action_comparison", [args.actions])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with np.load(args.actions, allow_pickle=False) as arrays:
        safe, optimized = arrays["safe"].astype(np.float64), arrays["optimized"].astype(np.float64)
    if safe.ndim != 2 or optimized.shape != safe.shape or not np.isfinite([safe, optimized]).all():
        raise ValueError("Expected finite, equally shaped action chunks [horizon, dimensions].")
    diff = optimized - safe
    error_rms = np.sqrt(np.mean(diff**2))
    safe_rms = np.sqrt(np.mean(safe**2))
    relative = f"{100 * error_rms / safe_rms:.1f}%" if safe_rms else "undefined (zero reference RMS)"
    limit = max(np.abs(safe).max(), np.abs(optimized).max()) or 1
    error_limit = np.abs(diff).max() or 1
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "error_by_dimension.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dimension", "safe_rms", "optimized_rms", "error_rms", "mean_abs", "max_abs"])
        for dim in range(safe.shape[1]):
            writer.writerow([dim, np.sqrt(np.mean(safe[:, dim]**2)),
                             np.sqrt(np.mean(optimized[:, dim]**2)), np.sqrt(np.mean(diff[:, dim]**2)),
                             np.abs(diff[:, dim]).mean(), np.abs(diff[:, dim]).max()])
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)
    for ax, values, title, scale in zip(axes, (safe, optimized, diff),
                                       ("Safe backend", "Optimized backend", "Optimized − safe"),
                                       (limit, limit, error_limit)):
        im = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="RdBu_r",
                       vmin=-scale, vmax=scale)
        ax.set(xlabel=f"Action dimension (all {safe.shape[1]} retained)", ylabel="Action horizon index", title=title)
        fig.colorbar(im, ax=ax, label="Normalized action value" if ax is not axes[2] else "Difference")
    fig.suptitle(f"π0.5 experimental PR 4419 — relative RMSE {relative}\n"
                 "Real weights, synthetic observation; numerical equivalence not established\n"
                 + stamp, fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"action_comparison.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
