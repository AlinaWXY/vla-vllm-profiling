"""Attribute complete-VLA kernel reports and compare independently measured ranges.

Kernel replay supplies per-operator FLOPs/time/traffic. Whole-range replay
supplies aggregate traffic/time and hardware Tensor FLOPs. If range replay
cannot count FP32 SASS, its numerator comes from the matched kernel workload;
that provenance is explicit and BF16 totals must agree before combining runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import hashlib
from pathlib import Path
import re

from profiling.experiments import figure_record, timestamp
from profiling.roofline import load_ncu, attach_operators, analyze, aggregate_operators


NAMES = {
    "_build_gemma_rope_from_positions": "Prefix RoPE table",
    "_matmul_gemma_rope_qkv": "QKV projection + Gemma RoPE",
    "_matmul_abt_scale": "Attention QKᵀ + scale",
    "_softmax_mask_vector": "Prefix masked softmax",
    "_softmax_prefix_suffix_mask_vector": "Prefix/suffix masked softmax",
    "_matmul_small": "Attention P × V",
    "_matmul_small_res": "Attention output + residual",
    "_matmul_small_res_gate_oproj": "Attention output + gated residual",
    "_matmul_small_gate": "FFN gate/up + GELU product",
    "_matmul_small_res_gate_ffn_down": "FFN down + gated residual",
    "_matmul_small_bias": "Action input projection + bias",
    "_matmul_small_bias_res": "Action head + Euler update",
    "_attention_prefix_suffix_fused": "Fused attention",
}


def semantic(name, source_line, source):
    snippet = "\n".join(source[source_line - 1:source_line + 12])
    if name == "_rms_norm_kernel":
        return "RMSNorm (pre-FFN)" if "post_norm_w" in snippet else "RMSNorm (pre-attention)"
    if name == "_adarms_norm_kernel":
        return ("AdaRMS (final)" if "final_modulation" in snippet else
                "AdaRMS (pre-FFN)" if "post_style" in snippet else "AdaRMS (pre-attention)")
    return NAMES.get(name, name)


def attribute(kernels, manifest, source):
    expected = Counter(m["operator"] for m in manifest)
    actual = Counter(k.get("operator") for k in kernels if k.get("operator") in expected)
    if actual != expected:
        raise ValueError(f"Source launch coverage mismatch: missing={expected-actual}; extra={actual-expected}")
    records = {m["operator"]: m for m in manifest}
    for k in kernels:
        nvtx = k.get("operator", "")
        k["nvtx"] = nvtx
        if nvtx.startswith("action_expert"):
            stage = "action_expert"
        elif nvtx.startswith("vlm"):
            stage = "vlm"
        else:
            raise ValueError(f"Unattributed full-VLA kernel: {k['id']} {nvtx} {k['kernel']}")
        k["stage"] = stage
        if nvtx in records:
            m = records[nvtx]
            k["source_line"] = m["source_line"]
            k["tensor_shapes"] = json.dumps(m["tensors"], separators=(",", ":"))
            name = semantic(m["triton_kernel"], m["source_line"], source)
            part = "Prefix" if stage == "vlm" else "Expert"
            key = f"{part}: {name} (line {m['source_line']})"
        else:
            k["source_line"], k["tensor_shapes"] = "", ""
            part = ("Vision" if "vision" in nvtx else "Prefix FFN" if "FFN" in nvtx else
                    "Text" if "text_embedding" in nvtx else "Concat" if "concat" in nvtx else
                    "Prefix" if "prefix" in nvtx else "Expert")
            key = f"{part}: {k['kernel']} | grid={k['grid']} block={k['block']}"
        # A semantic key replaces raw NVTX only after retaining the exact label.
        k["operator"] = key
    return {"manifest_launches": sum(expected.values()), "matched_launches": sum(actual.values()),
            "kernel_invocations": len(kernels), "stages": dict(Counter(k["stage"] for k in kernels))}


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as out:
        writer = csv.DictWriter(out, fields)
        writer.writeheader()
        writer.writerows(rows)


def action_fingerprints(path):
    import numpy as np
    with np.load(path) as arrays:
        return {name: {"shape":list(arrays[name].shape), "dtype":str(arrays[name].dtype),
                       "sha256":hashlib.sha256(arrays[name].tobytes()).hexdigest()}
                for name in ("optimized", "noise")}


def short_label(key):
    part, name = key.split(": ", 1) if ": " in key else ("", key)
    if "triton_" in name:
        name = re.sub(r"^triton_(poi|red|per)_fused_", "", name.split(" | grid")[0])
    elif any(term in name.lower() for term in ("gemm", "cutlass", "xmma")):
        name = "GEMM specialization"
    elif "elementwise" in name or "copy_kernel" in name:
        name = "copy / elementwise specialization"
    elif "flash" in name.lower() or "fmha" in name.lower():
        name = "Fused attention specialization"
    name = re.sub(r" \(line \d+\)", "", name)
    if len(name) > 64:
        name = name[:61] + "..."
    return f"{part}: {name}" if part else name


def plot_ops(rows, output, ceilings, title, inputs, invocations=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    stamp = figure_record(output, inputs)
    names = sorted({r["operator"] for r in rows})
    ids = {name: i + 1 for i, name in enumerate(names)}
    colors = plt.get_cmap("turbo")
    height = max(7.5, 2.5 + .21 * len(names))
    fig = plt.figure(figsize=(20, height))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.5], height_ratios=[1, .04])
    for col, domain in enumerate(ceilings["compute_tflops"]):
        ax = fig.add_subplot(grid[0, col])
        pts = [r for r in rows if r["domain"] == domain and r["status"] == "ok"]
        peak, bw = ceilings["compute_tflops"][domain], ceilings["l2_bandwidth_gbps"]
        ridge = peak * 1000 / bw
        values = [r["ai_flops_per_byte"] for r in pts] + [ridge]
        x = np.geomspace(min(values) / 3, max(values) * 3, 300)
        ax.loglog(x, np.minimum(peak, bw*x/1000), color="#26364d", label="Empirical L2 / compute reference")
        for name in names:
            members = [r for r in pts if r["operator"] == name]
            if not members:
                continue
            ident = ids[name]
            positions = np.array([[r["ai_flops_per_byte"],r["performance_tflops"]] for r in members])
            ax.scatter(positions[:,0], positions[:,1], color=colors((ident-1)/max(1,len(names)-1)),
                       s=10 if invocations else 36, alpha=.35 if invocations else 1)
            center = np.median(np.log10(positions),axis=0)
            anchor = positions[np.argmin(np.sum((np.log10(positions)-center)**2,axis=1))]
            ax.annotate(str(ident), anchor,
                        xytext=(4, 5+(ident % 3)*7), textcoords="offset points", fontsize=8,
                        arrowprops={"arrowstyle":"-", "lw":.4, "color":"#888"})
        ax.set(title=domain, xlabel="L2 arithmetic intensity (FLOP/byte)", ylabel="Counted throughput (TFLOP/s)")
        ax.grid(True, which="both", alpha=.2)
        ax.legend(fontsize=8)
    legend = fig.add_subplot(grid[0, 2]); legend.axis("off")
    table = []
    for name in names:
        members = [r for r in rows if r["operator"] == name]
        label = short_label(name)
        domains = ", ".join(sorted({r["domain"] for r in members if r["status"] == "ok"}))
        table.append({"id":ids[name], "label":label, "operator":name, "plotted_domains":domains})
    legend.text(0, 1, "POINT → OPERATOR\n\n" + "\n".join(
        f"{r['id']:2d}  {r['label']}" + (" [no counted FLOPs]" if not r["plotted_domains"] else "")
        for r in table), va="top", fontsize=8.5, linespacing=1.4)
    fig.suptitle(title, y=.985, fontsize=15)
    note = ("Each point is one measured invocation; numeric callouts identify operator clusters." if invocations else
            "Repeated calls pooled as ΣF/Σbytes and ΣF/Σkernel-time.")
    fig.text(.5, .035, note+" Kernel replay clears caches; this is not whole-stage latency.", ha="center", fontsize=9)
    fig.text(.5, .013, stamp, ha="center", fontsize=9)
    fig.subplots_adjust(top=.93, bottom=.1, left=.055, right=.99, wspace=.3)
    for ext in ("png", "pdf"):
        fig.savefig(str(output)+"."+ext, dpi=180)
    plt.close(fig)
    csv_write(str(output)+"_legend.csv", table)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernels", type=Path, required=True, help="Experiment directory with ncu/ and workload/")
    p.add_argument("--ceilings", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.kernels/"workload/full_scopes.json").read_text())
    raw, labeled = args.kernels/"ncu/raw.csv", args.kernels/"ncu/operators.csv"
    kernels = load_ncu(raw)
    attach_operators(kernels, load_ncu(labeled))
    source = Path(report["framework"]["path"])/"vllm_omni/diffusion/models/pi05/realtime_triton.py"
    audit = attribute(kernels, report["operator_manifest"], source.read_text().splitlines())
    contract = json.loads((args.kernels/"metrics.json").read_text())
    missing = [(k["id"], name) for k in kernels for name in contract["metrics"]
               if name not in k["metrics"] or k["metrics"][name][0] is None]
    if missing:
        raise ValueError(f"Incomplete per-kernel metric coverage: {missing[:10]}")
    audit["all_requested_metrics_present"] = True
    ceilings = json.loads(args.ceilings.read_text())
    rows = analyze(kernels, contract)
    import numpy as np
    with np.load(args.kernels/"workload/actions.npz") as arrays:
        for name in ("scoped", "scoped_graph"):
            error = np.max(np.abs(arrays[name].reshape(arrays["optimized"].shape)-arrays["optimized"]))
            if not np.isfinite(error) or error > 1e-5:
                raise ValueError(f"Profiled {name} output differs from production: {error}")
    # aggregate_operators expects operator keys; the legacy helper otherwise
    # falls back to kernel names. Here semantic keys are intentional identities.
    for stage in ("vlm", "action_expert"):
        names = sorted({r["operator"] for r in rows if r["stage"] == stage})
        group_ids = {name:i+1 for i,name in enumerate(names)}
        for row in rows:
            if row["stage"] == stage:
                row["group_id"] = group_ids[row["operator"]]
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"kernels.csv").exists():
        raise FileExistsError(args.output)
    csv_write(args.output/"kernels.csv", rows)
    totals = {}
    for stage in ("vlm", "action_expert"):
        stage_rows = [r for r in rows if r["stage"] == stage]
        grouped = aggregate_operators(stage_rows)
        csv_write(args.output/f"{stage}_operators.csv", grouped)
        plot_ops(grouped, args.output/f"{stage}_operators", ceilings,
                 f"π0.5 {stage.upper()} operators — Omni {report['framework']['commit'][:7]}",
                 [raw, labeled, args.kernels/"workload/full_scopes.json", args.ceilings])
        plot_ops(stage_rows, args.output/f"{stage}_invocations", ceilings,
                 f"π0.5 {stage.upper()} — every kernel invocation — Omni {report['framework']['commit'][:7]}",
                 [raw, labeled, args.kernels/"workload/full_scopes.json", args.ceilings], invocations=True)
        one_domain = [r for r in stage_rows if r["domain"] == next(iter(contract["domains"]))]
        totals[stage] = {"kernel_time_sum_ms": sum(r["duration_ns"] for r in one_domain)/1e6,
                         "kernel_l2_bytes_sum": sum(r["memory_bytes"] for r in one_domain),
                         "flops": {d:sum(r["flops"] for r in stage_rows if r["domain"] == d)
                                   for d in contract["domains"]}}
    summary = {"analyzed_at": timestamp(), "coverage":audit, "totals":totals,
               "input_audit":report["input_audit"],
               "model_parameter_fingerprint":report["model_parameter_fingerprint"],
               "actions":action_fingerprints(args.kernels/"workload/actions.npz"),
               "framework":report["framework"], "scope":"Kernel replay only; whole-stage ranges are independent measurements"}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
