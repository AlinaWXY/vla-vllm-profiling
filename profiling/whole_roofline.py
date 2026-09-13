"""Draw whole-stage rooflines using measured range traffic and range duration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from profiling.experiments import figure_record, timestamp
from profiling.full_roofline import csv_write, action_fingerprints
from profiling.roofline import load_ncu, metric_value


def combine(ranges, kernel_summary, scope_order):
    if len(ranges) != len(scope_order) or set(scope_order) != {"vlm", "action_expert", "vla"}:
        raise ValueError("Exactly three independently captured ranges VLM/Expert/VLA are required")
    rows = []
    totals = kernel_summary["totals"]
    for measured, scope in zip(ranges, scope_order):
        duration = metric_value(measured, "gpu__time_duration.sum", "time")
        traffic = metric_value(measured, "lts__t_bytes.sum", "bytes")
        bf16 = metric_value(measured, "sm__ops_path_tensor_src_bf16_dst_fp32.sum", "count")
        if any(value is None or value <= 0 for value in (duration, traffic, bf16)):
            raise ValueError(f"Incomplete or invalid whole-range counters for {scope}")
        stages = ["vlm", "action_expert"] if scope == "vla" else [scope]
        expected = sum(totals[s]["flops"]["bf16_tensor"] for s in stages)
        if bf16 != expected:
            raise ValueError(f"Whole/kernel BF16 count mismatch for {scope}: {bf16} vs {expected}")
        fp32 = sum(totals[s]["flops"]["fp32_simt"] for s in stages)
        for domain, flops in (("bf16_tensor", bf16), ("fp32_simt", fp32)):
            rows.append({"scope":scope, "range_id":measured["id"], "range_name":measured["kernel"],
                "domain":domain, "flops":flops, "memory_level":"l2", "range_l2_bytes":traffic,
                "range_duration_ms":duration*1000, "ai_flops_per_byte":flops/traffic,
                "performance_tflops":flops/duration/1e12,
                "flops_source":"whole-range hardware counter" if domain == "bf16_tensor" else
                               "sum of matching kernel-replay SASS counts (unsupported in range mode)",
                "traffic_time_source":"whole CUDA Graph profiling; cache-control=none",
                "bf16_counts_match_kernel_replay":True})
    return rows


def plot(rows, output, ceilings, title, inputs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    stamp = figure_record(output, inputs)
    fig, axes = plt.subplots(1,2,figsize=(13,5.5))
    colors = {"vlm":"#276fbf", "action_expert":"#d45d22", "vla":"#358c57"}
    labels = {"vlm":"VLM", "action_expert":"Action Expert", "vla":"VLA overall"}
    for ax, domain in zip(axes, ceilings["compute_tflops"]):
        pts = [r for r in rows if r["domain"] == domain]
        peak, bw = ceilings["compute_tflops"][domain], ceilings["l2_bandwidth_gbps"]
        ridge = peak*1000/bw
        values = [r["ai_flops_per_byte"] for r in pts]+[ridge]
        x = np.geomspace(min(values)/5,max(values)*5,300)
        ax.loglog(x,np.minimum(peak,bw*x/1000),color="#26364d",label="Empirical L2 / compute reference")
        for i,r in enumerate(pts):
            ax.scatter(r["ai_flops_per_byte"],r["performance_tflops"],s=75,color=colors[r["scope"]])
            ax.annotate(f"{labels[r['scope']]}\n{r['range_duration_ms']:.3f} ms graph",
                (r["ai_flops_per_byte"],r["performance_tflops"]),xytext=(10,10-32*i),
                textcoords="offset points",fontsize=10,
                arrowprops={"arrowstyle":"-","color":colors[r["scope"]]})
        ax.set(title=domain,xlabel="L2 arithmetic intensity (FLOP/byte)",ylabel="Counted throughput (TFLOP/s)")
        ax.grid(True,which="both",alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(title,fontsize=14)
    fig.text(.5,.085,"Time and L2 bytes: complete CUDA Graph profiling. FP32 FLOPs: matching per-kernel counts; BF16 totals cross-checked.",ha="center",fontsize=8.5)
    fig.text(.5,.05,"GPU scope graphs recaptured from the same compiled kernels; original pipeline wall time is reported separately.",ha="center",fontsize=8.5)
    fig.text(.5,.015,stamp,ha="center",fontsize=8.5)
    fig.tight_layout(rect=(0,.12,1,.96))
    for ext in ("png","pdf"):
        fig.savefig(str(output)+"."+ext,dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranges",type=Path,required=True)
    p.add_argument("--kernel-analysis",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--ceilings",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    kernel_summary=json.loads((args.kernel_analysis/"summary.json").read_text())
    report=json.loads((args.ranges/"workload/full_scopes.json").read_text())
    baseline=json.loads((args.baseline/"workload/full_scopes.json").read_text())
    if report["framework"]["commit"] != kernel_summary["framework"]["commit"] or report["framework"] != baseline["framework"]:
        raise ValueError("Framework differs between whole-range, kernels and baseline")
    if report["input_audit"] != baseline["input_audit"]:
        raise ValueError("Input configuration differs")
    if report["model_parameter_fingerprint"] != baseline["model_parameter_fingerprint"] or report["model_parameter_fingerprint"] != kernel_summary["model_parameter_fingerprint"]:
        raise ValueError("Actual model parameters differ across captures")
    if report["input_audit"] != kernel_summary["input_audit"] or kernel_summary["actions"] != action_fingerprints(args.ranges/"workload/actions.npz"):
        raise ValueError("Kernel and whole-range workloads differ")
    with np.load(args.ranges/"workload/actions.npz") as a,np.load(args.baseline/"workload/actions.npz") as b:
        if not np.array_equal(a["optimized"],b["optimized"]) or not np.array_equal(a["noise"],b["noise"]):
            raise ValueError("Actions or noise differ between baseline and range run")
    rows=combine(load_ncu(args.ranges/"ncu/raw.csv"),kernel_summary,report["range_order"])
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"whole_ranges.csv").exists():
        raise FileExistsError(args.output)
    csv_write(args.output/"whole_ranges.csv",rows)
    ceilings=json.loads(args.ceilings.read_text())
    inputs=[args.ranges/"ncu/raw.csv",args.ranges/"workload/full_scopes.json",
            args.kernel_analysis/"summary.json",args.baseline/"workload/full_scopes.json",args.ceilings]
    for scope in ("vlm","action_expert","vla","comparison"):
        selected=rows if scope=="comparison" else [r for r in rows if r["scope"]==scope]
        plot(selected,args.output/f"{scope}_overall",ceilings,
             f"π0.5 {scope.upper()} — whole GPU graph — Omni {report['framework']['commit'][:7]}",inputs)
    (args.output/"whole_summary.json").write_text(json.dumps({"analyzed_at":timestamp(),
        "framework":report["framework"],"ranges":rows,"unprofiled_gpu":baseline["gpu_graph_timings"],
        "unprofiled_pipeline_wall":baseline["pipeline_wall"]},indent=2)+"\n")


if __name__=="__main__":
    main()
