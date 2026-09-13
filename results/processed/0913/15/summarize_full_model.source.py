"""Report complete-pipeline numerical differences and plot final action errors."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from profiling.experiments import figure_record, file_record, timestamp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    root = args.input
    record = json.loads((root / "full_pipeline.json").read_text())
    diagnostics = json.loads((root / "numerics.json").read_text())
    labels = {"original": "原始优化版", "rope_model": "修正 RoPE 配对和频率",
              "rope_model_euler": "再还原 safe 输出投影 / Euler / FP32 状态",
              "rope_model_eager_euler": "再匹配 eager RoPE 中间舍入"}
    if "rope_model_image_snapshot" in record["cases"][0]["metrics"]:
        labels.update(rope_model_image_snapshot="RoPE 修正 + 保存每路相机图特征",
                      rope_model_images_eager="RoPE 修正 + eager 图像编码")
    names = {"legacy_masked_images": "历史缺图条件复现", "three_valid_images": "三路有效图像"}
    lines = [f"# {root.parent.name}/{root.name} — π0.5 整模型计算差异", "",
             f"硬件：{record['gpu']}；Torch：{record['torch']}。",
             f"完整 pipeline 采集：{record['started_at']} 至 {record['finished_at']}。",
             f"报告生成：{timestamp()}。", "",
             "所有表格比较 `Pi05Pipeline.forward` 的最终 `[50,32]` 动作，覆盖预处理、图像编码、"
             "前缀编码及全部 10 步去噪，优化路径包含实际 CUDA Graph 捕获和重放。",
             "同一真实 checkpoint、同一初始噪声、同一输入；batch=1、seed=17、BF16 权重和 FP32 初始噪声。",
             "相对 RMSE = RMSE(optimized−safe) / RMS(safe)。这些是归一化动作输出，没有真实动作标签。",
             "", "修正仅用于本次诊断进程；每次改动后重新捕获相关 CUDA Graph。没有修改上游源码或已有环境，也没有测量加速比。"]
    rows = []
    for case in record["cases"]:
        lines += ["", "## "+names[case["input"]], "", f"图像 mask：{case['image_masks']}。", "",
                  "| 完整 pipeline 配置 | 绝对 RMSE | 相对 RMSE | 最大绝对误差 | 平均绝对误差 |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for mode, label in labels.items():
            e = case["metrics"][mode]
            lines += [f"| {label} | {e['rmse']:.8f} | {100*e['relative_rmse']:.4f}% | {e['max_abs']:.8f} | {e['mean_abs']:.8f} |"]
            rows.append({"input": case["input"], "variant": mode,
                         **{k: e[k] for k in ("rmse", "relative_rmse", "reference_rms", "max_abs", "mean_abs")}})
        repeat = case["metrics"]["safe_repeat"]
        max_replay = max(case["metrics"][mode]["capture_vs_replay"]["max_abs"] for mode in labels)
        lines += ["", f"safe 重复运行最大绝对差：{repeat['max_abs']:.8f}；"
                  f"各优化配置捕获后首轮与重放最大绝对差的最大值：{max_replay:.8f}。"]
        if "image_audit" in case:
            audit = case["image_audit"]
            lines += ["", f"三路 CUDA Graph 图像输出是否共用同一存储：{audit['all_camera_outputs_share_storage']}。",
                      "三路被保留的图像输出相对最后一路的最大绝对差："
                      + str([e['max_abs'] for e in audit['retained_outputs_vs_last_camera']]) + "。",
                      "各路特征被覆盖相对于即时快照的相对 RMSE："
                      + str([round(100*e['relative_rmse'],4) for e in audit['overwritten_vs_snapshot_per_camera']]) + "% 。",
                      "只在 safe pipeline 中启用编译和图像 CUDA Graph，最终动作相对 RMSE："
                      f"{100*case['metrics']['safe_graph_images']['relative_rmse']:.4f}%；"
                      "每路立即保存特征后："
                      f"{100*case['metrics']['safe_graph_image_snapshot']['relative_rmse']:.4f}%。",
                      "图像快照和 eager 图像组以 RoPE 修正为基础，保持原 Euler；前缀与解码器仍使用 CUDA Graph。"]
    lines += ["", "## 独立控制组与解释", "",
              f"加载审计：{diagnostics['load_audit']['loaded_parameter_names']} 个参数名称已加载，"
              f"缺失 {len(diagnostics['load_audit']['missing_parameter_names'])} 个。",
              f"实际 RoPE 频率 dtype：expert `{diagnostics['rotary_audit']['expert_inv_freq_dtype']}`，"
              f"prefix `{diagnostics['rotary_audit']['prefix_inv_freq_dtype']}`。", ""]
    for case in diagnostics["cases"]:
        c = case["comparisons"]
        lines += [f"- {names[case['input']]}：只换优化前缀、保留 safe 解码器的相对 RMSE "
                  f"{100*c['fast_prefix_safe_decoder_vs_safe']['relative_rmse']:.4f}%；"
                  f"safe 去除被 mask token 的相对 RMSE {100*c['safe_trim_vs_full']['relative_rmse']:.4f}%；"
                  f"safe 预计算条件的相对 RMSE {100*c['safe_static_vs_dynamic']['relative_rmse']:.4f}%。"]
    lines += ["", "`numerics.json` 还单独记录了只改 RoPE 布局、只改 Euler、同时匹配模型频率等消融，"
              "以及每一步去噪误差和逐层 prefix KV 误差。它们共用 eager 图像 embeddings；上面主表则是完整 pipeline 实测。",
              "", "误差下降不等于通过部署容差；不同修正之间有交互，不能把误差差值直接相加为各 bug 的贡献。",
              "匹配 safe 的 BF16 频率只用于内部对齐；外部 OpenPI/LeRobot 正确性和机器人任务成功率尚未验证。",
              "", "完整结果：`full_pipeline.json`、`full_pipeline_actions.npz`、`full_pipeline_comparison.csv`。",
              "源码定位与单内核复现见 [0913/05 调查说明](../05/README.md)。", ""]
    with (root / "full_pipeline_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "README.md").write_text("\n".join(lines))
    if args.plot:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        inputs = (root / "full_pipeline.json", root / "full_pipeline_actions.npz")
        stamp = figure_record(root / "full_model_rmse", inputs)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        plot_modes = list(labels)[:4]
        plot_labels = ("Original optimized", "RoPE layout\n+ model frequency",
                       "+ safe Euler\n+ FP32 state", "+ eager RoPE\nrounding")
        if "rope_model_image_snapshot" in labels:
            plot_modes = ["original", "rope_model", "rope_model_image_snapshot", "rope_model_images_eager"]
            plot_labels = ("Original optimized", "RoPE layout\n+ model frequency",
                           "RoPE + preserve\neach camera output", "RoPE + eager\nimage encoder")
        for case, color, title in zip(record["cases"], ("#c96926", "#287d9b"),
                                      ("Missing cameras (historical reproduction)", "Three valid cameras")):
            values = [100*case["metrics"][mode]["relative_rmse"] for mode in plot_modes]
            ax.plot(range(4), values, "o-", color=color, label=title, linewidth=2)
            for i, value in enumerate(values):
                ax.annotate(f"{value:.3f}%", (i,value), xytext=(0,12 if color=="#c96926" else -18),
                            textcoords="offset points", ha="center", fontsize=9, color=color)
        ax.set(xticks=range(4), xticklabels=plot_labels,
               ylabel="Final-action relative RMSE vs safe (%)", yscale="log",
               title=f"π0.5 complete pipeline — {record['gpu']}")
        ax.grid(axis="y", alpha=.2)
        ax.margins(y=.16)
        ax.legend(loc="best", fontsize=9)
        fig.text(.5, .055, "Real fixed weights; synthetic observations; B=1; 10 denoising steps; normalized 50×32 actions.",
                 ha="center", fontsize=9)
        fig.text(.5, .02, stamp, ha="center", fontsize=8)
        fig.tight_layout(rect=(0,.11,1,1))
        for ext in ("png","pdf"):
            fig.savefig(root / f"full_model_rmse.{ext}", dpi=180)
        plt.close(fig)
    provenance = {"generated_at": timestamp(), "collection_started_at": record["started_at"],
                  "collection_finished_at": record["finished_at"],
                  "sources": [file_record(root / p) for p in ("full_pipeline.json", "full_pipeline_actions.npz", "numerics.json")]}
    (root / "full_summary.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
