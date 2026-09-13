"""Write measured ablation tables, retaining historical versus new collection times."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from profiling.experiments import file_record, timestamp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--report-name", default="README.md")
    args = p.parse_args()
    path = args.input
    r = json.loads((path / "numerics.json").read_text())
    import numpy as np
    labels = {
        "original": "原始优化解码器",
        "rope": "仅修正 RoPE 维度配对",
        "rope_model": "修正配对 + 匹配模型 RoPE 频率",
        "euler": "仅还原 safe 输出投影 / Euler / FP32 状态",
        "rope_euler": "修正配对 + 还原 Euler，频率仍为重新计算",
        "rope_model_euler": "配对 + 模型频率 + Euler",
        "rope_eager_euler": "配对 + eager RoPE 舍入 + Euler，频率仍为重新计算",
        "rope_model_eager_euler": "配对 + 模型频率 + eager RoPE 舍入 + Euler",
    }
    lines = [f"# {path.parent.name}/{path.name} — 真实权重数值消融", "",
             f"实际采集：{r['started_at']} 至 {r['finished_at']}。报告生成：{timestamp()}。",
             "", "本实验固定真实权重、初始噪声和观测，比较同一 PR 的 safe 与优化实现。",
             "共用 eager 图像 embeddings；不采集延迟。替换仅发生在诊断进程，不修改上游源码或现有环境。",
             "", "RoPE 错误的源码与最小 GPU 复现见 [调查说明](../05/README.md)。",
             "", f"模型实际 expert inv_freq dtype：`{r['rotary_audit']['expert_inv_freq_dtype']}`；"
             f"prefix inv_freq dtype：`{r['rotary_audit']['prefix_inv_freq_dtype']}`。",
             f"权重加载缺失参数：{r['load_audit']['missing_parameter_names']}。",
             "", "相对 RMSE 的分母是相同条件下 safe 动作的 RMS；没有真实动作标签。"]
    for c in r["cases"]:
        lines += ["", f"## {c['input']}", "",
                  f"图像 mask：{c['image_masks']}；原始 prefix 长度 {c['original_prefix_len']}，"
                  f"有效 prefix 长度 {c['valid_prefix_len']}。", "",
                  "| 解码器配置 | safe prefix KV：相对 RMSE | 优化 prefix KV：相对 RMSE | 优化 prefix KV：绝对 RMSE |",
                  "| --- | ---: | ---: | ---: |"]
        for mode, label in labels.items():
            a = c["comparisons"]["safe_prefix__"+mode]
            b = c["comparisons"]["fast_prefix__"+mode]
            lines += [f"| {label} | {100*a['relative_rmse']:.4f}% | {100*b['relative_rmse']:.4f}% | {b['rmse']:.8f} |"]
        lines += ["", "控制组：", ""]
        for key, label in (("safe_trim_vs_full", "safe 移除被 mask 的 token"),
                           ("safe_static_vs_dynamic", "safe 预计算条件替代逐步计算"),
                           ("fast_prefix_safe_decoder_vs_safe", "仅替换前缀编码器，保留 safe 解码器")):
            e = c["comparisons"][key]
            lines += [f"- {label}：相对 RMSE {100*e['relative_rmse']:.4f}%，最大绝对误差 {e['max_abs']:.8f}。"]
    historical = Path("results/raw/thor_pi05_20260913/baseline_retry/actions.npz")
    with np.load(path / "actions_and_traces.npz") as new, np.load(historical) as old:
        noise_equal = np.array_equal(new["legacy_masked_images__noise"], old["noise"])
        checks = {"noise_equal": noise_equal}
        for key, legacy_key in (("safe_full", "safe"), ("fast_prefix__original", "optimized")):
            x, y = new["legacy_masked_images__"+key][0].astype(np.float64), old[legacy_key].astype(np.float64)
            d = x-y
            checks[key] = {"rmse": float(np.mean(d*d)**0.5), "max_abs": float(np.abs(d).max())}
        if not noise_equal:
            raise ValueError("Historical reproduction used different initial noise")
    lines += ["", "## 历史结果复核", "",
              "历史采集属于 2026-09-13 的 baseline_retry；本次采集时间见文首，二者分别保留。",
              f"初始噪声逐元素相同：{noise_equal}。",
              f"本次完整 safe 与历史 safe 最大绝对差：{checks['safe_full']['max_abs']:.8f}。",
              f"本次原始优化路径与历史 optimized 最大绝对差：{checks['fast_prefix__original']['max_abs']:.8f}。",
              "", "## 解释边界", "",
              "这些是固定种子合成观测下的数值消融；不同改动存在交互，误差变化不能简单相加为归因百分比。",
              "匹配 safe 的 BF16 RoPE 频率用于内部对齐；对外部模型正确性仍需另行核对 OpenPI/LeRobot。",
              "即使相对误差下降，也不自动等于满足部署容差。原历史延迟不能用于这里改变后的计算路径。",
              "", "原始指标与逐层 prefix KV 差异：`numerics.json`；动作与逐步轨迹：`actions_and_traces.npz`。",
              "实际脚本快照：`diagnose_numerics.source.py`；加载与运行日志：`run.log`。", ""]
    metadata = {"generated_at": timestamp(), "historical_checks": checks,
                "sources": [file_record(p) for p in (path / "numerics.json", path / "actions_and_traces.npz", historical)]}
    (path / "summary.json").write_text(json.dumps(metadata, indent=2)+"\n")
    (path / args.report_name).write_text("\n".join(lines))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
