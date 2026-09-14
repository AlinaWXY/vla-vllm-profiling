# π0.5 · L20 DRAM Roofline

比较 Omni 与 SGLang 的 **VLM / Action Expert 大算子 DRAM Roofline**。当前仓库只保留下面两组已审计结果，FFN up/down 分开，坐标采用 10、10²、10³ 的十进制对数刻度。

| 结果 | 固定实现 | Slurm 作业 | 报告 |
| --- | --- | --- | --- |
| Omni · 0914/15 | [`6bdbf97`](https://github.com/AlinaWXY/vllm-omni/tree/6bdbf97e357232b8aa3b6caf5a6c8b0fed713c25) | 1506597 | [说明与数据](results/processed/0914/15/README.md) · [PDF](results/processed/0914/15/major_operators.pdf) |
| SGLang · 0914/16 | [`5200508`](https://github.com/sgl-project/sglang/tree/5200508b0fd25733752c2c5e3af5539508023c27) | 1506879 | [说明与数据](results/processed/0914/16/README.md) · [PDF](results/processed/0914/16/major_operators.pdf) |

## Omni

![Omni L20 DRAM Roofline](results/processed/0914/15/major_operators.png)

## SGLang

![SGLang L20 DRAM Roofline](results/processed/0914/16/major_operators.png)

## 实验条件与读图

两组均在 NVIDIA L20 上通过 Slurm 采集，使用相同的 `lerobot/pi05_base` 权重，revision `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`。输入为 batch 1、3 路有效的合成 224×224 相机图像、prefix 918、10 步去噪、50×32 动作。SGLang 与 Omni 的预处理图像、tokens、masks 已逐项核验一致。

- 主图统计 BF16 Tensor 运算，FP32 累加；FP32 辅助操作单独保留在数据中。
- 实线为 L20 标称 119 TFLOP/s BF16、864 GB/s DRAM；虚线为各作业内 GEMM/copy 实测参考。
- 实心点表示利用率达到报告筛选标准，支持对应瓶颈判断；空心点只表示所处 roofline 区域，不能证明资源饱和。
- SGLang 的 QK、softmax、PV 使用融合 kernel，合为一个 attention 点。NCU 采集采用 eager 调度以保留 NVTX，原生 CUDA Graph 的整体延迟另行测量。
- NCU kernel 耗时之和不等于端到端延迟；两个框架存在融合边界、软件版本和计时范围差异，本仓库不据此宣称数值等价加速。

完整口径见 [方法说明](docs/methodology.md)。模型来源见 [版本锁定](sources.lock.json)。

## 数据核验与重绘

分析只需 Python、NumPy 和 Matplotlib，不加载模型或使用 GPU。在独立环境安装 [分析依赖](requirements/analysis.txt)，然后运行：

```bash
python -m profiling.verify_results
output_dir=$(python -m profiling.experiments --purpose 'Replot retained Omni L20 DRAM roofline')
python -m profiling.replot --source results/processed/0914/15 --output "$output_dir"
```

SGLang 重绘将 `--source` 换成 `results/processed/0914/16`，并先分配新的输出目录。采集时间与重绘时间分别记录，已发布图片不会被覆盖。

每组结果附带原始 NCU CSV 的无损压缩副本、逐 kernel 数据、算子汇总、Slurm 记录、重放一致性证据及实际采集代码。详见 [复现说明](docs/reproduce.md)。大型 `.ncu-rep` 保存在本地实验归档，仓库记录其 SHA256。

旧的 Thor/L2 结果、失败或中止的实验、早期绘图及数值诊断入口已从当前版本移除。本地实验和 Git 历史保留追溯记录；当前结果目录仅发布 `0914/15` 与 `0914/16`。
