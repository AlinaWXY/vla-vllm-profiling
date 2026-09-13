# 0913/30 — VLM 与历史 Action Expert 分别绘图

按用户要求，本次仅解析已有报告并绘图，**没有运行模型、启动或续跑 NCU 采集**。
逐算子实验 [0913/28](../28/README.md) 和未启动的整体实验 [0913/29](../29/README.md) 保持取消。
每张图下方记录 `0913/30` 和香港时间，`*.plot.json` 保存绘图时间、输入文件及 SHA256。

| 项目 | VLM | Action Expert |
| --- | --- | --- |
| 数据来源 | 取消的 0913/28 中已完成的 VLM 阶段 | 已完成的历史 Action Expert breakdown |
| Omni 版本 | `6bdbf97`，含数值修正 | `1826509`，旧 PR 实现 |
| 实际图像 / 前缀 | 三路有效图像，918 token | 三路图像被 mask，150 token |
| 覆盖 | 3 个视觉编码器、文字 embedding、拼接、18 层前缀 | 10 步去噪、18 层 expert、输入/输出投影 |
| kernel 调用数 / 分组数 | 1,018 / 37 | 1,654 / 16 |
| NCU kernel 耗时之和 | 84.741056 ms | 57.699360 ms |
| 各 kernel 的 L2 流量之和 | 50.220652 GB | 17.822035 GB |
| BF16 Tensor 计数 | 4.289449 TFLOP | 0.417155 TFLOP |
| FP32 add/mul/FMA 计数 | 10.092614 GFLOP | 1.250647 GFLOP |

两套数据的源码与实际输入不同，**不相加为一次 VLA，不作为两模块性能优劣或新旧实现加速比的直接比较**。
这里的耗时是清缓存的 NCU 重放耗时；不能替代 [0913/27](../27/README.md)
未插桩基线的 VLM 69.054 ms / Action Expert 41.949 ms / VLA GPU 111.005 ms（p50）。
本次没有测得整体图的 L2 流量，因此不生成 VLM、Action Expert 或 VLA 的整阶段 roofline。

## VLM

![VLM 分算子 roofline](vlm/roofline_by_operator.png)

[高清 PDF](vlm/roofline_by_operator.pdf) · [每次调用散点图](vlm/roofline.png) ·
[每次调用 PDF](vlm/roofline.pdf) · [算子清单](vlm/operator_legend.csv) ·
[算子汇总与耗时](vlm/operators.csv) · [逐调用数据](vlm/kernels.csv) · [覆盖审计](vlm/summary.json)

图内编号与清单一致。37 组按阶段、调用位置、kernel 名称、grid/block 分开；
相同配置的重复层和相机调用使用 `ΣF/Σbytes`、`ΣF/Σtime` 汇总，不平均各次比率。
例如 3 为 Prefix FFN gate/up 投影，2 为 down + residual，24 为视觉 Q/K/V/O 投影，
20 为视觉 FlashAttention，9 为前缀 QKV + Gemma RoPE。
无计入浮点运算的拷贝/填充保留在表中，不在对数坐标中伪造零值点。

VLM 的逐调用 ID 为 **0–1017**，下一条 ID 1018 已进入 Action Expert。
三个相机各 280 次调用，名称、顺序及 launch geometry 完全一致，每相机含 27 个视觉 attention 层；
随后 2 次文字 embedding、1 次拼接，以及 175 次 prefix 调用均完整。
Prefix 的 122 个手写 Triton launch 与固定源码行号和层顺序逐条一致；
17 个 FFN 各有 3 次 kernel 调用。第 18 层只生成所需 KV，源码有意跳过该层后续 attention/FFN。
所有 1,018 次调用的六项计数器均完整，后续不完整的 651 个 expert 调用全部排除。

原采集区间为 2026-09-13 19:33:12–20:19:35 +08:00，实际 kernel 采集从 19:37:24 开始；
没有每个 kernel 的墙钟时间，不能把整个区间当作 VLM 实际运行时长。
NCU 报告可成功导出；见 [导出记录](export/export.json)、[原始报告](../28/ncu/vla.ncu-rep)、
[采集日志](../28/ncu/ncu.log)。报告 SHA256 为 `f45a7c0e91fbac4aab44c206f18f5b96aa02bf2d43931fc83a093f38b7ee9596`。

由于完整 VLA 采集被中断，28 没有保存末尾动作数组及运行时 manifest。
因此覆盖判断使用实际 NVTX/调用序列和冻结源码，不能声称存在完整的 28 输出校验文件。
日志表明采集前已通过模型指纹和分段输出误差阈值检查；独立基线 27 记录了分段与原 pipeline 输出完全相同。
[校验记录](validation.json) 确认缺失调用、缺失计数器、相机 geometry 不一致和源码行不匹配均会拒绝出图。

## 历史 Action Expert

![历史 Action Expert 分算子 roofline](action_expert/roofline_by_operator.png)

[高清 PDF](action_expert/roofline_by_operator.pdf) · [每次调用散点图](action_expert/roofline.png) ·
[每次调用 PDF](action_expert/roofline.pdf) · [算子清单](action_expert/operator_legend.csv) ·
[算子汇总与耗时](action_expert/operators.csv) · [逐调用数据](action_expert/kernels.csv) ·
[来源与原始时间](action_expert/source.json)

原采集：2026-09-13 13:08:14–13:50:29 +08:00；复用完整的 1,654 次调用，包含 1,650 个手写
Triton launch。旧源码与缺失视觉前缀的限制保留在图内及表中，本次没有修改历史数值。

## 如何看参考线

两个精度域分别显示：BF16 Tensor 与 FP32 add/mul/FMA。一个融合 kernel 可在两个面板出现，
但耗时和字节数只能计一次；整数、超越函数和未选浮点指令不计入这两个 FLOP 域。
字节数均来自 `lts__t_bytes.sum`，只画 L2 算术强度，不冒充 L1 或 DRAM 流量。

线沿用 [此前微基准](../../compute_counter_validation/ceilings_l2.json) 的 **114.088 TFLOP/s BF16、
6.375 TFLOP/s FP32、954.716 GB/s L2 copy**。这些是经验参考，**不是本轮验证过的硬件上限**；
旧校准没有保留完整频率上限记录，不能假定与本次状态严格一致。

当前 Prefix FFN gate/up 的池化吞吐已达 **144.903 TFLOP/s**，高于旧 BF16 参考；
Prefix QKV 的 L2 流量/时间为 **1,349.070 GB/s**，也高于旧 copy 参考。
这说明旧参考不能包住所有这次测量点；图保留原始数据，不移动点、不反向拟合或抬高参考线。
同样，低于参考线本身不证明 L2 饱和、DRAM/L1 受限或 vLLM 没优化好。
此次按用户要求停止采集，不新增硬件校准实验。

## 复现绘图（仅 CPU 解析已有文件）

先分配新日期编号，不能覆盖这次图片。原始 CSV 已随本目录发布，重画无需调用 NCU。

```bash
source scripts/cache_env.sh
python3 -m profiling.experiments --purpose '已有 VLM 与历史 Action Expert 报告重画'
# 将下面的 NEW_RESULT 替换为刚分配的新目录。
python3 -m profiling.recover_vlm --capture results/processed/0913/28 \
  --export results/processed/0913/30/export --baseline results/processed/0913/27 \
  --ceilings results/processed/compute_counter_validation/ceilings_l2.json \
  --output NEW_RESULT/vlm
python3 -m profiling.roofline results/processed/thor_pi05_20260913/ncu/raw.csv \
  --operators-csv results/processed/thor_pi05_20260913/ncu/operators.csv \
  --contract results/processed/thor_pi05_20260913/metrics.json \
  --ceilings results/processed/compute_counter_validation/ceilings_l2.json \
  --caption 'Historical Omni 1826509 | 10 denoising steps | image masks false, prefix=150 | NOT the new 3-camera workload' \
  --output NEW_RESULT/action_expert
```
