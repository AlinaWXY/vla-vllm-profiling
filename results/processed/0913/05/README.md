# 0913/05 — π0.5 safe / realtime_triton 数值差异定位

完整模型结果已补充于 **[0913/15](../15/README.md)**：L20 三路有效图像的最终动作相对 RMSE 为原始 166.084%、修正 RoPE 3.113%、再修正相机 CUDA Graph 输出覆盖 1.007%。两处错误和整模型控制实验见 [最终定位说明](../15/findings.md)。下文保留本次调查早期的源码分析与历史 Thor 结果；Thor `0913/07` 的进度独立记录，不作为 L20 已完成结果的采集时间。

已确认 PR4419 的 Action Expert RoPE 实现存在语义错误：Gemma 的半区配对被改成了相邻元素配对，且 Q/K 权重没有相应重排。它改变了模型的注意力计算，不能归为普通 BF16 舍入误差。

本目录是 2026-09-13 Asia/Hong_Kong 的独立调查记录。编号创建时间见 `experiment.json`。本目录的初始消融在模型加载前停止，原因是继续检查发现频率精度还需独立控制；记录见 `collection.json`。扩展真实权重实验另行分配为 **[0913/07](../07/)**，其采集时间单独记录。`0913/06` 因共享 NFS 逐页加载过慢而在权重加载阶段停止，没有产生模型输出；`0913/07` 改为校验后的匿名内存读取，保留原有模型和实验条件。没有绘图时，不将分析时间写为绘图时间。最小 GPU 复现来自独立实验 `0913/04`，采集时间为 **2026-09-13T15:27:39+08:00 至 2026-09-13T15:32:49+08:00**。

## 先核实误差含义

历史文件 `results/raw/thor_pi05_20260913/baseline_retry/actions.npz` 保存了同一真实权重、同一初始噪声、10 步去噪的 `[50,32]` 输出：

| 指标 | 数值 |
| --- | ---: |
| RMSE(optimized − safe) | 0.05937523 |
| RMS(safe) | 0.05725442 |
| 相对 RMSE | 103.7042% |
| 最大绝对误差 | 0.19186890 |

相对 RMSE 定义为 `sqrt(mean((optimized-safe)^2)) / sqrt(mean(safe^2))`。这里比较的是两种实现的归一化动作输出，并没有真实动作标签。103.7% 不是“任务失败率”，也不是两种算法各自对真实动作的 RMSE。分母较小会放大百分比，但实际差异已达到 safe 输出本身的幅度，不能仅用分母解释。

历史采集的相机键另有一处输入问题：短名称没有匹配模型要求的 `observation.images.*`，实际三路图像均被 mask。现有输入检查已经修正此问题；本次同时重现历史 mask 条件和三路有效图像，以免把两种输入混为一谈。这一输入问题本身不能解释同一观测下两个后端的 RoPE 语义不同。

重新计算及原始文件 SHA256 见 `historical_metrics.json`。原数据采集时间属于 2026-09-13 的历史 baseline，原始日志和内存采样继续保留；本次分析时间不替代它。

## 已证实的错误：RoPE 布局不匹配

固定源码：`vllm-omni` commit `1826509403bfa3d378d3476a4a341b78640165ea`。

- `modeling_pi05.py:1443` 调用 Transformers Gemma 的 `apply_rotary_pos_emb`。本地 Transformers 5.8.1 的 `modeling_gemma.py:165` 使用 `rotate_half`：head_dim=256 时配对 `(0,128), (1,129), …`。
- `realtime_triton.py:727` 将矩阵乘结果 reshape 成 `[..., block_n//2, 2]` 后 split，配对变成 `(0,1), (2,3), …`。
- `realtime_triton.py:1696` 的权重打包仅连接 Q/K/V 的转置，并没有将 Gemma 的半区布局转换成相邻布局。
- 同文件 `:1506` 的前缀编码器使用 `_matmul_gemma_rope_qkv`，仍然是 Gemma 布局。因而不仅 suffix 自身的位置编码变了，suffix Q 与 prefix K 也不在兼容的旋转坐标下。

数学上，Gemma 对 `(x_i,x_{i+128})` 使用同一个角度：`(x_i cosθ − x_{i+128} sinθ, x_{i+128} cosθ + x_i sinθ)`。优化解码器把上述另一半分量换成相邻分量，频率与特征的对应也随之改变。仅更改 cos/sin 表的排列不能修复；必须同时满足 Q/K 特征布局和输出布局契约。

最小 GPU 对照使用随机但固定种子的 BF16 输入/权重，shape 为 50 tokens、1024 输入宽度、8 个 Q heads、head_dim 256。对照使用相同的投影矩阵与 Gemma RoPE 公式；同时检查不参与旋转的 V 和全零位置控制组。

| 位置 | 原优化 Q 相对 RMSE | 原优化 K 相对 RMSE | Gemma 内核 Q | Gemma 内核 K |
| --- | ---: | ---: | ---: | ---: |
| 全部为 0 | 0% | 0% | 0% | 0% |
| 150–199 | 130.889% | 131.467% | 0.328% | 0.322% |
| 918–967 | 134.737% | 132.969% | 0.329% | 0.336% |

所有组的 V 都完全一致。全零位置时 RoPE 是恒等变换，原内核也完全一致；非零位置才出现巨大差异。这把问题隔离到了 RoPE，而不是一般矩阵乘、随机噪声、权重加载或 CUDA Graph。Gemma 融合内核残留约 0.33% 的差异来自与 eager BF16 路径不同的中间舍入步骤；它仍不意味着整模型已经数值对齐。

可复现入口：`python -m profiling.check_rope --output <新的实验目录>`，需通过现有 Thor 运行环境执行。原始记录见 `../04/rope_check.json`。

## 真实权重消融的设计

运行入口为 `python -m profiling.diagnose_numerics --checkpoint <固定本地权重> --tokenizer <固定本地 tokenizer> --output <新的实验目录>`。扩展实验实际进展保留在 `../07/run.log`，脚本快照与开始时间也保存在该目录。

固定 batch=1、seed=17、10 步去噪、BF16 权重、FP32 初始噪声，并共用 eager 图像 embeddings。分别比较历史缺图条件与三路有效图像。每种条件下检查：

1. safe 原始 padding 与移除被 mask token 的差异；safe 动态与预计算 timestep/AdaRMS 的差异。
2. 只把前缀换成优化编码器，后续保持 safe 解码器。
3. 在同一份 safe 或优化 prefix KV 上，运行原始解码器、仅修正 RoPE 配对、同时匹配模型频率、仅还原 Euler 路径，以及组合替换和额外匹配 eager RoPE 中间舍入。

源码还显示频率来源不一致：`pipeline_pi05.py:290` 对整个模型调用 `.to(dtype=bfloat16)`，会连同 `inv_freq` buffer 一起转换；safe 从此 buffer 计算 RoPE，前缀 Triton 内核也读取同一 buffer。解码器 `realtime_triton.py:1739` 则重新计算 FP32 的 `10000**(-2i/256)`。例如 FP32 频率约 `0.69783056` 舍入到 BF16 后为 `0.69921875`；乘以位置 918，角度差约 1.274 弧度。这是公式演算，实际模型 buffer 的 dtype 和差异由 `0913/07` 的 `rotary_audit` 记录验证。匹配 safe 的频率是内部对齐对照，不表示 BF16 频率就是外部模型 oracle 应采用的精度。

Euler 的独立疑点是：safe 保留 FP32 的 `x_t`，而优化解码器在 `realtime_triton.py:1823` 将初始状态复制进 BF16 buffer，后续每步均回写 BF16；`:1789` 还提前把 `dt` 乘入 BF16 输出权重和 bias。它们在实数中可以变换，但在低精度运算中并不严格等价。本实验将其作为独立消融，不预先断言其贡献大小。

替换只发生在诊断进程中，不改动固定上游源码或已有环境；不测量加速比。扩展实验数据保存在 `../07/numerics.json` 和 `../07/actions_and_traces.npz`，逐步轨迹用于核对误差传播。

## 测试覆盖与结论边界

固定版本的 `tests/pi05/test_pi05_e2e.py:180` LeRobot 对齐测试调用默认 `sample_actions`，没有启用 `use_realtime_triton_decoder` / `use_realtime_triton_prefix_encoder`。因此该测试即使通过，也不能证明优化路径正确。随机或真实模型输出“形状正确、全为有限值”同样不足以证明数值一致。

当前结论针对这个实验性 PR 的 π0.5 Triton 后端。其 [PR #4419](https://github.com/vllm-project/vllm-omni/pull/4419) 为已关闭的 WIP，不能将此结论推广成所有 vLLM-Omni 实现均有同一问题。Gemma 布局也可对照 [Transformers 5.8.1 官方源码](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/models/gemma/modeling_gemma.py)。

已确认至少一处改变模型数学语义的实现错误。单内核复现不能给出它对整模型 103.7% 误差的精确归因比例，也不能替代对外部 OpenPI/LeRobot oracle 或真实机器人任务的验证。修正后仍需重新评估数值误差及性能，原有延迟比不能视为等价模型的加速比。
