# π0.5 整模型数值差异：已确认的原因

**实测支持“这个优化分支存在实现错误”。确认了 Action Expert 的 RoPE 配对错误，以及多相机 CUDA Graph 输出被覆盖。** 结论针对固定的实验性 PR4419（`1826509403bfa3d378d3476a4a341b78640165ea`），不能推广为所有 vLLM-Omni 版本。

## 完整模型结果

L20 / Slurm 1490549，完整 pipeline 实际采集时间为 **2026-09-13T16:53:22+08:00 至 2026-09-13T16:53:32+08:00**。包括预处理、三路图像编码、语言前缀、全部 10 步 Action Expert 去噪和最终动作输出；优化路径实际捕获并重放 CUDA Graph。模型加载和分段控制实验时间另见 `collection.json` / `numerics.json`，不是这个完整 pipeline 阶段的起止时间。

同一真实 checkpoint、同一组有效相机输入、同一 FP32 初始噪声；BF16 权重，batch=1，seed=17。表中均比较最终 `[50,32]` 归一化动作与 safe 输出，分母 RMS(safe)=0.1340099132。

| 完整 pipeline 配置 | RMSE | 相对 RMSE | 最大绝对差 |
| --- | ---: | ---: | ---: |
| 原始优化后端 | 0.22256906 | 166.0840% | 0.72293675 |
| 修正 RoPE 配对并匹配模型频率 | 0.00417133 | 3.1127% | 0.02228448 |
| 再保存每路相机图像特征 | 0.00134986 | 1.0073% | 0.00645256 |
| RoPE 修正 + eager 图像编码控制组 | 0.00118975 | 0.8878% | 0.00567603 |

保存相机特征的组仍保留编译后的图像编码和图像 CUDA Graph，只在每路编码后复制输出。eager 图像组保留优化前缀与解码器 CUDA Graph。两组都保持原优化 Euler 路径；没有把多项改动的误差差值相加作为归因比例。

safe 的两次运行逐元素相同，各优化组捕获后的首次输出与下一次重放也逐元素相同。原始组及原有修正组还与独立实验 `0913/14` 一致。全部 813 个参数名称已加载，缺失为 0。

## 1. Action Expert RoPE 配对写错

- [realtime_triton.py:727](https://github.com/vllm-project/vllm-omni/blob/1826509403bfa3d378d3476a4a341b78640165ea/vllm_omni/diffusion/models/pi05/realtime_triton.py#L727) 把投影结果 reshape 后分成相邻的 `(0,1), (2,3), …`。
- safe 使用 Gemma 的 `rotate_half`，256 维 head 对应 `(0,128), (1,129), …`。
- [同文件的权重打包](https://github.com/vllm-project/vllm-omni/blob/1826509403bfa3d378d3476a4a341b78640165ea/vllm_omni/diffusion/models/pi05/realtime_triton.py#L1696) 直接拼接原始 Q/K/V 的转置，没有对应的特征重排。

因此 Q/K 特征旋转的配对和频率对应关系改变了。这改变注意力计算的数学含义，不能解释为普通 BF16 舍入。共用 eager 图像特征的控制实验中，三路有效图像的最终动作相对 RMSE 从原始 166.1589% 降至只修正布局的 1.9293%，再匹配模型频率后为 0.9177%；这是控制实验的最终动作，不替代上表的完整 pipeline 结果。

还有频率来源差异：[解码器重新计算 FP32 inv_freq](https://github.com/vllm-project/vllm-omni/blob/1826509403bfa3d378d3476a4a341b78640165ea/vllm_omni/diffusion/models/pi05/realtime_triton.py#L1736)，实际模型的 expert/prefix inv_freq 则随模型转换为 BF16。运行时审计确认了 dtype。匹配这里的 BF16 频率是对齐 safe 的对照，不是在宣称该频率精度符合外部 OpenPI oracle。

## 2. 三路相机图像特征被最后一路覆盖

[图像 CUDA Graph cache key](https://github.com/vllm-project/vllm-omni/blob/1826509403bfa3d378d3476a4a341b78640165ea/vllm_omni/diffusion/models/pi05/modeling_pi05.py#L2013) 按形状、dtype、设备与编译设置共享。同形状的三路相机复用一个 graph。每次调用 [返回同一个 static_output](https://github.com/vllm-project/vllm-omni/blob/1826509403bfa3d378d3476a4a341b78640165ea/vllm_omni/diffusion/models/pi05/modeling_pi05.py#L2088)，而 `embed_prefix` 在 [追加特征](https://github.com/vllm-project/vllm-omni/blob/1826509403bfa3d378d3476a4a341b78640165ea/vllm_omni/diffusion/models/pi05/modeling_pi05.py#L2300) 后，要到循环结束才 [拼接](https://github.com/vllm-project/vllm-omni/blob/1826509403bfa3d378d3476a4a341b78640165ea/vllm_omni/diffusion/models/pi05/modeling_pi05.py#L2311)。后续相机重放会覆盖此前保存的张量内容。

真实模型观测确认：三路返回张量指向同一存储，最后三者与第三路特征的最大绝对差均为 0。第一、第二路被覆盖后相对于其即时正确快照的相对 RMSE 分别是 **17.9104%、18.2932%**。

只给 safe pipeline 的图像编码启用编译和图像 CUDA Graph，最终动作相对 RMSE 已为 **3.0756%**；每路输出立即保存后降至 **0.7957%**。这与完整优化路径中修正相机输出后从 3.1127% 降至 1.0073% 的结果共同验证了覆盖问题。

## 剩余误差与历史结果

- 只还原输出投影、Euler 顺序和 FP32 状态，对原始 166% 差异几乎没有改善。在完整 pipeline 修正 RoPE 后，Euler 控制为 3.1351%，未优于 3.1127%；所以没有把 Euler 认定为主要原因。
- 共用 eager 图像特征时，仅替换前缀、保留 safe 解码器的最终动作相对 RMSE 为 0.7929%；safe 仅移除被屏蔽 token 的差异为 0.6963%。这些控制表明仍有低精度计算和执行路径差异，未逐项完全归因，不能把约 1% 残差直接声明为可接受。
- 历史 Thor 的 103.7042% 来自三路图像均被 mask 的输入。本次 L20 同条件复现为 103.3719%；该条件下保存相机输出不会改变最终动作。三路有效图像的 166.0840% 属于新的有效输入条件。硬件、采集时间和输入条件分别保留。
- 本次只验证一个固定种子的合成观测与真实权重。没有真实动作标签、外部 OpenPI/LeRobot 对齐或机器人任务成功率结论，也没有修正后的性能测量。

## 可复核记录

- [完整 pipeline 表格](README.md)、[最终动作 CSV](full_pipeline_comparison.csv)、[最终动作张量](full_pipeline_actions.npz)、[完整指标及相机存储审计](full_pipeline.json)。
- [分段控制实验表](ablations.md)、`numerics.json`、`actions_and_traces.npz`：包含两种输入、两种前缀、八种解码配置及 10 步动作轨迹。
- `submission.json`、`run.slurm`、`slurm_1490549.log`、两个诊断脚本快照，以及 `verification.json` 保存资源、源码和数值复核信息。
- 诊断替换只存在于测试进程，不修改固定上游源码。为适配 L20 共享内存，新增的修正 RoPE 内核使用 `block_k=32,num_stages=2`；原始优化内核与原始前缀内核的配置保持不变。
- 图的编号和绘图时间见 `full_model_rmse.plot.json`；图使用本目录新采集数据，没有覆盖历史图。
