# 0913/27 — 新 Omni：VLM / Action Expert / VLA 基线

框架 `6bdbf97e357232b8aa3b6caf5a6c8b0fed713c25`；Thor；真实原始 F32 checkpoint 加载后由框架转 BF16。
三路有效 224×224 合成图像，batch 1，seed 17，10 次去噪，输出 50×32；有效 prefix=918。
进程首条日志：2026-09-13 19:28:25（Asia/Hong_Kong）。GPU 工作准备与测量：2026-09-13T19:32:04+08:00 至 2026-09-13T19:32:23+08:00。
本说明生成：2026-09-13T19:37:55+08:00。

| GPU 范围 | p50 (ms) | p95 (ms) |
| --- | ---: | ---: |
| vlm | 69.054 | 70.287 |
| action_expert | 41.949 | 42.070 |
| vla | 111.005 | 111.200 |

各项 20 次样本。原始 Pi05Pipeline.forward wall time：p50 **113.410 ms**，p95 114.455 ms，包含预处理和结果回传，无服务传输。

GPU 范围是从同一已编译 vision callable 和原框架的 prefix/expert kernels 重新捕获的分段图。包含静态图像输入拷贝和修正后的相机输出 clone；预处理、tokenizer、静态 timestep/AdaRMS 准备与输出回传不在 GPU 图内。完整原始 pipeline 的耗时独立报告。

分段直接执行和分段 CUDA Graph 相对原始 pipeline 的最大输出差均为 **0**。这验证采集拆分没有改变本输入下的模型输出，不代表对 safe 或外部 oracle 的误差为零。813 个参数名称全部加载，缺失 0；实际模型参数 SHA256：`a1a0b85c81588a70b3ee0af5905f6dea15e6fbee639c16c0d19e3353ca624bb2`。

GPU 峰值 allocated=14.507 GiB，reserved=14.865 GiB。全部样本、输出、相机 masks、参数指纹及源码记录在 workload/。冻结的采集代码在 harness/，运行参数在 run.sh。
