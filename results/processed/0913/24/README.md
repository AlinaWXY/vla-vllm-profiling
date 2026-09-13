# 0913/24 — NCU 整图计数验证

Thor 的 `--graph-profiling graph --replay-mode kernel --cache-control none` 采集成功，仅需一轮硬件计数。
已知工作为一个原生 BF16 GEMM + 一个 Triton FP32 FMA。BF16 Tensor 实测和理论计算量均为 2,147,483,648 FLOP。
输出断言全部通过。记录见 verification.json、range_probe/workload.json 和原始 NCU 报告。

FP32 指令计数器不支持整图模式；未用零代替。完整模型整体图的 FP32 分子将从匹配的逐 kernel 记录取得，并校验 BF16 总量一致。
