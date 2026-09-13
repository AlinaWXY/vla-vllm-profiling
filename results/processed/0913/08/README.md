# 0913/08 — VLM / Action Expert / VLA 整体基线

按用户新范围继续测量原始 PR4419，不修复其数值语义、不声称等价模型加速。
本实验使用 3 路匹配配置的图像键，并在运行时检查 image masks 全为 True 和有效 prefix 长度。

- VLM：3 路视觉编码、文本嵌入、拼接、前缀 Transformer。
- Action Expert：输入投影、10 步去噪、输出投影和 Euler 更新。
- VLA GPU 整体：以上两部分顺序执行；输入预处理和静态 timestep/AdaRMS 准备在测量范围外。
- VLA 端到端：原始 Pi05Pipeline.forward 的独立 wall time，包含预处理和输出拷贝，不含服务传输。

整体 GPU 时间分别用完整 CUDA Graph 测量，不能由逐 kernel NCU 时间求和替代。
分段重放与原 PR 的同次完整输出需匹配；此检查只保证采集没有换成另一条计算路径，
不等于与 safe / 外部 oracle 数值对齐。

正式基线等待 Thor 上另一任务结束；`queue.json` 记录排队和运行时间。
开始前用 `range_probe` 验证 NCU range replay 是否完整采到原生 BF16 和 JIT FP32 计数。
后续逐核和整体 NCU 采集将分别使用独立编号。
