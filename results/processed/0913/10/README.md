# 0913/10 — kernels

等待 0913/08 的 GPU 基线和采集路径检查通过后执行。
完整范围：VLM + Action Expert；3 路有效图像，10 步去噪。
静态输入预处理及 timestep/AdaRMS 准备不计入 GPU 核心范围。
