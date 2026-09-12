# 在其他机器准备 Thor 环境

用户于 2026-09-12 确认：在其他机器准备 ARM64 环境，写入共享目录，再由
Thor 执行。现有环境不变，不在 Thor 运行 pip/uv 安装或框架源码构建。
模型和 tokenizer 的运行入口继续强制使用本地文件。

## 两个架构分别处理

- CPU/Python：aarch64、CPython 3.12、Ubuntu 24.04 的 glibc 2.39。
  x86 主机只解析依赖、下载/展开 ARM64 wheel；解释器使用 Thor 已有的
  `/usr/bin/python3.12`。共享环境不包含 x86 Python。
- GPU：Thor SM 11.0。新增 CUDA 扩展使用 `TORCH_CUDA_ARCH_LIST=11.0a`；
  直接调用 NVCC/PTXAS 时对应 `-arch=sm_110a`。运行时 Triton 指向 Thor 已有的
  CUDA 13.2 PTXAS，编译并发默认 1。Triton 3.6 对 SM >= 100 使用独立的 PTXAS 配置，因此启动脚本
  同时设置 `TRITON_PTXAS_PATH` 和 `TRITON_PTXAS_BLACKWELL_PATH`。预编译 wheel 的架构由其上游构建决定，
  设置环境变量不会重新编译或改变 wheel 中已有的二进制。

[NVIDIA NVCC 13.2 文档](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-compiler-driver-nvcc/index.html)
分别描述了主机代码与设备代码的编译，并列出 `sm_110a` 目标。
[vLLM 安装文档](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
提供 ARM64 wheel 和 CUDA 版本索引。

## 环境内容与准备入口

`vllm-omni/docker/Dockerfile.cuda` 使用 `vllm/vllm-openai:v0.22.0`。
因此将 `vllm/` 从之前尚未验证的主线提交对齐到 v0.22.0：
`0b3ba88f165976e77ca5e6a7a3f5bba4562b80af`。下载同一提交的官方 ARM64 CUDA 13
wheel，并解析 PyTorch 2.11.0+cu130、Triton 3.6.0、Transformers < 5.9。
两份 Omni 源码保持原来的固定提交。

`requirements/thor-cu130.lock` 固定解析出的包版本；
`requirements/thor-cu130.in` 包含 vLLM 的完整依赖和 pi0.5 导入路径所需的
Omni 依赖。Omni 直接通过共享源码导入；这不是包含所有 Omni 音频/视频模型的
完整通用安装。唯一预先打包的源码依赖 `antlr4-python3-runtime==4.9.3` 是
纯 Python wheel；其余包禁止从源码构建，避免误生成 x86 本机扩展。

在非 Thor 主机、项目根目录执行：

```bash
bash scripts/prepare_thor_env.sh --dry-run
bash scripts/prepare_thor_env.sh
```

输出目录为 `.venv-thor-cu130/`，不进入 Git。包/工具缓存位于
`/scratch/$USER/vla-vllm-profiling/`。准备过程固定下载并发 2、安装并发 1，
不导入 Torch、不申请 GPU、不启动原生大规模编译。若将来需要大规模源码
构建，其他主机仍遵守 Slurm 调度规则。

在 Thor 使用现有解释器运行共享包：

```bash
bash scripts/thor_python.sh -c 'import torch; print(torch.__version__, torch.version.cuda)'
bash scripts/thor_python.sh -m profiling.bench_pi05 --help
```

启动脚本隔离系统/user site，设置共享 Omni 源码、离线模型读取、scratch 缓存
和编译并发。Thor 上执行推理时必要的 Triton/torch.compile JIT 仍会产生缓存；
这是运行过程的一部分，区别于在 Thor 安装或构建整个框架。

## sm_110a 外部编译探针

`profiling/sm110a_probe.ptx` 是一个输出 1024 个浮点数的小算子。
可在 x86 构建主机使用 CUDA 13.2 的 PTXAS：

```bash
ptxas -arch=sm_110a profiling/sm110a_probe.ptx -o /fact_data/用户/结果/probe.cubin
```

随后在 Thor 用其现有 Python/驱动加载产物：

```bash
python3 -B -m profiling.probe_cubin /fact_data/用户/结果/probe.cubin \
  --output /fact_data/用户/结果/probe.json
```

探针检查 1024 个输出值并记录 cubin SHA-256；显式设备数据分配只有 4096 bytes，
但 CUDA context/模块仍有驱动开销。它不验证 pi0.5，不测量部署延迟，也不生成
VLA roofline 点。真实模型、优化 kernel 和 NCU 的验证需分别执行。

实际验证：2026-09-12，`deep-space`（x86_64）上的 PTXAS 13.2.86 生成的
`sm_110a` cubin 已在 `fact-thor`（aarch64 / NVIDIA Thor）执行通过。
构建命令、源码/产物校验和及运行检查分别保存在
`results/processed/environment_sm110a/build.json` 和 `probe.json`。

共享运行环境的 CUDA driver、CuTe DSL、FlashInfer 和 TVM-FFI 缓存也显式指向
`/scratch`，避免这些库在 home 或 `/tmp` 使用其默认数据缓存目录。

完整运行验证已通过：PyTorch 2.11.0+cu130、vLLM 0.22.0、Triton 3.6.0、
Transformers 5.8.1；BF16 matmul、Triton `sm_110a` 运算、vLLM 原生扩展及
`Pi05Pipeline` / `Pi05RealtimeTritonDecoder` 导入均成功。记录见
`results/processed/environment_sm110a/runtime.json`。首次共享盘读入较慢，最初的
3 分钟检查超时；延长观察后完成，未在 Thor 补装依赖。

环境标记为 READY 后，准备脚本拒绝修改它。运行入口仍使用
`bash scripts/thor_python.sh`；模型权重和 tokenizer 仍未加载。
