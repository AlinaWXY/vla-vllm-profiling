# vla-vllm-profiling

在 NVIDIA Thor 上部署 vLLM-Omni π0.5，采集 Action Expert 的逐 kernel timing、
FLOP 与 DRAM 流量，并生成可追溯的 Roofline 图。

**当前状态（2026-09-11）：准备阶段。源码和采集/分析脚本已就绪；Thor 当前故障，
实测暂停。尚未安装 Thor 运行环境、执行 GPU 推理、获得 NCU 数据或生成实测
Roofline。本仓库先保存现有代码与实验方法，待 Thor 恢复后补充实测结果。**

## 代码来源与实验边界

| 目录 | 用途 | 固定版本 |
| --- | --- | --- |
| `vllm/` | vLLM 框架源码；运行时版本兼容性待 Thor 验证 | `69db1c26b4fe4474ab4c9df1c9701efac8bedde1` |
| `vllm-omni/` | π0.5 `realtime_triton_prefix` 优化实现 | `1826509403bfa3d378d3476a4a341b78640165ea` |
| `vllm-omni-reference/` | 新 π0.5 功能实现及 LeRobot 对齐 oracle | `41a6da68fcb2da7c2717dda32069ef9541797bbe` |

优化实现来自已关闭、未合并的 [PR #4419](https://github.com/vllm-project/vllm-omni/pull/4419)，
不能称为 vLLM 主线正式支持。新的 [PR #6950](https://github.com/vllm-project/vllm-omni/pull/6950)
提供功能实现及对齐测试，但不包含该 Triton/CUDA Graph 优化。两分支的 Transformers
版本约束不同，参考验证应使用单独环境。上游 PR 报告的性能与对齐结果不是本项目实测。

模型固定为 [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base)，revision
`b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`，约 14.47 GB 的 FP32 权重。
采用真实权重与可复现的合成观测，默认 batch=1、3 路 224×224 图像、10 次去噪、
输出 `[50, 32]`。这用于性能分析；不构成机器人任务成功率验证。

源码依赖和权重版本记录在 `sources.lock.json`。源码重建入口：

```bash
python3 scripts/bootstrap_sources.py
```

该脚本校验已有目录，遇到本地修改或不同版本时会停止。上游源码目录未并入
项目 Git 历史；固定版本引用与重建脚本随本项目代码一并保存。

## Thor 环境准备

通过 SSH 执行只读检查：

```bash
ssh thor0 'bash -s' < scripts/inspect_thor.sh
```

根据返回的操作系统、Python、CUDA、PyTorch、vLLM、Triton、NCU 和 Slurm 配置，
确定安装与资源申请参数。不要把 x86 CUDA wheel 装到 ARM Thor 上，也不要默认当前
vLLM main 与旧 Omni PR 可直接搭配。**尚未给出已验证的安装命令或 Slurm 分区。**

所有 GPU 作业均须在 Slurm allocation 内执行。以下 GPU 入口会检查 `SLURM_JOB_ID`。
当前没有提交任何作业，也没有编写未经确认的 Slurm 任务脚本。

源码、虚拟环境和结果放在 `/fact_data`；可重建缓存和权重放在用户自己的 `/scratch`：

```bash
source scripts/cache_env.sh
# 在依赖安装完成后的 Python 环境中：
python scripts/download_checkpoint.py
```

需要 PaliGemma tokenizer 的本地目录或可访问的模型 ID；如果 tokenizer 需要 Hugging Face
访问授权，使用已有正常认证流程。不要把凭据、权重或缓存提交到 GitHub。

## 采集流程

以下命令是 **Slurm allocation 内部的工作负载命令**；须先完成 Thor 环境验证。
`$CHECKPOINT` 和 `$TOKENIZER` 指向已就绪的真实模型、tokenizer。运行前 source
`scripts/cache_env.sh`。所有 `results/` 相对路径以本项目的 `/fact_data` 目录为根。

1. 查询 Thor 上实际支持的 NCU 指标：

   ```bash
   python -m profiling.ncu discover --output results/raw/ncu_inventory
   ```

   如果 Thor 没有脚本认识的 BF16 Tensor 计数器，入口会报错并保留完整查询输出。
   此时需对照该 NCU 安装中的 Tensor Roofline section 更新指标约定，不能用 0 代替。

2. 独立测量不受 NCU 插桩影响的延迟：

   ```bash
   python -m profiling.bench_pi05 --checkpoint "$CHECKPOINT" --tokenizer "$TOKENIZER" \
       --output results/raw/baseline --cameras 3 --steps 10 --iterations 20
   ```

   保存 safe/优化 pipeline wall time、独立 Action Expert 的 CUDA Graph 与普通 launch
   device time、输出动作、相同 PR 内两种后端的数值差异。这里调用 `Pi05Pipeline`
   本体，不包含 websocket 传输或服务端调度开销。外部 LeRobot 对齐仍需单独执行。

3. 采集同一优化实现的 Action Expert kernel：

   ```bash
   python -m profiling.ncu capture --contract results/raw/ncu_inventory/metrics.json \
       --output results/raw/ncu_expert -- \
       python -m profiling.bench_pi05 --checkpoint "$CHECKPOINT" --tokenizer "$TOKENIZER" \
       --output results/raw/profile_workload --cameras 3 --steps 10 --iterations 3 --profile
   ```

   使用真实前缀 KV 和预计算 AdaRMS；NVTX 定位去噪步、层号与 kernel 源码位置。
   为逐算子归因，诊断阶段关闭 CUDA Graph launch，但不更换优化后的 Triton kernel。
   初次 NCU 采集是逐 kernel replay、清缓存、`clock-control none`。
   **NCU 下产生的 benchmark timing 不用于部署性能结论。**

4. 在同一硬件状态下测量参考带宽/算力：

   ```bash
   python -m profiling.calibrate --output results/raw/ceilings.json \
       --power-clock-note '填写实际记录的 Thor 功耗模式、频率及记录文件位置'
   ```

   这些是 GEMM/流式 copy 达到的经验参考值，不是经过证明的硬件最大上限。
   也可提供有出处且精度匹配的硬件理论值；不能把 FP4 TOPS 当成 BF16 TFLOP/s。

5. 生成逐 kernel 表和图：

   ```bash
   python -m profiling.roofline results/raw/ncu_expert/raw.csv \
       --operators-csv results/raw/ncu_expert/operators.csv \
       --contract results/raw/ncu_inventory/metrics.json \
       --ceilings results/raw/ceilings.json --output results/processed/thor_pi05
   ```

   输出 `kernels.csv`、`summary.json`、`roofline.png` 和 `roofline.pdf`。
   缺失计数器、零 DRAM 流量或零浮点运算的 kernel 保留在表中并注明原因，不绘制虚假点。
   FP32 与 BF16 的点按精度分开；同一 kernel 可出现在多个精度面板，耗时不能重复求和。

## 校验与后续发布

本机 CPU 侧的数据处理测试：

```bash
python3 -m unittest discover -s tests -v
```

运行环境依赖由 Thor 的实际安装决定。分析/绘图只需要 Python、NumPy 和 Matplotlib；
NCU CSV 解析本身只使用标准库。方法、结果口径及当前待办见 `docs/methodology.md`
和 `docs/status.md`。

`vla-vllm-profiling` 仓库先保存本项目完整代码、固定上游版本与实验方法。
完成实测后补充环境信息、命令、算子 CSV、图和结论。原始 `.ncu-rep` 如超过 GitHub 单文件限制则作为
Release 附件保存，并在结果文档中记录链接及校验和。
