# π0.5 数值修正版与复现

框架源代码位于 [AlinaWXY/vllm-omni 的 codex/pi05-numerical-fixes 分支](https://github.com/AlinaWXY/vllm-omni/tree/codex/pi05-numerical-fixes)，提交 `6bdbf97e357232b8aa3b6caf5a6c8b0fed713c25`。本项目的子模块和 `sources.lock.json` 固定到该提交，原始上游版本仍记录为 `1826509403bfa3d378d3476a4a341b78640165ea`。

修改包括：

1. Action Expert 调用已有的 Gemma 半区配对 RoPE 内核，删除未使用的相邻配对内核。
2. 解码器使用实际模型 rotary 模块生成 cos/sin，匹配其频率 buffer 和缩放。
3. 多相机图像编码在拼接之前保存每路 CUDA Graph 输出，避免共享缓冲区把前两路覆盖为最后一路。
4. 新增相机输出生命周期、模型频率表及非零位置 fused QKV 的回归测试。

修正内核默认 `PI05_REALTIME_DECODER_QKV_BLOCK_K=32`、`num_stages=2`，可满足 L20 的共享内存限制。RoPE 使用固定的两个 128 维半区，旧 `PI05_REALTIME_DECODER_QKV_BLOCK_N` 调节项不再使用。没有改变 Euler 路径，也没有测量此版本的加速比。

## 实测结果和限制

[0913/15](../results/processed/0913/15/README.md) 是原始分支加诊断替换的整模型消融；[0913/21](../results/processed/0913/21/README.md) 是实际提交源码的运行。两者均使用真实权重、固定 seed=17、三路合成图像和 10 步去噪，覆盖整个 `Pi05Pipeline.forward`。

原始优化版相对 RMSE 为 166.0840%；诊断组合修正为 1.0073%；实际源码为 1.0047%。实际源码与同进程的原始解码器加修正控制逐元素一致。独立运行之间存在小幅差异，已保留其指标，没有把跨运行逐元素一致作为结论。历史缺图输入仍有 3.6133% 相对残差，约 1% 也不代表达到部署容差。未验证外部 OpenPI/LeRobot oracle、机器人任务成功率或修正后的 Thor 性能。

## 复现实际源码

使用已有兼容环境；x86 CUDA 13 环境依赖记录在 `requirements/numerics-x86-cu130.in`。框架可从项目子模块加载：

```bash
source scripts/cache_env.sh
export PYTHONPATH="$PWD/vllm-omni:$PWD"
python vllm-omni/tests/pi05/test_realtime_numerical_contracts.py
experiment_dir=$(python -m profiling.experiments --purpose 'pi05 actual source numerical regression')
python -m profiling.validate_framework \
  --checkpoint /你的固定本地模型目录 \
  --tokenizer /你的固定本地tokenizer目录 \
  --output "$experiment_dir" \
  --diagnostic results/processed/0913/15
```

GPU 工作应在已授权 Thor 或 Slurm 分配内运行，缓存放 `/scratch`。无需下载新模型；同一 checkpoint/tokenizer 的固定身份见 `sources.lock.json`。没有框架数值补丁注入到这里的实际源码测量。

可选 `--baseline-source /原始PR4419/realtime_triton.py` 增加同进程对照；只为控制组载入原始解码器并应用已验证的 RoPE 替换。框架主测量仍使用真实提交源码。完整示例与资源、路径见 `results/processed/0913/21/run.slurm`。

`profiling.diagnose_numerics`、`profiling.diagnose_full_model` 和 `profiling.check_rope` 用于原始 PR4419 的错误定位，必须在原始上游源码 checkout 下运行，不应对当前已删除旧内核的修正版调用。诊断脚本、运行条件和输出已保存在对应实验目录；重新采集或绘图应分配新的日期编号。

项目 benchmark 也修正了相机键名，并在推理前验证真实图像 mask，避免再次静默生成缺图基线。
