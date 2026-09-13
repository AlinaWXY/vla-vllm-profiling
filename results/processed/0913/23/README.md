# 0913/23 — 新 Omni profiling（待运行）

源码：6bdbf97；方式：整体 CUDA Graph profiling，cache-control none；VLM / Action Expert / VLA 三个图。

等待 0913/20 基线和 0913/24 已知计算量整图探针通过后串行采集。
原始参数在 run.sh，计数器约定在 metrics.json；尚无实测性能结果。
