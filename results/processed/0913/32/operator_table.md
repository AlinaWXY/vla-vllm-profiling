| ID | 算子 | 次数 | 逻辑 FLOPs/次 (GF) | NCU BF16 / FP32 (GF/次) | 平均耗时 μs | 耗时占比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Prefix FFN: GELU product | 17 | 0.135365 | 0 / 0.230911 | 343.563 | 6.89% |
| 2 | Prefix FFN: down + residual | 17 | 61.6078 | 61.7402 / 0.00376832 | 786.869 | 15.79% |
| 3 | Prefix FFN: gate/up projection | 17 | 123.212 | 124.554 / 0.0608174 | 859.569 | 17.24% |
| 4 | Prefix: Attention P × V | 17 | 3.4518 | 3.61759 / 0 | 266.040 | 5.34% |
| 5 | Prefix: Attention QKᵀ + scale | 17 | 3.45854 | 3.497 / 0.00683008 | 255.720 | 5.13% |
| 6 | Prefix: Attention output + residual | 17 | 7.70262 | 8.05306 / 0 | 215.390 | 4.32% |
| 7 | Prefix: Prefix RoPE table | 1 | 0.000352512 | 0 / 0.00467849 | 18.784 | 0.02% |
| 8 | Prefix: Prefix masked softmax | 17 | 0.020218 | 0 / 0.0358689 | 146.552 | 2.94% |
| 9 | Prefix: QKV projection + Gemma RoPE | 18 | 9.63227 | 9.73079 / 0.00641434 | 255.236 | 5.42% |
| 10 | Prefix: RMSNorm (pre-FFN) | 17 | 0.00940124 | 0 / 0.0103404 | 29.316 | 0.59% |
| 11 | Prefix: RMSNorm (pre-attention) | 18 | 0.00940124 | 0 / 0.0103404 | 29.136 | 0.62% |
| 12 | Prefix: concatenate image/text embeddings | 1 | 0 | 0 / 0 | 28.000 | 0.03% |
| 13 | Prefix: mask conversion/copy | 1 | 0 | 0 / 0 | 15.872 | 0.02% |
| 14 | Prefix: valid-length fill | 1 | 0 | 0 / 0 | 12.288 | 0.01% |
| 15 | Text: embedding lookup | 1 | 0 | 0 / 0 | 18.560 | 0.02% |
| 16 | Text: embedding scale | 1 | 0.0004096 | 0 / 0.0004096 | 20.064 | 0.02% |
| 17 | Vision: FFN GELU | 81 | 0.00881459 | 0 / 0.015103 | 24.674 | 2.36% |
| 18 | Vision: FFN contraction + bias | 81 | 2.5389 | 2.56691 / 0.000884736 | 76.798 | 7.34% |
| 19 | Vision: FFN expansion + bias | 81 | 2.5397 | 2.64241 / 0.00344064 | 60.982 | 5.83% |
| 20 | Vision: FlashAttention | 81 | 0.30618 | 0.402653 / 0.00507904 | 31.572 | 3.02% |
| 21 | Vision: NCHW → NHWC (input) | 3 | 0 | 0 / 0.000150528 | 28.437 | 0.10% |
| 22 | Vision: NCHW → NHWC (weights) | 3 | 0 | 0 / 0.000677376 | 104.437 | 0.37% |
| 23 | Vision: NHWC → NCHW | 3 | 0 | 0 / 0.000294912 | 18.400 | 0.07% |
| 24 | Vision: attention Q/K/V/O projections | 324 | 0.679772 | 0.679477 / 0.000884736 | 28.189 | 10.78% |
| 25 | Vision: image embedding scale | 3 | 0.000524288 | 0 / 0.000524288 | 16.971 | 0.06% |
| 26 | Vision: image-to-text projection + bias | 3 | 1.20848 | 1.20796 / 0.00157286 | 42.645 | 0.15% |
| 27 | Vision: input cast/layout | 3 | 0 | 0 / 0 | 12.736 | 0.05% |
| 28 | Vision: patch embedding convolution | 3 | 0.346817 | 3.69938 / 0.000884736 | 101.013 | 0.36% |
| 29 | Vision: patch/LayerNorm fusion 1 | 3 | 0.0032233 | 0 / 0.00382464 | 20.011 | 0.07% |
| 30 | Vision: patch/LayerNorm fusion 2 | 3 | 1.8432e-05 | 0 / 0.000140288 | 13.824 | 0.05% |
| 31 | Vision: patch/LayerNorm fusion 3 | 3 | 0.0023593 | 0 / 0.0018432 | 24.203 | 0.09% |
| 32 | Vision: residual/LayerNorm fusion 10 | 3 | 0.00442189 | 0 / 0.00480217 | 24.427 | 0.09% |
| 33 | Vision: residual/LayerNorm fusion 4 | 3 | 0.0047168 | 0 / 0.00629146 | 26.389 | 0.09% |
| 34 | Vision: residual/LayerNorm fusion 6 | 39 | 0.00442189 | 0 / 0.00521248 | 21.859 | 1.01% |
| 35 | Vision: residual/LayerNorm fusion 7 | 39 | 0.00501171 | 0 / 0.0053919 | 24.248 | 1.12% |
| 36 | Vision: residual/LayerNorm fusion 8 | 39 | 0.00560154 | 0 / 0.00661914 | 27.205 | 1.25% |
| 37 | Vision: residual/LayerNorm fusion 9 | 39 | 0.00501171 | 0 / 0.0219059 | 29.453 | 1.36% |
