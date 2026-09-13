L=918, D=2048, H=16384; S=256, V=1152, F=4304。

| ID | 形状 | FLOPs 估算公式 | 单独列出的特殊运算 |
| --- | --- | --- | --- |
| 1 | `gate/up [918,32768] -> [918,16384]` | `9*L*H add/mul + L*H tanh` | 15040512 tanh |
| 2 | `[918,16384] @ [16384,2048]` | `2*918*16384*2048 tensor; 1880064 scalar` |  |
| 3 | `[918,2048] @ [2048,32768]` | `2*918*2048*32768 tensor; 0 scalar` |  |
| 4 | `[7344,918] @ [918,256]` | `2*(8*L)*L*256` |  |
| 5 | `[7344,256] @ [256,918]` | `2*(8*L)*256*L tensor + 8*L*L scale` |  |
| 6 | `[918,2048] @ [2048,2048]` | `2*918*2048*2048 tensor; 1880064 scalar` |  |
| 7 | `positions [918], half-dim 128` | `3*L*128 mul before folding unit attention_scaling` | 117504 sin + 117504 cos |
| 8 | `[7344,918] softmax` | `R*(3*C-1), R=8*L, C=L: subtract + reduction + reciprocal multiply` | 6741792 exp + 7344 reciprocal + max comparisons |
| 9 | `[918,2048] @ [2048,2560]` | `2*918*2048*2560 tensor; 6345216 scalar` |  |
| 10 | `[918,2048]` | `L*(5*D+1): x^2, sum, mean, epsilon, (1+w), two output multiplies` | 918 rsqrt |
| 11 | `[918,2048]` | `L*(5*D+1): x^2, sum, mean, epsilon, (1+w), two output multiplies` | 918 rsqrt |
| 12 | `[1,968,2048] concatenate, then view first 918` | `0 (copy)` |  |
| 13 | `[918] mask conversion` | `0 (integer/copy)` |  |
| 14 | `one valid-length integer` | `0 (integer fill)` |  |
| 15 | `[200] -> [200,2048] embedding` | `0 (gather)` |  |
| 16 | `[200,2048] scale` | `200*D mul` |  |
| 17 | `[256,4304] GELU` | `8*S*F add/mul + S*F tanh` | 1101824 tanh |
| 18 | `[256,4304] @ [4304,1152]` | `2*256*4304*1152 tensor; 294912 scalar` |  |
| 19 | `[256,1152] @ [1152,4304]` | `2*256*1152*4304 tensor; 1101824 scalar` |  |
| 20 | `[1,16,256,72] Q/K/V, full attention` | `4*heads*S^2*d tensor; ~heads*S*(4*S-1) scalar` | 1048576 exp + 4096 reciprocal + max comparisons |
| 21 | `[1,3,224,224]` | `0 (layout copy)` |  |
| 22 | `[1152,3,14,14]` | `0 (layout copy)` |  |
| 23 | `[1,1152,16,16]` | `0 (layout copy)` |  |
| 24 | `[256,1152] @ [1152,1152]` | `2*256*1152*1152 tensor; 294912 scalar` |  |
| 25 | `[256,2048] scale` | `S*D mul` |  |
| 26 | `[256,1152] @ [1152,2048]` | `2*256*1152*2048 tensor; 524288 scalar` |  |
| 27 | `[1,3,224,224] F32 -> BF16` | `0 (cast/layout)` |  |
| 28 | `[1,3,224,224], 1152 filters 14x14, stride14` | `2*256*1152*(3*14*14)` |  |
| 29 | `[256,1152] split into 2304 groups of 128` | `2*S*V + 9*2304*(128-1)` | ~2304*127 Welford divisions |
| 30 | `256 rows x 9 partial Welford triples` | `9*256*(9-1)` | ~256*8 Welford divisions |
| 31 | `[256,1152] patch/LayerNorm normalize` | `8*S*V including broadcast variance scaling` | up to 294912 rsqrt before hoisting |
| 32 | `[256,1152], LayerNorm fusion 10` | `(2+4)*S*V + 9*S*(V-1) + 2*S` | ~294656 Welford divisions + 256 rsqrt |
| 33 | `[256,1152], LayerNorm fusion 4` | `(3+4)*S*V + 9*S*(V-1) + 2*S` | ~294656 Welford divisions + 256 rsqrt |
| 34 | `[256,1152], LayerNorm fusion 6` | `(2+4)*S*V + 9*S*(V-1) + 2*S` | ~294656 Welford divisions + 256 rsqrt |
| 35 | `[256,1152], LayerNorm fusion 7` | `(4+4)*S*V + 9*S*(V-1) + 2*S` | ~294656 Welford divisions + 256 rsqrt |
| 36 | `[256,1152], LayerNorm fusion 8` | `(6+4)*S*V + 9*S*(V-1) + 2*S` | ~294656 Welford divisions + 256 rsqrt |
| 37 | `[256,1152], LayerNorm fusion 9` | `(4+4)*S*V + 9*S*(V-1) + 2*S` | ~294656 Welford divisions + 256 rsqrt |
