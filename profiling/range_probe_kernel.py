"""Known FP32 work for checking SASS counters inside an NCU graph/range."""
import triton
import triton.language as tl


@triton.jit
def fma_probe(x, y, output, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(x + i, i < N, other=0)
    b = tl.load(y + i, i < N, other=0)
    tl.store(output + i, tl.fma(a, 2.0, b), i < N)
