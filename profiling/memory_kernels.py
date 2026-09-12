"""Loaded only inside the GPU calibration path."""
import triton
import triton.language as tl


@triton.jit
def copy_l2(source, destination, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Bypass L1 so the read path matches the L2 traffic roof.
    values = tl.load(source + offsets, offsets < N, cache_modifier=".cg")
    tl.store(destination + offsets, values, offsets < N, cache_modifier=".wb")
