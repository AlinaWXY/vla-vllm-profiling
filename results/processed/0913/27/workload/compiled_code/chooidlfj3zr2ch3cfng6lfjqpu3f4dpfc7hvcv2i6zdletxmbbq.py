# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /scratch/xinyaowang/vla-vllm-profiling/inductor/qm/cqmenrb7zitt2vhgkdnexcm4mwckqvsdunwxuqrw55kp2cjxoj3r.py
# Topologically Sorted Source Nodes: [chunk, gelu, hidden], Original ATen: [aten.split, aten.gelu, aten.mul]
# Source node to ATen node mapping:
#   chunk => split
#   gelu => add, add_1, convert_element_type_2, convert_element_type_3, mul, mul_1, mul_2, mul_3, mul_4, mul_5, tanh
#   hidden => mul_6
# Graph fragment:
#   %mm : Tensor "bf16[918, 32768][32768, 1]cuda:0" = PlaceHolder[target=mm]
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%mm, 16384, -1), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[918, 16384][16384, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem, torch.float32), kwargs = {})
#   %mul_4 : Tensor "f32[918, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, 0.5), kwargs = {})
#   %mul : Tensor "f32[918, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %convert_element_type_2), kwargs = {})
#   %mul_1 : Tensor "f32[918, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %convert_element_type_2), kwargs = {})
#   %mul_2 : Tensor "f32[918, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, 0.044715), kwargs = {})
#   %add : Tensor "f32[918, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, %mul_2), kwargs = {})
#   %mul_3 : Tensor "f32[918, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 0.7978845608028654), kwargs = {})
#   %tanh : Tensor "f32[918, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_3,), kwargs = {})
#   %add_1 : Tensor "f32[918, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1), kwargs = {})
#   %mul_5 : Tensor "f32[918, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_1), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[918, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.bfloat16), kwargs = {})
#   %mul_6 : Tensor "bf16[918, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, %getitem_1), kwargs = {})
#   return %mul_6
triton_poi_fused_gelu_mul_split_0 = async_compile.triton('triton_poi_fused_gelu_mul_split_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_mul_split_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '9C3337B899FEA3CB29E8ADD9D131EBEE0EC33016577309EAF4B44AE3452087DC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 120324096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_mul_split_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15040512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 16384)
    x1 = xindex // 16384
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32768*x1), None).to(tl.float32)
    tmp16 = tl.load(in_ptr0 + (16384 + x0 + 32768*x1), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp1 * tmp1
    tmp5 = tmp4 * tmp1
    tmp6 = tl.full([1], 0.044715, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = tl.full([1], 0.7978845608028654, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tl.full([1], 1.0, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        assert_size_stride(arg0_1, (918, 2048), (2048, 1))
        assert_size_stride(arg1_1, (2048, 32768), (32768, 1))
        assert_size_stride(arg2_1, (16384, 2048), (2048, 1))
        assert_size_stride(arg3_1, (918, 2048), (2048, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((918, 32768), (32768, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [gate_up], Original ATen: [aten.mm]
            extern_kernels.mm(arg0_1, arg1_1, out=buf0)
            del arg0_1
            del arg1_1
            buf1 = empty_strided_cuda((918, 16384), (16384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [chunk, gelu, hidden], Original ATen: [aten.split, aten.gelu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_mul_split_0.run(buf0, buf1, 15040512, stream=stream0)
            del buf0
            buf2 = empty_strided_cuda((918, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [chunk, gelu, hidden], Original ATen: [aten.split, aten.gelu, aten.mul, aten.addmm]
            extern_kernels.addmm(arg3_1, buf1, arg2_1, alpha=1, beta=1, out=buf2)
            del arg2_1
            del arg3_1
            del buf1
        return (buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((918, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((2048, 32768), (32768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((16384, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((918, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
