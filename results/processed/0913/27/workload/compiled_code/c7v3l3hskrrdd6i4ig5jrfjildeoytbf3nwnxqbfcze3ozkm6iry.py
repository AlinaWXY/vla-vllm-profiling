# AOT ID: ['0_inference']
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


# kernel path: /scratch/xinyaowang/vla-vllm-profiling/inductor/be/cbenbi4ow6nrnsclejpewizhnnu4ozlfs5g7bdwqcnqn2ucoctrt.py
# Topologically Sorted Source Nodes: [pixel_values, patch_embeds], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   patch_embeds => convolution
#   pixel_values => convert_element_type
# Graph fragment:
#   %arg0_1 : Tensor "f32[1, 3, 224, 224][3, 1, 672, 3]cuda:0" = PlaceHolder[target=arg0_1]
#   %buf0 : Tensor "bf16[1, 3, 224, 224][672, 1, 672, 3]cuda:0" = PlaceHolder[target=buf0]
#   %convert_element_type : Tensor "bf16[1, 3, 224, 224][3, 1, 672, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[1, 1152, 16, 16][294912, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type, %arg1_1, %arg2_1, [14, 14], [0], [1, 1], False, [0], 1), kwargs = {})
#   return %buf0,%buf1
triton_poi_fused__to_copy_convolution_0 = async_compile.triton('triton_poi_fused__to_copy_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '9C3337B899FEA3CB29E8ADD9D131EBEE0EC33016577309EAF4B44AE3452087DC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 225792, 'x': 602112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_0(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 3*x1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr1 + (x1 + 50176*y0), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: /scratch/xinyaowang/vla-vllm-profiling/inductor/zq/czqsgfnct5vrkdjqigtnjitkgfeeawt7f7pxkchc5w724beejzx5.py
# Topologically Sorted Source Nodes: [pixel_values, patch_embeds, flatten, embeddings, embedding, embeddings_1, hidden_states], Original ATen: [aten._to_copy, aten.convolution, aten.view, aten.transpose, aten.embedding, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embedding => embedding
#   embeddings => permute
#   embeddings_1 => add
#   flatten => view
#   hidden_states => clone, convert_element_type_1, var_mean
#   patch_embeds => convolution
#   pixel_values => convert_element_type
# Graph fragment:
#   %buf2 : Tensor "bf16[1, 1152, 16, 16][294912, 256, 16, 1]cuda:0" = PlaceHolder[target=buf2]
#   %arg2_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "i64[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "bf16[256, 1152][1152, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %convert_element_type : Tensor "bf16[1, 3, 224, 224][3, 1, 672, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[1, 1152, 16, 16][294912, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type, %arg1_1, %arg2_1, [14, 14], [0], [1, 1], False, [0], 1), kwargs = {})
#   %view : Tensor "bf16[1, 1152, 256][294912, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution, [1, 1152, 256]), kwargs = {})
#   %permute : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1]), kwargs = {})
#   %embedding : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %arg3_1), kwargs = {})
#   %add : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %embedding), kwargs = {})
#   %clone : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_1 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_1, [2]), kwargs = {correction: 0, keepdim: True})
#   return %buf3,%buf4,%buf5
triton_red_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_1 = async_compile.triton('triton_red_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 3, 'num_reduction': 3, 'backend_hash': '9C3337B899FEA3CB29E8ADD9D131EBEE0EC33016577309EAF4B44AE3452087DC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 647168, 'r0_': 2304}}
)
@triton.jit
def triton_red_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2304
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tl.full([1, 1], 256, tl.int32)
        tmp5 = tmp3 + tmp4
        tmp6 = tmp3 < 0
        tmp7 = tl.where(tmp6, tmp5, tmp3)
        tl.device_assert(((0 <= tmp7) & (tmp7 < 256)) | ~(xmask), "index out of bounds: 0 <= tmp7 < 256")
        tmp9 = tl.load(in_ptr3 + (r0_2 + 128*x1 + 1152*tmp7), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tmp2 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight, roffset == 0
        )
        tmp13_mean = tl.where(r0_mask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(r0_mask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(r0_mask & xmask, tmp13_weight_next, tmp13_weight)
    tmp14, tmp15, tmp16 = triton_helpers.welford(tmp13_mean, tmp13_m2, tmp13_weight, 1)
    tmp13 = tmp14[:, None]
    tmp17 = tmp15[:, None]
    tmp18 = tmp16[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
    tl.store(out_ptr2 + (x3), tmp18, xmask)
''', device_str='cuda')


# kernel path: /scratch/xinyaowang/vla-vllm-profiling/inductor/6j/c6j5gva3iaqni5hvuc3jtfva5cdfpgupfrvmt6sqsxzee3w3wr4r.py
# Topologically Sorted Source Nodes: [pixel_values, patch_embeds, flatten, embeddings, embedding, embeddings_1, hidden_states], Original ATen: [aten._to_copy, aten.convolution, aten.view, aten.transpose, aten.embedding, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embedding => embedding
#   embeddings => permute
#   embeddings_1 => add
#   flatten => view
#   hidden_states => clone, convert_element_type_1, var_mean
#   patch_embeds => convolution
#   pixel_values => convert_element_type
# Graph fragment:
#   %buf3 : Tensor "f32[1, 256, 1, 9][2304, 1, 2304, 256]cuda:0" = PlaceHolder[target=buf3]
#   %buf4 : Tensor "f32[1, 256, 1, 9][2304, 1, 2304, 256]cuda:0" = PlaceHolder[target=buf4]
#   %buf5 : Tensor "f32[1, 256, 1, 9][2304, 1, 2304, 256]cuda:0" = PlaceHolder[target=buf5]
#   %convert_element_type : Tensor "bf16[1, 3, 224, 224][3, 1, 672, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[1, 1152, 16, 16][294912, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type, %arg1_1, %arg2_1, [14, 14], [0], [1, 1], False, [0], 1), kwargs = {})
#   %view : Tensor "bf16[1, 1152, 256][294912, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution, [1, 1152, 256]), kwargs = {})
#   %permute : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1]), kwargs = {})
#   %embedding : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %arg3_1), kwargs = {})
#   %add : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %embedding), kwargs = {})
#   %clone : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_1 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_1, [2]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf7
triton_per_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_2 = async_compile.triton('triton_per_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 16},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '9C3337B899FEA3CB29E8ADD9D131EBEE0EC33016577309EAF4B44AE3452087DC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 31744, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 9
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 256*r0_1), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 256*r0_1), r0_mask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /scratch/xinyaowang/vla-vllm-profiling/inductor/6h/c6h2lbhrxdltbcu7dkgusxjbebpx4lwd7ccqlqwsront7fq2jxfc.py
# Topologically Sorted Source Nodes: [pixel_values, patch_embeds, flatten, embeddings, embedding, embeddings_1, hidden_states], Original ATen: [aten._to_copy, aten.convolution, aten.view, aten.transpose, aten.embedding, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embedding => embedding
#   embeddings => permute
#   embeddings_1 => add
#   flatten => view
#   hidden_states => add_1, add_2, clone, convert_element_type_1, convert_element_type_2, mul, mul_1, rsqrt, sub, var_mean
#   patch_embeds => convolution
#   pixel_values => convert_element_type
# Graph fragment:
#   %buf2 : Tensor "bf16[1, 1152, 16, 16][294912, 256, 16, 1]cuda:0" = PlaceHolder[target=buf2]
#   %arg2_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "i64[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "bf16[256, 1152][1152, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %getitem_1 : Tensor "f32[1, 256, 1][256, 1, 256]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf7 : Tensor "f32[1, 256, 1][256, 1, 256]cuda:0" = PlaceHolder[target=buf7]
#   %arg5_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convert_element_type : Tensor "bf16[1, 3, 224, 224][3, 1, 672, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[1, 1152, 16, 16][294912, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type, %arg1_1, %arg2_1, [14, 14], [0], [1, 1], False, [0], 1), kwargs = {})
#   %view : Tensor "bf16[1, 1152, 256][294912, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution, [1, 1152, 256]), kwargs = {})
#   %permute : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1]), kwargs = {})
#   %embedding : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %arg3_1), kwargs = {})
#   %add : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %embedding), kwargs = {})
#   %clone : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_1 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_1, %getitem_1), kwargs = {})
#   %add_1 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg5_1), kwargs = {})
#   %add_2 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg6_1), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
#   return %convert_element_type_2
triton_poi_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_3 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '9C3337B899FEA3CB29E8ADD9D131EBEE0EC33016577309EAF4B44AE3452087DC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 593920, 'x': 1186560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 1152
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp24 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tl.full([1, 1], 256, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert(((0 <= tmp7) & (tmp7 < 256)) | ~(ymask), "index out of bounds: 0 <= tmp7 < 256")
    tmp9 = tl.load(in_ptr3 + (x1 + 1152*tmp7), xmask & ymask).to(tl.float32)
    tmp10 = tmp2 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 - tmp12
    tmp15 = tl.full([1, 1], 1152.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tmp17 = tl.full([1, 1], 1e-06, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 * tmp22
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 + tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr0 + (x1 + 1152*y0), tmp27, xmask & ymask)
''', device_str='cuda')


# kernel path: /scratch/xinyaowang/vla-vllm-profiling/inductor/up/cupzhsvyapsz6didt5y2cqsrlqisxqldwvpnrzoxawmakmwjheb7.py
# Topologically Sorted Source Nodes: [pixel_values, patch_embeds, flatten, embeddings, embedding, embeddings_1, attn_output_3, hidden_states_1, hidden_states_2], Original ATen: [aten._to_copy, aten.convolution, aten.view, aten.transpose, aten.embedding, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_3 => view_12
#   embedding => embedding
#   embeddings => permute
#   embeddings_1 => add
#   flatten => view
#   hidden_states_1 => add_3
#   hidden_states_2 => add_4, add_5, clone_1, convert_element_type_15, convert_element_type_16, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   patch_embeds => convolution
#   pixel_values => convert_element_type
# Graph fragment:
#   %buf2 : Tensor "bf16[1, 1152, 16, 16][294912, 256, 16, 1]cuda:0" = PlaceHolder[target=buf2]
#   %arg2_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "i64[1, 256][256, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "bf16[256, 1152][1152, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %addmm_3 : Tensor "bf16[256, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %add_3 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0" = PlaceHolder[target=add_3]
#   %getitem_12 : Tensor "f32[1, 256, 1][256, 1, 256]cuda:0" = PlaceHolder[target=getitem_12]
#   %buf22 : Tensor "f32[1, 256, 1][256, 1, 256]cuda:0" = PlaceHolder[target=buf22]
#   %arg15_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %arg16_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %convert_element_type : Tensor "bf16[1, 3, 224, 224][3, 1, 672, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[1, 1152, 16, 16][294912, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type, %arg1_1, %arg2_1, [14, 14], [0], [1, 1], False, [0], 1), kwargs = {})
#   %view : Tensor "bf16[1, 1152, 256][294912, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution, [1, 1152, 256]), kwargs = {})
#   %permute : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1]), kwargs = {})
#   %embedding : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %arg3_1), kwargs = {})
#   %add : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %embedding), kwargs = {})
#   %view_12 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, 256, 1152]), kwargs = {})
#   %add_3 : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_12), kwargs = {})
#   %clone_1 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_3,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_15 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_1, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_15, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_15, %getitem_12), kwargs = {})
#   %add_4 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_11, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg15_1), kwargs = {})
#   %add_5 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg16_1), kwargs = {})
#   %convert_element_type_16 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.bfloat16), kwargs = {})
#   return %add_3,%getitem_12,%buf22,%convert_element_type_16
triton_red_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_4 = async_compile.triton('triton_red_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 2048},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '9C3337B899FEA3CB29E8ADD9D131EBEE0EC33016577309EAF4B44AE3452087DC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 591872, 'r0_': 2956032}}
)
@triton.jit
def triton_red_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 1152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_out_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tl.full([1, 1], 256, tl.int32)
        tmp5 = tmp3 + tmp4
        tmp6 = tmp3 < 0
        tmp7 = tl.where(tmp6, tmp5, tmp3)
        tl.device_assert(((0 <= tmp7) & (tmp7 < 256)) | ~(xmask), "index out of bounds: 0 <= tmp7 < 256")
        tmp9 = tl.load(in_ptr3 + (r0_1 + 1152*tmp7), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tmp2 + tmp9
        tmp12 = tmp10 + tmp11
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_reduce(
            tmp14, tmp15_mean, tmp15_m2, tmp15_weight, roffset == 0
        )
        tmp15_mean = tl.where(r0_mask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(r0_mask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(r0_mask & xmask, tmp15_weight_next, tmp15_weight)
        tl.store(in_out_ptr0 + (r0_1 + 1152*x0), tmp12, r0_mask & xmask)
    tmp16, tmp17, tmp18 = triton_helpers.welford(tmp15_mean, tmp15_m2, tmp15_weight, 1)
    tmp15 = tmp16[:, None]
    tmp19 = tmp17[:, None]
    tmp20 = tmp18[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22 - tmp15
        tmp24 = tl.full([1, 1], 1152.0, tl.float32)
        tmp25 = (tmp19 / tmp24)
        tmp26 = tl.full([1, 1], 1e-06, tl.float32)
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.rsqrt(tmp27)
        tmp29 = tmp23 * tmp28
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 * tmp31
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 + tmp34
        tmp36 = tmp35.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 1152*x0), tmp36, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /scratch/xinyaowang/vla-vllm-profiling/inductor/ui/cui3olio2fzniuwv7aiz6ged76dnsfwx3rwctsmokghqsd7aeubl.py
# Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.view, aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_3 => view_14
#   hidden_states_4 => add_6, add_7, convert_element_type_20, convert_element_type_21, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, tanh
# Graph fragment:
#   %addmm_4 : Tensor "bf16[256, 4304][4304, 1]cuda:0" = PlaceHolder[target=addmm_4]
#   %view_14 : Tensor "bf16[1, 256, 4304][1101824, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_4, [1, 256, 4304]), kwargs = {})
#   %convert_element_type_20 : Tensor "f32[1, 256, 4304][1101824, 4304, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_14, torch.float32), kwargs = {})
#   %mul_8 : Tensor "f32[1, 256, 4304][1101824, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, 0.5), kwargs = {})
#   %mul_4 : Tensor "f32[1, 256, 4304][1101824, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, %convert_element_type_20), kwargs = {})
#   %mul_5 : Tensor "f32[1, 256, 4304][1101824, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %convert_element_type_20), kwargs = {})
#   %mul_6 : Tensor "f32[1, 256, 4304][1101824, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, 0.044715), kwargs = {})
#   %add_6 : Tensor "f32[1, 256, 4304][1101824, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_20, %mul_6), kwargs = {})
#   %mul_7 : Tensor "f32[1, 256, 4304][1101824, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 0.7978845608028654), kwargs = {})
#   %tanh : Tensor "f32[1, 256, 4304][1101824, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_7,), kwargs = {})
#   %add_7 : Tensor "f32[1, 256, 4304][1101824, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1), kwargs = {})
#   %mul_9 : Tensor "f32[1, 256, 4304][1101824, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %add_7), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[1, 256, 4304][1101824, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9, torch.bfloat16), kwargs = {})
#   return %convert_element_type_21
triton_poi_fused_gelu_view_5 = async_compile.triton('triton_poi_fused_gelu_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '9C3337B899FEA3CB29E8ADD9D131EBEE0EC33016577309EAF4B44AE3452087DC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 6610944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_view_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1101824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 4304)
    x1 = xindex // 4304
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
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
    tl.store(out_ptr0 + (x0 + 4352*x1), tmp15, None)
''', device_str='cuda')


# kernel path: /scratch/xinyaowang/vla-vllm-profiling/inductor/hg/chgjqf2rnu6xhy7boucknsp4cd6vgattwiurvtrehyjrt6mox6c4.py
# Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6, hidden_states_7], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_5 => view_16
#   hidden_states_6 => add_8
#   hidden_states_7 => add_10, add_9, clone_2, convert_element_type_25, convert_element_type_26, mul_10, mul_11, rsqrt_2, sub_2, var_mean_2
# Graph fragment:
#   %add_3 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0" = PlaceHolder[target=add_3]
#   %addmm_5 : Tensor "bf16[256, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %getitem_14 : Tensor "f32[1, 256, 1][256, 1, 256]cuda:0" = PlaceHolder[target=getitem_14]
#   %buf29 : Tensor "f32[1, 256, 1][256, 1, 256]cuda:0" = PlaceHolder[target=buf29]
#   %arg21_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg21_1]
#   %arg22_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %view_16 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_5, [1, 256, 1152]), kwargs = {})
#   %add_8 : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_16), kwargs = {})
#   %clone_2 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_8,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_25 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_2, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_25, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_25, %getitem_14), kwargs = {})
#   %add_9 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_13, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_10 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_11 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %arg21_1), kwargs = {})
#   %add_10 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %arg22_1), kwargs = {})
#   %convert_element_type_26 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.bfloat16), kwargs = {})
#   return %getitem_14,%buf29,%convert_element_type_26
triton_red_fused_add_native_layer_norm_view_6 = async_compile.triton('triton_red_fused_add_native_layer_norm_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '9C3337B899FEA3CB29E8ADD9D131EBEE0EC33016577309EAF4B44AE3452087DC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 2363904}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 1152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(r0_mask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(r0_mask & xmask, tmp5_weight_next, tmp5_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp11 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14 - tmp5
        tmp16 = tl.full([1, 1], 1152.0, tl.float32)
        tmp17 = (tmp9 / tmp16)
        tmp18 = tl.full([1, 1], 1e-06, tl.float32)
        tmp19 = tmp17 + tmp18
        tmp20 = libdevice.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 + tmp26
        tmp28 = tmp27.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 1152*x0), tmp28, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /scratch/xinyaowang/vla-vllm-profiling/inductor/yk/cykwzbzdrxk2ijifiug7baymgtlqn7s6nf57c6b6nahks7kfqg4h.py
# Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_9], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_7 => view_28
#   hidden_states_5 => view_16
#   hidden_states_6 => add_8
#   hidden_states_8 => add_11
#   hidden_states_9 => add_12, add_13, clone_3, convert_element_type_39, convert_element_type_40, mul_12, mul_13, rsqrt_3, sub_3, var_mean_3
# Graph fragment:
#   %add_3 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0" = PlaceHolder[target=add_3]
#   %addmm_5 : Tensor "bf16[256, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %addmm_9 : Tensor "bf16[256, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_9]
#   %getitem_25 : Tensor "f32[1, 256, 1][256, 1, 256]cuda:0" = PlaceHolder[target=getitem_25]
#   %buf43 : Tensor "f32[1, 256, 1][256, 1, 256]cuda:0" = PlaceHolder[target=buf43]
#   %arg31_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg31_1]
#   %arg32_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg32_1]
#   %view_16 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_5, [1, 256, 1152]), kwargs = {})
#   %add_8 : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_16), kwargs = {})
#   %view_28 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_9, [1, 256, 1152]), kwargs = {})
#   %add_11 : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_28), kwargs = {})
#   %clone_3 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_11,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_39 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_3, torch.float32), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_39, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_39, %getitem_25), kwargs = {})
#   %add_12 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-06), kwargs = {})
#   %rsqrt_3 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_12 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_13 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %arg31_1), kwargs = {})
#   %add_13 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %arg32_1), kwargs = {})
#   %convert_element_type_40 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_13, torch.bfloat16), kwargs = {})
#   return %getitem_25,%buf43,%convert_element_type_40
triton_red_fused_add_native_layer_norm_view_7 = async_compile.triton('triton_red_fused_add_native_layer_norm_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '9C3337B899FEA3CB29E8ADD9D131EBEE0EC33016577309EAF4B44AE3452087DC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 2953728}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_view_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 1152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp7_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp7_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp7_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp7_mean_next, tmp7_m2_next, tmp7_weight_next = triton_helpers.welford_reduce(
            tmp6, tmp7_mean, tmp7_m2, tmp7_weight, roffset == 0
        )
        tmp7_mean = tl.where(r0_mask & xmask, tmp7_mean_next, tmp7_mean)
        tmp7_m2 = tl.where(r0_mask & xmask, tmp7_m2_next, tmp7_m2)
        tmp7_weight = tl.where(r0_mask & xmask, tmp7_weight_next, tmp7_weight)
    tmp8, tmp9, tmp10 = triton_helpers.welford(tmp7_mean, tmp7_m2, tmp7_weight, 1)
    tmp7 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    tmp12 = tmp10[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr2 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp29 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 - tmp7
        tmp20 = tl.full([1, 1], 1152.0, tl.float32)
        tmp21 = (tmp11 / tmp20)
        tmp22 = tl.full([1, 1], 1e-06, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 * tmp27
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp28 + tmp30
        tmp32 = tmp31.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 1152*x0), tmp32, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /scratch/xinyaowang/vla-vllm-profiling/inductor/rv/crv7tmduxs6d5iwruwsacushpgrhwglkuv4u22cokbberlgjw546.py
# Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_12, hidden_states_13, hidden_states_14], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_7 => view_28
#   hidden_states_12 => view_32
#   hidden_states_13 => add_16
#   hidden_states_14 => add_17, add_18, clone_4, convert_element_type_49, convert_element_type_50, mul_20, mul_21, rsqrt_4, sub_4, var_mean_4
#   hidden_states_5 => view_16
#   hidden_states_6 => add_8
#   hidden_states_8 => add_11
# Graph fragment:
#   %add_3 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0" = PlaceHolder[target=add_3]
#   %addmm_5 : Tensor "bf16[256, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %addmm_9 : Tensor "bf16[256, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_9]
#   %addmm_11 : Tensor "bf16[256, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_11]
#   %getitem_27 : Tensor "f32[1, 256, 1][256, 1, 256]cuda:0" = PlaceHolder[target=getitem_27]
#   %buf50 : Tensor "f32[1, 256, 1][256, 1, 256]cuda:0" = PlaceHolder[target=buf50]
#   %arg37_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg37_1]
#   %arg38_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg38_1]
#   %view_16 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_5, [1, 256, 1152]), kwargs = {})
#   %add_8 : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_16), kwargs = {})
#   %view_28 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_9, [1, 256, 1152]), kwargs = {})
#   %add_11 : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_28), kwargs = {})
#   %view_32 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_11, [1, 256, 1152]), kwargs = {})
#   %add_16 : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_32), kwargs = {})
#   %clone_4 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_16,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_49 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_4, torch.float32), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_49, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_49, %getitem_27), kwargs = {})
#   %add_17 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %mul_20 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_21 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %arg37_1), kwargs = {})
#   %add_18 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %arg38_1), kwargs = {})
#   %convert_element_type_50 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_18, torch.bfloat16), kwargs = {})
#   return %getitem_27,%buf50,%convert_element_type_50
triton_red_fused_add_native_layer_norm_view_8 = async_compile.triton('triton_red_fused_add_native_layer_norm_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '9C3337B899FEA3CB29E8ADD9D131EBEE0EC33016577309EAF4B44AE3452087DC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 3543552}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_view_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 1152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp9_mean, tmp9_m2, tmp9_weight, roffset == 0
        )
        tmp9_mean = tl.where(r0_mask & xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(r0_mask & xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(r0_mask & xmask, tmp9_weight_next, tmp9_weight)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp9_mean, tmp9_m2, tmp9_weight, 1)
    tmp9 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr3 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 + tmp18
        tmp21 = tmp19 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22 - tmp9
        tmp24 = tl.full([1, 1], 1152.0, tl.float32)
        tmp25 = (tmp13 / tmp24)
        tmp26 = tl.full([1, 1], 1e-06, tl.float32)
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.rsqrt(tmp27)
        tmp29 = tmp23 * tmp28
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 * tmp31
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 + tmp34
        tmp36 = tmp35.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 1152*x0), tmp36, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /scratch/xinyaowang/vla-vllm-profiling/inductor/x2/cx2wuitswytufeoeatjy5jshvtliufdauuragznmh6j6zpe4etyh.py
# Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_12, hidden_states_13, attn_output_11, hidden_states_15, hidden_states_16], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_11 => view_44
#   attn_output_7 => view_28
#   hidden_states_12 => view_32
#   hidden_states_13 => add_16
#   hidden_states_15 => add_19
#   hidden_states_16 => add_20, add_21, clone_5, convert_element_type_63, convert_element_type_64, mul_22, mul_23, rsqrt_5, sub_5, var_mean_5
#   hidden_states_5 => view_16
#   hidden_states_6 => add_8
#   hidden_states_8 => add_11
# Graph fragment:
#   %add_3 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0" = PlaceHolder[target=add_3]
#   %addmm_5 : Tensor "bf16[256, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %addmm_9 : Tensor "bf16[256, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_9]
#   %addmm_11 : Tensor "bf16[256, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_11]
#   %addmm_15 : Tensor "bf16[256, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_15]
#   %add_19 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0" = PlaceHolder[target=add_19]
#   %getitem_38 : Tensor "f32[1, 256, 1][256, 1, 256]cuda:0" = PlaceHolder[target=getitem_38]
#   %buf65 : Tensor "f32[1, 256, 1][256, 1, 256]cuda:0" = PlaceHolder[target=buf65]
#   %arg47_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg47_1]
#   %arg48_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg48_1]
#   %view_16 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_5, [1, 256, 1152]), kwargs = {})
#   %add_8 : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_16), kwargs = {})
#   %view_28 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_9, [1, 256, 1152]), kwargs = {})
#   %add_11 : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_28), kwargs = {})
#   %view_32 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_11, [1, 256, 1152]), kwargs = {})
#   %add_16 : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_32), kwargs = {})
#   %view_44 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_15, [1, 256, 1152]), kwargs = {})
#   %add_19 : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %view_44), kwargs = {})
#   %clone_5 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_19,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_63 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_5, torch.float32), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_63, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_63, %getitem_38), kwargs = {})
#   %add_20 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_37, 1e-06), kwargs = {})
#   %rsqrt_5 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_20,), kwargs = {})
#   %mul_22 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_5), kwargs = {})
#   %mul_23 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %arg47_1), kwargs = {})
#   %add_21 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %arg48_1), kwargs = {})
#   %convert_element_type_64 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_21, torch.bfloat16), kwargs = {})
#   return %add_19,%getitem_38,%buf65,%convert_element_type_64
triton_red_fused_add_native_layer_norm_view_9 = async_compile.triton('triton_red_fused_add_native_layer_norm_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_view_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '9C3337B899FEA3CB29E8ADD9D131EBEE0EC33016577309EAF4B44AE3452087DC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 5313024}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_view_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 1152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp11_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight, roffset == 0
        )
        tmp11_mean = tl.where(r0_mask & xmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(r0_mask & xmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(r0_mask & xmask, tmp11_weight_next, tmp11_weight)
        tl.store(in_out_ptr0 + (r0_1 + 1152*x0), tmp8, r0_mask & xmask)
    tmp12, tmp13, tmp14 = triton_helpers.welford(tmp11_mean, tmp11_m2, tmp11_weight, 1)
    tmp11 = tmp12[:, None]
    tmp15 = tmp13[:, None]
    tmp16 = tmp14[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp17 = tl.load(in_out_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp29 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 - tmp11
        tmp20 = tl.full([1, 1], 1152.0, tl.float32)
        tmp21 = (tmp15 / tmp20)
        tmp22 = tl.full([1, 1], 1e-06, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 * tmp27
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp28 + tmp30
        tmp32 = tmp31.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 1152*x0), tmp32, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /scratch/xinyaowang/vla-vllm-profiling/inductor/zp/czpntwwatbbqbrauxkiemjznf7x7bjzlgmtgsejm2ud6t5w74p4v.py
# Topologically Sorted Source Nodes: [hidden_states_187, hidden_states_188, last_hidden_state], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_187 => view_432
#   hidden_states_188 => add_216
#   last_hidden_state => add_217, add_218, clone_54, convert_element_type_649, convert_element_type_650, mul_270, mul_271, rsqrt_54, sub_54, var_mean_54
# Graph fragment:
#   %add_211 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0" = PlaceHolder[target=add_211]
#   %addmm_161 : Tensor "bf16[256, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_161]
#   %getitem_352 : Tensor "f32[1, 256, 1][256, 1, 256]cuda:0" = PlaceHolder[target=getitem_352]
#   %buf588 : Tensor "f32[1, 256, 1][256, 1, 256]cuda:0" = PlaceHolder[target=buf588]
#   %arg437_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg437_1]
#   %arg438_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg438_1]
#   %view_432 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_161, [1, 256, 1152]), kwargs = {})
#   %add_216 : Tensor "bf16[1, 256, 1152][294912, 1, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_211, %view_432), kwargs = {})
#   %clone_54 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_216,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_649 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_54, torch.float32), kwargs = {})
#   %var_mean_54 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_649, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_54 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_649, %getitem_352), kwargs = {})
#   %add_217 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_351, 1e-06), kwargs = {})
#   %rsqrt_54 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_217,), kwargs = {})
#   %mul_270 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %rsqrt_54), kwargs = {})
#   %mul_271 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_270, %arg437_1), kwargs = {})
#   %add_218 : Tensor "f32[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_271, %arg438_1), kwargs = {})
#   %convert_element_type_650 : Tensor "bf16[1, 256, 1152][294912, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_218, torch.bfloat16), kwargs = {})
#   return %getitem_352,%buf588,%convert_element_type_650
triton_red_fused_add_native_layer_norm_view_10 = async_compile.triton('triton_red_fused_add_native_layer_norm_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_view_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '9C3337B899FEA3CB29E8ADD9D131EBEE0EC33016577309EAF4B44AE3452087DC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 2363904}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_view_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 1152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(r0_mask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(r0_mask & xmask, tmp5_weight_next, tmp5_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp11 = tl.load(in_out_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14 - tmp5
        tmp16 = tl.full([1, 1], 1152.0, tl.float32)
        tmp17 = (tmp9 / tmp16)
        tmp18 = tl.full([1, 1], 1e-06, tl.float32)
        tmp19 = tmp17 + tmp18
        tmp20 = libdevice.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 + tmp26
        tmp28 = tmp27.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 1152*x0), tmp28, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /scratch/xinyaowang/vla-vllm-profiling/inductor/w7/cw7pv62pvyiw2z66czacjjbwkxxgomak3v3nyflqkdk3lvluala5.py
# Topologically Sorted Source Nodes: [hidden_states_189, features], Original ATen: [aten.view, aten.mul]
# Source node to ATen node mapping:
#   features => mul_272
#   hidden_states_189 => view_434
# Graph fragment:
#   %addmm_162 : Tensor "bf16[256, 2048][2048, 1]cuda:0" = PlaceHolder[target=addmm_162]
#   %view_434 : Tensor "bf16[1, 256, 2048][524288, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_162, [1, 256, 2048]), kwargs = {})
#   %mul_272 : Tensor "bf16[1, 256, 2048][524288, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_434, 45.254833995939045), kwargs = {})
#   return %mul_272
triton_poi_fused_mul_view_11 = async_compile.triton('triton_poi_fused_mul_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '9C3337B899FEA3CB29E8ADD9D131EBEE0EC33016577309EAF4B44AE3452087DC', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3145728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_view_11(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.full([1], 45.254833995939045, tl.float32)
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 3, 224, 224), (3, 1, 672, 3))
        assert_size_stride(arg1_1, (1152, 3, 14, 14), (588, 196, 14, 1))
        assert_size_stride(arg2_1, (1152, ), (1, ))
        assert_size_stride(arg3_1, (1, 256), (256, 1))
        assert_size_stride(arg4_1, (256, 1152), (1152, 1))
        assert_size_stride(arg5_1, (1152, ), (1, ))
        assert_size_stride(arg6_1, (1152, ), (1, ))
        assert_size_stride(arg7_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg8_1, (1152, ), (1, ))
        assert_size_stride(arg9_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg10_1, (1152, ), (1, ))
        assert_size_stride(arg11_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg12_1, (1152, ), (1, ))
        assert_size_stride(arg13_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg14_1, (1152, ), (1, ))
        assert_size_stride(arg15_1, (1152, ), (1, ))
        assert_size_stride(arg16_1, (1152, ), (1, ))
        assert_size_stride(arg17_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg18_1, (4304, ), (1, ))
        assert_size_stride(arg19_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg20_1, (1152, ), (1, ))
        assert_size_stride(arg21_1, (1152, ), (1, ))
        assert_size_stride(arg22_1, (1152, ), (1, ))
        assert_size_stride(arg23_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg24_1, (1152, ), (1, ))
        assert_size_stride(arg25_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg26_1, (1152, ), (1, ))
        assert_size_stride(arg27_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg28_1, (1152, ), (1, ))
        assert_size_stride(arg29_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg30_1, (1152, ), (1, ))
        assert_size_stride(arg31_1, (1152, ), (1, ))
        assert_size_stride(arg32_1, (1152, ), (1, ))
        assert_size_stride(arg33_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg34_1, (4304, ), (1, ))
        assert_size_stride(arg35_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg36_1, (1152, ), (1, ))
        assert_size_stride(arg37_1, (1152, ), (1, ))
        assert_size_stride(arg38_1, (1152, ), (1, ))
        assert_size_stride(arg39_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg40_1, (1152, ), (1, ))
        assert_size_stride(arg41_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg42_1, (1152, ), (1, ))
        assert_size_stride(arg43_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg44_1, (1152, ), (1, ))
        assert_size_stride(arg45_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg46_1, (1152, ), (1, ))
        assert_size_stride(arg47_1, (1152, ), (1, ))
        assert_size_stride(arg48_1, (1152, ), (1, ))
        assert_size_stride(arg49_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg50_1, (4304, ), (1, ))
        assert_size_stride(arg51_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg52_1, (1152, ), (1, ))
        assert_size_stride(arg53_1, (1152, ), (1, ))
        assert_size_stride(arg54_1, (1152, ), (1, ))
        assert_size_stride(arg55_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg56_1, (1152, ), (1, ))
        assert_size_stride(arg57_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg58_1, (1152, ), (1, ))
        assert_size_stride(arg59_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg60_1, (1152, ), (1, ))
        assert_size_stride(arg61_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg62_1, (1152, ), (1, ))
        assert_size_stride(arg63_1, (1152, ), (1, ))
        assert_size_stride(arg64_1, (1152, ), (1, ))
        assert_size_stride(arg65_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg66_1, (4304, ), (1, ))
        assert_size_stride(arg67_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg68_1, (1152, ), (1, ))
        assert_size_stride(arg69_1, (1152, ), (1, ))
        assert_size_stride(arg70_1, (1152, ), (1, ))
        assert_size_stride(arg71_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg72_1, (1152, ), (1, ))
        assert_size_stride(arg73_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg74_1, (1152, ), (1, ))
        assert_size_stride(arg75_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg76_1, (1152, ), (1, ))
        assert_size_stride(arg77_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg78_1, (1152, ), (1, ))
        assert_size_stride(arg79_1, (1152, ), (1, ))
        assert_size_stride(arg80_1, (1152, ), (1, ))
        assert_size_stride(arg81_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg82_1, (4304, ), (1, ))
        assert_size_stride(arg83_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg84_1, (1152, ), (1, ))
        assert_size_stride(arg85_1, (1152, ), (1, ))
        assert_size_stride(arg86_1, (1152, ), (1, ))
        assert_size_stride(arg87_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg88_1, (1152, ), (1, ))
        assert_size_stride(arg89_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg90_1, (1152, ), (1, ))
        assert_size_stride(arg91_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg92_1, (1152, ), (1, ))
        assert_size_stride(arg93_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg94_1, (1152, ), (1, ))
        assert_size_stride(arg95_1, (1152, ), (1, ))
        assert_size_stride(arg96_1, (1152, ), (1, ))
        assert_size_stride(arg97_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg98_1, (4304, ), (1, ))
        assert_size_stride(arg99_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg100_1, (1152, ), (1, ))
        assert_size_stride(arg101_1, (1152, ), (1, ))
        assert_size_stride(arg102_1, (1152, ), (1, ))
        assert_size_stride(arg103_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg104_1, (1152, ), (1, ))
        assert_size_stride(arg105_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg106_1, (1152, ), (1, ))
        assert_size_stride(arg107_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg108_1, (1152, ), (1, ))
        assert_size_stride(arg109_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg110_1, (1152, ), (1, ))
        assert_size_stride(arg111_1, (1152, ), (1, ))
        assert_size_stride(arg112_1, (1152, ), (1, ))
        assert_size_stride(arg113_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg114_1, (4304, ), (1, ))
        assert_size_stride(arg115_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg116_1, (1152, ), (1, ))
        assert_size_stride(arg117_1, (1152, ), (1, ))
        assert_size_stride(arg118_1, (1152, ), (1, ))
        assert_size_stride(arg119_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg120_1, (1152, ), (1, ))
        assert_size_stride(arg121_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg122_1, (1152, ), (1, ))
        assert_size_stride(arg123_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg124_1, (1152, ), (1, ))
        assert_size_stride(arg125_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg126_1, (1152, ), (1, ))
        assert_size_stride(arg127_1, (1152, ), (1, ))
        assert_size_stride(arg128_1, (1152, ), (1, ))
        assert_size_stride(arg129_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg130_1, (4304, ), (1, ))
        assert_size_stride(arg131_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg132_1, (1152, ), (1, ))
        assert_size_stride(arg133_1, (1152, ), (1, ))
        assert_size_stride(arg134_1, (1152, ), (1, ))
        assert_size_stride(arg135_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg136_1, (1152, ), (1, ))
        assert_size_stride(arg137_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg138_1, (1152, ), (1, ))
        assert_size_stride(arg139_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg140_1, (1152, ), (1, ))
        assert_size_stride(arg141_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg142_1, (1152, ), (1, ))
        assert_size_stride(arg143_1, (1152, ), (1, ))
        assert_size_stride(arg144_1, (1152, ), (1, ))
        assert_size_stride(arg145_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg146_1, (4304, ), (1, ))
        assert_size_stride(arg147_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg148_1, (1152, ), (1, ))
        assert_size_stride(arg149_1, (1152, ), (1, ))
        assert_size_stride(arg150_1, (1152, ), (1, ))
        assert_size_stride(arg151_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg152_1, (1152, ), (1, ))
        assert_size_stride(arg153_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg154_1, (1152, ), (1, ))
        assert_size_stride(arg155_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg156_1, (1152, ), (1, ))
        assert_size_stride(arg157_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg158_1, (1152, ), (1, ))
        assert_size_stride(arg159_1, (1152, ), (1, ))
        assert_size_stride(arg160_1, (1152, ), (1, ))
        assert_size_stride(arg161_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg162_1, (4304, ), (1, ))
        assert_size_stride(arg163_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg164_1, (1152, ), (1, ))
        assert_size_stride(arg165_1, (1152, ), (1, ))
        assert_size_stride(arg166_1, (1152, ), (1, ))
        assert_size_stride(arg167_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg168_1, (1152, ), (1, ))
        assert_size_stride(arg169_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg170_1, (1152, ), (1, ))
        assert_size_stride(arg171_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg172_1, (1152, ), (1, ))
        assert_size_stride(arg173_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg174_1, (1152, ), (1, ))
        assert_size_stride(arg175_1, (1152, ), (1, ))
        assert_size_stride(arg176_1, (1152, ), (1, ))
        assert_size_stride(arg177_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg178_1, (4304, ), (1, ))
        assert_size_stride(arg179_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg180_1, (1152, ), (1, ))
        assert_size_stride(arg181_1, (1152, ), (1, ))
        assert_size_stride(arg182_1, (1152, ), (1, ))
        assert_size_stride(arg183_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg184_1, (1152, ), (1, ))
        assert_size_stride(arg185_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg186_1, (1152, ), (1, ))
        assert_size_stride(arg187_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg188_1, (1152, ), (1, ))
        assert_size_stride(arg189_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg190_1, (1152, ), (1, ))
        assert_size_stride(arg191_1, (1152, ), (1, ))
        assert_size_stride(arg192_1, (1152, ), (1, ))
        assert_size_stride(arg193_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg194_1, (4304, ), (1, ))
        assert_size_stride(arg195_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg196_1, (1152, ), (1, ))
        assert_size_stride(arg197_1, (1152, ), (1, ))
        assert_size_stride(arg198_1, (1152, ), (1, ))
        assert_size_stride(arg199_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg200_1, (1152, ), (1, ))
        assert_size_stride(arg201_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg202_1, (1152, ), (1, ))
        assert_size_stride(arg203_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg204_1, (1152, ), (1, ))
        assert_size_stride(arg205_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg206_1, (1152, ), (1, ))
        assert_size_stride(arg207_1, (1152, ), (1, ))
        assert_size_stride(arg208_1, (1152, ), (1, ))
        assert_size_stride(arg209_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg210_1, (4304, ), (1, ))
        assert_size_stride(arg211_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg212_1, (1152, ), (1, ))
        assert_size_stride(arg213_1, (1152, ), (1, ))
        assert_size_stride(arg214_1, (1152, ), (1, ))
        assert_size_stride(arg215_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg216_1, (1152, ), (1, ))
        assert_size_stride(arg217_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg218_1, (1152, ), (1, ))
        assert_size_stride(arg219_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg220_1, (1152, ), (1, ))
        assert_size_stride(arg221_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg222_1, (1152, ), (1, ))
        assert_size_stride(arg223_1, (1152, ), (1, ))
        assert_size_stride(arg224_1, (1152, ), (1, ))
        assert_size_stride(arg225_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg226_1, (4304, ), (1, ))
        assert_size_stride(arg227_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg228_1, (1152, ), (1, ))
        assert_size_stride(arg229_1, (1152, ), (1, ))
        assert_size_stride(arg230_1, (1152, ), (1, ))
        assert_size_stride(arg231_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg232_1, (1152, ), (1, ))
        assert_size_stride(arg233_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg234_1, (1152, ), (1, ))
        assert_size_stride(arg235_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg236_1, (1152, ), (1, ))
        assert_size_stride(arg237_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg238_1, (1152, ), (1, ))
        assert_size_stride(arg239_1, (1152, ), (1, ))
        assert_size_stride(arg240_1, (1152, ), (1, ))
        assert_size_stride(arg241_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg242_1, (4304, ), (1, ))
        assert_size_stride(arg243_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg244_1, (1152, ), (1, ))
        assert_size_stride(arg245_1, (1152, ), (1, ))
        assert_size_stride(arg246_1, (1152, ), (1, ))
        assert_size_stride(arg247_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg248_1, (1152, ), (1, ))
        assert_size_stride(arg249_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg250_1, (1152, ), (1, ))
        assert_size_stride(arg251_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg252_1, (1152, ), (1, ))
        assert_size_stride(arg253_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg254_1, (1152, ), (1, ))
        assert_size_stride(arg255_1, (1152, ), (1, ))
        assert_size_stride(arg256_1, (1152, ), (1, ))
        assert_size_stride(arg257_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg258_1, (4304, ), (1, ))
        assert_size_stride(arg259_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg260_1, (1152, ), (1, ))
        assert_size_stride(arg261_1, (1152, ), (1, ))
        assert_size_stride(arg262_1, (1152, ), (1, ))
        assert_size_stride(arg263_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg264_1, (1152, ), (1, ))
        assert_size_stride(arg265_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg266_1, (1152, ), (1, ))
        assert_size_stride(arg267_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg268_1, (1152, ), (1, ))
        assert_size_stride(arg269_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg270_1, (1152, ), (1, ))
        assert_size_stride(arg271_1, (1152, ), (1, ))
        assert_size_stride(arg272_1, (1152, ), (1, ))
        assert_size_stride(arg273_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg274_1, (4304, ), (1, ))
        assert_size_stride(arg275_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg276_1, (1152, ), (1, ))
        assert_size_stride(arg277_1, (1152, ), (1, ))
        assert_size_stride(arg278_1, (1152, ), (1, ))
        assert_size_stride(arg279_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg280_1, (1152, ), (1, ))
        assert_size_stride(arg281_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg282_1, (1152, ), (1, ))
        assert_size_stride(arg283_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg284_1, (1152, ), (1, ))
        assert_size_stride(arg285_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg286_1, (1152, ), (1, ))
        assert_size_stride(arg287_1, (1152, ), (1, ))
        assert_size_stride(arg288_1, (1152, ), (1, ))
        assert_size_stride(arg289_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg290_1, (4304, ), (1, ))
        assert_size_stride(arg291_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg292_1, (1152, ), (1, ))
        assert_size_stride(arg293_1, (1152, ), (1, ))
        assert_size_stride(arg294_1, (1152, ), (1, ))
        assert_size_stride(arg295_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg296_1, (1152, ), (1, ))
        assert_size_stride(arg297_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg298_1, (1152, ), (1, ))
        assert_size_stride(arg299_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg300_1, (1152, ), (1, ))
        assert_size_stride(arg301_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg302_1, (1152, ), (1, ))
        assert_size_stride(arg303_1, (1152, ), (1, ))
        assert_size_stride(arg304_1, (1152, ), (1, ))
        assert_size_stride(arg305_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg306_1, (4304, ), (1, ))
        assert_size_stride(arg307_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg308_1, (1152, ), (1, ))
        assert_size_stride(arg309_1, (1152, ), (1, ))
        assert_size_stride(arg310_1, (1152, ), (1, ))
        assert_size_stride(arg311_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg312_1, (1152, ), (1, ))
        assert_size_stride(arg313_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg314_1, (1152, ), (1, ))
        assert_size_stride(arg315_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg316_1, (1152, ), (1, ))
        assert_size_stride(arg317_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg318_1, (1152, ), (1, ))
        assert_size_stride(arg319_1, (1152, ), (1, ))
        assert_size_stride(arg320_1, (1152, ), (1, ))
        assert_size_stride(arg321_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg322_1, (4304, ), (1, ))
        assert_size_stride(arg323_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg324_1, (1152, ), (1, ))
        assert_size_stride(arg325_1, (1152, ), (1, ))
        assert_size_stride(arg326_1, (1152, ), (1, ))
        assert_size_stride(arg327_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg328_1, (1152, ), (1, ))
        assert_size_stride(arg329_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg330_1, (1152, ), (1, ))
        assert_size_stride(arg331_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg332_1, (1152, ), (1, ))
        assert_size_stride(arg333_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg334_1, (1152, ), (1, ))
        assert_size_stride(arg335_1, (1152, ), (1, ))
        assert_size_stride(arg336_1, (1152, ), (1, ))
        assert_size_stride(arg337_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg338_1, (4304, ), (1, ))
        assert_size_stride(arg339_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg340_1, (1152, ), (1, ))
        assert_size_stride(arg341_1, (1152, ), (1, ))
        assert_size_stride(arg342_1, (1152, ), (1, ))
        assert_size_stride(arg343_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg344_1, (1152, ), (1, ))
        assert_size_stride(arg345_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg346_1, (1152, ), (1, ))
        assert_size_stride(arg347_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg348_1, (1152, ), (1, ))
        assert_size_stride(arg349_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg350_1, (1152, ), (1, ))
        assert_size_stride(arg351_1, (1152, ), (1, ))
        assert_size_stride(arg352_1, (1152, ), (1, ))
        assert_size_stride(arg353_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg354_1, (4304, ), (1, ))
        assert_size_stride(arg355_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg356_1, (1152, ), (1, ))
        assert_size_stride(arg357_1, (1152, ), (1, ))
        assert_size_stride(arg358_1, (1152, ), (1, ))
        assert_size_stride(arg359_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg360_1, (1152, ), (1, ))
        assert_size_stride(arg361_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg362_1, (1152, ), (1, ))
        assert_size_stride(arg363_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg364_1, (1152, ), (1, ))
        assert_size_stride(arg365_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg366_1, (1152, ), (1, ))
        assert_size_stride(arg367_1, (1152, ), (1, ))
        assert_size_stride(arg368_1, (1152, ), (1, ))
        assert_size_stride(arg369_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg370_1, (4304, ), (1, ))
        assert_size_stride(arg371_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg372_1, (1152, ), (1, ))
        assert_size_stride(arg373_1, (1152, ), (1, ))
        assert_size_stride(arg374_1, (1152, ), (1, ))
        assert_size_stride(arg375_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg376_1, (1152, ), (1, ))
        assert_size_stride(arg377_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg378_1, (1152, ), (1, ))
        assert_size_stride(arg379_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg380_1, (1152, ), (1, ))
        assert_size_stride(arg381_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg382_1, (1152, ), (1, ))
        assert_size_stride(arg383_1, (1152, ), (1, ))
        assert_size_stride(arg384_1, (1152, ), (1, ))
        assert_size_stride(arg385_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg386_1, (4304, ), (1, ))
        assert_size_stride(arg387_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg388_1, (1152, ), (1, ))
        assert_size_stride(arg389_1, (1152, ), (1, ))
        assert_size_stride(arg390_1, (1152, ), (1, ))
        assert_size_stride(arg391_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg392_1, (1152, ), (1, ))
        assert_size_stride(arg393_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg394_1, (1152, ), (1, ))
        assert_size_stride(arg395_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg396_1, (1152, ), (1, ))
        assert_size_stride(arg397_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg398_1, (1152, ), (1, ))
        assert_size_stride(arg399_1, (1152, ), (1, ))
        assert_size_stride(arg400_1, (1152, ), (1, ))
        assert_size_stride(arg401_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg402_1, (4304, ), (1, ))
        assert_size_stride(arg403_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg404_1, (1152, ), (1, ))
        assert_size_stride(arg405_1, (1152, ), (1, ))
        assert_size_stride(arg406_1, (1152, ), (1, ))
        assert_size_stride(arg407_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg408_1, (1152, ), (1, ))
        assert_size_stride(arg409_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg410_1, (1152, ), (1, ))
        assert_size_stride(arg411_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg412_1, (1152, ), (1, ))
        assert_size_stride(arg413_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg414_1, (1152, ), (1, ))
        assert_size_stride(arg415_1, (1152, ), (1, ))
        assert_size_stride(arg416_1, (1152, ), (1, ))
        assert_size_stride(arg417_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg418_1, (4304, ), (1, ))
        assert_size_stride(arg419_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg420_1, (1152, ), (1, ))
        assert_size_stride(arg421_1, (1152, ), (1, ))
        assert_size_stride(arg422_1, (1152, ), (1, ))
        assert_size_stride(arg423_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg424_1, (1152, ), (1, ))
        assert_size_stride(arg425_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg426_1, (1152, ), (1, ))
        assert_size_stride(arg427_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg428_1, (1152, ), (1, ))
        assert_size_stride(arg429_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg430_1, (1152, ), (1, ))
        assert_size_stride(arg431_1, (1152, ), (1, ))
        assert_size_stride(arg432_1, (1152, ), (1, ))
        assert_size_stride(arg433_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg434_1, (4304, ), (1, ))
        assert_size_stride(arg435_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg436_1, (1152, ), (1, ))
        assert_size_stride(arg437_1, (1152, ), (1, ))
        assert_size_stride(arg438_1, (1152, ), (1, ))
        assert_size_stride(arg439_1, (2048, 1152), (1152, 1))
        assert_size_stride(arg440_1, (2048, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((1, 3, 224, 224), (150528, 50176, 224, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [pixel_values, patch_embeds], Original ATen: [aten._to_copy, aten.convolution]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_0.run(arg0_1, buf1, 3, 50176, stream=stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [pixel_values, patch_embeds], Original ATen: [aten._to_copy, aten.convolution]
            buf2 = extern_kernels.convolution(buf1, arg1_1, stride=(14, 14), padding=(0,), dilation=(1, 1), transposed=False, output_padding=(0,), groups=1, bias=None)
            assert_size_stride(buf2, (1, 1152, 16, 16), (294912, 256, 16, 1), 'torch.ops.aten.convolution.default')
            del arg1_1
            del buf1
            buf3 = empty_strided_cuda((1, 256, 1, 9), (2304, 1, 2304, 256), torch.float32)
            buf4 = empty_strided_cuda((1, 256, 1, 9), (2304, 1, 2304, 256), torch.float32)
            buf5 = empty_strided_cuda((1, 256, 1, 9), (2304, 1, 2304, 256), torch.float32)
            # Topologically Sorted Source Nodes: [pixel_values, patch_embeds, flatten, embeddings, embedding, embeddings_1, hidden_states], Original ATen: [aten._to_copy, aten.convolution, aten.view, aten.transpose, aten.embedding, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_1.run(buf2, arg2_1, arg3_1, arg4_1, buf3, buf4, buf5, 2304, 128, stream=stream0)
            buf6 = empty_strided_cuda((1, 256, 1), (256, 1, 256), torch.float32)
            buf7 = empty_strided_cuda((1, 256, 1), (256, 1, 256), torch.float32)
            # Topologically Sorted Source Nodes: [pixel_values, patch_embeds, flatten, embeddings, embedding, embeddings_1, hidden_states], Original ATen: [aten._to_copy, aten.convolution, aten.view, aten.transpose, aten.embedding, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_2.run(buf3, buf4, buf5, buf6, buf7, 256, 9, stream=stream0)
            del buf3
            del buf4
            del buf5
            buf9 = empty_strided_cuda((1, 256, 1152), (294912, 1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [pixel_values, patch_embeds, flatten, embeddings, embedding, embeddings_1, hidden_states], Original ATen: [aten._to_copy, aten.convolution, aten.view, aten.transpose, aten.embedding, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_3.run(buf2, arg2_1, arg3_1, arg4_1, buf6, buf7, arg5_1, arg6_1, buf9, 256, 1152, stream=stream0)
            del arg5_1
            del arg6_1
            del buf6
            del buf7
            buf10 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg8_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf9, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg7_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf10)
            del arg7_1
            del arg8_1
            buf11 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg10_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf9, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg9_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf11)
            del arg10_1
            del arg9_1
            buf12 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg12_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf9, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg11_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf12)
            del arg11_1
            del arg12_1
            del buf9
            # Topologically Sorted Source Nodes: [linear, view, queries, linear_1, view_1, keys, linear_2, view_2, values, attn_output], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf13 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf10, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf11, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf12, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf10
            del buf11
            buf14 = buf13[0]
            assert_size_stride(buf14, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf14, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf13
            buf19 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [transpose_4, reshape, attn_output_3], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg14_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf14, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg13_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf19)
            del arg13_1
            del arg14_1
            buf20 = reinterpret_tensor(buf19, (1, 256, 1152), (294912, 1152, 1), 0); del buf19  # reuse
            buf24 = reinterpret_tensor(buf14, (1, 256, 1152), (294912, 1152, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [pixel_values, patch_embeds, flatten, embeddings, embedding, embeddings_1, attn_output_3, hidden_states_1, hidden_states_2], Original ATen: [aten._to_copy, aten.convolution, aten.view, aten.transpose, aten.embedding, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_4.run(buf20, buf2, arg2_1, arg3_1, arg4_1, arg15_1, arg16_1, buf24, 256, 1152, stream=stream0)
            del arg15_1
            del arg16_1
            del arg2_1
            del arg3_1
            del arg4_1
            buf25 = empty_strided_cuda((256, 4304), (4304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_2, hidden_states_3], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg18_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf24, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg17_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf25)
            del arg17_1
            del arg18_1
            buf26 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf25, buf26, 1101824, stream=stream0)
            buf27 = reinterpret_tensor(buf24, (256, 1152), (1152, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4, hidden_states_5], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg20_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf26, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg19_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf27)
            del arg19_1
            del arg20_1
            del buf26
            buf31 = reinterpret_tensor(buf2, (1, 256, 1152), (294912, 1152, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6, hidden_states_7], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf20, buf27, arg21_1, arg22_1, buf31, 256, 1152, stream=stream0)
            del arg21_1
            del arg22_1
            buf32 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg24_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf31, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg23_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf32)
            del arg23_1
            del arg24_1
            buf33 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg26_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf31, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg25_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf33)
            del arg25_1
            del arg26_1
            buf34 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg28_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf31, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg27_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf34)
            del arg27_1
            del arg28_1
            del buf31
            # Topologically Sorted Source Nodes: [linear_6, view_3, queries_1, linear_7, view_4, keys_1, linear_8, view_5, values_1, attn_output_4], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf35 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf32, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf33, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf34, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf32
            del buf33
            buf36 = buf35[0]
            assert_size_stride(buf36, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf36, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf35
            buf41 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [transpose_8, reshape_1, attn_output_7], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg30_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf36, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg29_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf41)
            del arg29_1
            del arg30_1
            buf45 = reinterpret_tensor(buf36, (1, 256, 1152), (294912, 1152, 1), 0); del buf36  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_9], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_7.run(buf20, buf27, buf41, arg31_1, arg32_1, buf45, 256, 1152, stream=stream0)
            del arg31_1
            del arg32_1
            buf46 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_9, hidden_states_10], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg34_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf45, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg33_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf46)
            del arg33_1
            del arg34_1
            del buf45
            buf47 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf46, buf47, 1101824, stream=stream0)
            del buf46
            buf48 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11, hidden_states_12], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg36_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf47, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg35_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf48)
            del arg35_1
            del arg36_1
            del buf47
            buf52 = empty_strided_cuda((1, 256, 1152), (294912, 1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_12, hidden_states_13, hidden_states_14], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf20, buf27, buf41, buf48, arg37_1, arg38_1, buf52, 256, 1152, stream=stream0)
            del arg37_1
            del arg38_1
            buf53 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg40_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf52, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg39_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf53)
            del arg39_1
            del arg40_1
            buf54 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg42_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf52, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg41_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf54)
            del arg41_1
            del arg42_1
            buf55 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg44_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf52, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg43_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf55)
            del arg43_1
            del arg44_1
            del buf52
            # Topologically Sorted Source Nodes: [linear_12, view_6, queries_2, linear_13, view_7, keys_2, linear_14, view_8, values_2, attn_output_8], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf56 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf53, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf54, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf55, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf53
            del buf54
            buf57 = buf56[0]
            assert_size_stride(buf57, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf57, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf56
            buf62 = buf55; del buf55  # reuse
            # Topologically Sorted Source Nodes: [transpose_12, reshape_2, attn_output_11], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg46_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf57, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg45_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf62)
            del arg45_1
            del arg46_1
            buf63 = buf20; del buf20  # reuse
            buf67 = reinterpret_tensor(buf57, (1, 256, 1152), (294912, 1152, 1), 0); del buf57  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_12, hidden_states_13, attn_output_11, hidden_states_15, hidden_states_16], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_9.run(buf63, buf27, buf41, buf48, buf62, arg47_1, arg48_1, buf67, 256, 1152, stream=stream0)
            del arg47_1
            del arg48_1
            del buf27
            del buf41
            del buf48
            buf68 = empty_strided_cuda((256, 4304), (4304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_16, hidden_states_17], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg50_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf67, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg49_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf68)
            del arg49_1
            del arg50_1
            buf69 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf68, buf69, 1101824, stream=stream0)
            buf70 = reinterpret_tensor(buf67, (256, 1152), (1152, 1), 0); del buf67  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18, hidden_states_19], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg52_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf69, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg51_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf70)
            del arg51_1
            del arg52_1
            del buf69
            buf74 = reinterpret_tensor(buf62, (1, 256, 1152), (294912, 1152, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20, hidden_states_21], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf63, buf70, arg53_1, arg54_1, buf74, 256, 1152, stream=stream0)
            del arg53_1
            del arg54_1
            buf75 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg56_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf74, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg55_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf75)
            del arg55_1
            del arg56_1
            buf76 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg58_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf74, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg57_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf76)
            del arg57_1
            del arg58_1
            buf77 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg60_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf74, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg59_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf77)
            del arg59_1
            del arg60_1
            del buf74
            # Topologically Sorted Source Nodes: [linear_18, view_9, queries_3, linear_19, view_10, keys_3, linear_20, view_11, values_3, attn_output_12], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf78 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf75, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf76, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf77, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf75
            del buf76
            buf79 = buf78[0]
            assert_size_stride(buf79, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf79, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf78
            buf84 = buf77; del buf77  # reuse
            # Topologically Sorted Source Nodes: [transpose_16, reshape_3, attn_output_15], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg62_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf79, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg61_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf84)
            del arg61_1
            del arg62_1
            buf88 = reinterpret_tensor(buf79, (1, 256, 1152), (294912, 1152, 1), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20, attn_output_15, hidden_states_22, hidden_states_23], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_7.run(buf63, buf70, buf84, arg63_1, arg64_1, buf88, 256, 1152, stream=stream0)
            del arg63_1
            del arg64_1
            buf89 = buf68; del buf68  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20, attn_output_15, hidden_states_22, hidden_states_23, hidden_states_24], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg66_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf88, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg65_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf89)
            del arg65_1
            del arg66_1
            del buf88
            buf90 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_24, hidden_states_25], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf89, buf90, 1101824, stream=stream0)
            del buf89
            buf91 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_24, hidden_states_25, hidden_states_26], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg68_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf90, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg67_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf91)
            del arg67_1
            del arg68_1
            del buf90
            buf95 = empty_strided_cuda((1, 256, 1152), (294912, 1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20, attn_output_15, hidden_states_22, hidden_states_26, hidden_states_27, hidden_states_28], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf63, buf70, buf84, buf91, arg69_1, arg70_1, buf95, 256, 1152, stream=stream0)
            del arg69_1
            del arg70_1
            buf96 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg72_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf95, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg71_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf96)
            del arg71_1
            del arg72_1
            buf97 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg74_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf95, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg73_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf97)
            del arg73_1
            del arg74_1
            buf98 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg76_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf95, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg75_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf98)
            del arg75_1
            del arg76_1
            del buf95
            # Topologically Sorted Source Nodes: [linear_24, view_12, queries_4, linear_25, view_13, keys_4, linear_26, view_14, values_4, attn_output_16], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf99 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf96, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf97, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf98, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf96
            del buf97
            buf100 = buf99[0]
            assert_size_stride(buf100, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf100, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf99
            buf105 = buf98; del buf98  # reuse
            # Topologically Sorted Source Nodes: [transpose_20, reshape_4, attn_output_19], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg78_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf100, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg77_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf105)
            del arg77_1
            del arg78_1
            buf106 = buf63; del buf63  # reuse
            buf110 = reinterpret_tensor(buf100, (1, 256, 1152), (294912, 1152, 1), 0); del buf100  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20, attn_output_15, hidden_states_22, hidden_states_26, hidden_states_27, attn_output_19, hidden_states_29, hidden_states_30], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_9.run(buf106, buf70, buf84, buf91, buf105, arg79_1, arg80_1, buf110, 256, 1152, stream=stream0)
            del arg79_1
            del arg80_1
            del buf105
            del buf70
            del buf84
            buf111 = empty_strided_cuda((256, 4304), (4304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_30, hidden_states_31], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg82_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf110, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg81_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf111)
            del arg81_1
            del arg82_1
            buf112 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf111, buf112, 1101824, stream=stream0)
            buf113 = reinterpret_tensor(buf110, (256, 1152), (1152, 1), 0); del buf110  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32, hidden_states_33], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg84_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf112, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg83_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf113)
            del arg83_1
            del arg84_1
            del buf112
            buf117 = reinterpret_tensor(buf91, (1, 256, 1152), (294912, 1152, 1), 0); del buf91  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_33, hidden_states_34, hidden_states_35], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf106, buf113, arg85_1, arg86_1, buf117, 256, 1152, stream=stream0)
            del arg85_1
            del arg86_1
            buf118 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg88_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf117, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg87_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf118)
            del arg87_1
            del arg88_1
            buf119 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg90_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf117, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg89_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf119)
            del arg89_1
            del arg90_1
            buf120 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg92_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf117, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg91_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf120)
            del arg91_1
            del arg92_1
            del buf117
            # Topologically Sorted Source Nodes: [linear_30, view_15, queries_5, linear_31, view_16, keys_5, linear_32, view_17, values_5, attn_output_20], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf121 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf118, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf119, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf120, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf118
            del buf119
            buf122 = buf121[0]
            assert_size_stride(buf122, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf122, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf121
            buf127 = buf120; del buf120  # reuse
            # Topologically Sorted Source Nodes: [transpose_24, reshape_5, attn_output_23], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg94_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf122, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg93_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf127)
            del arg93_1
            del arg94_1
            buf131 = reinterpret_tensor(buf122, (1, 256, 1152), (294912, 1152, 1), 0); del buf122  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_33, hidden_states_34, attn_output_23, hidden_states_36, hidden_states_37], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_7.run(buf106, buf113, buf127, arg95_1, arg96_1, buf131, 256, 1152, stream=stream0)
            del arg95_1
            del arg96_1
            buf132 = buf111; del buf111  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_33, hidden_states_34, attn_output_23, hidden_states_36, hidden_states_37, hidden_states_38], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg98_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf131, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg97_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf132)
            del arg97_1
            del arg98_1
            del buf131
            buf133 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_38, hidden_states_39], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf132, buf133, 1101824, stream=stream0)
            del buf132
            buf134 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_38, hidden_states_39, hidden_states_40], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg100_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf133, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg99_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf134)
            del arg100_1
            del arg99_1
            del buf133
            buf138 = empty_strided_cuda((1, 256, 1152), (294912, 1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_33, hidden_states_34, attn_output_23, hidden_states_36, hidden_states_40, hidden_states_41, hidden_states_42], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf106, buf113, buf127, buf134, arg101_1, arg102_1, buf138, 256, 1152, stream=stream0)
            del arg101_1
            del arg102_1
            buf139 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg104_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf138, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg103_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf139)
            del arg103_1
            del arg104_1
            buf140 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg106_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf138, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg105_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf140)
            del arg105_1
            del arg106_1
            buf141 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg108_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf138, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg107_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf141)
            del arg107_1
            del arg108_1
            del buf138
            # Topologically Sorted Source Nodes: [linear_36, view_18, queries_6, linear_37, view_19, keys_6, linear_38, view_20, values_6, attn_output_24], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf142 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf139, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf140, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf141, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf139
            del buf140
            buf143 = buf142[0]
            assert_size_stride(buf143, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf143, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf142
            buf148 = buf141; del buf141  # reuse
            # Topologically Sorted Source Nodes: [transpose_28, reshape_6, attn_output_27], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg110_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf143, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg109_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf148)
            del arg109_1
            del arg110_1
            buf149 = buf106; del buf106  # reuse
            buf153 = reinterpret_tensor(buf143, (1, 256, 1152), (294912, 1152, 1), 0); del buf143  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_33, hidden_states_34, attn_output_23, hidden_states_36, hidden_states_40, hidden_states_41, attn_output_27, hidden_states_43, hidden_states_44], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_9.run(buf149, buf113, buf127, buf134, buf148, arg111_1, arg112_1, buf153, 256, 1152, stream=stream0)
            del arg111_1
            del arg112_1
            del buf113
            del buf127
            del buf134
            buf154 = empty_strided_cuda((256, 4304), (4304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_44, hidden_states_45], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg114_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf153, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg113_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf154)
            del arg113_1
            del arg114_1
            buf155 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf154, buf155, 1101824, stream=stream0)
            buf156 = reinterpret_tensor(buf153, (256, 1152), (1152, 1), 0); del buf153  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, hidden_states_47], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg116_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf155, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg115_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf156)
            del arg115_1
            del arg116_1
            del buf155
            buf160 = reinterpret_tensor(buf148, (1, 256, 1152), (294912, 1152, 1), 0); del buf148  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_47, hidden_states_48, hidden_states_49], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf149, buf156, arg117_1, arg118_1, buf160, 256, 1152, stream=stream0)
            del arg117_1
            del arg118_1
            buf161 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg120_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf160, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg119_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf161)
            del arg119_1
            del arg120_1
            buf162 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg122_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf160, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg121_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf162)
            del arg121_1
            del arg122_1
            buf163 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg124_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf160, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg123_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf163)
            del arg123_1
            del arg124_1
            del buf160
            # Topologically Sorted Source Nodes: [linear_42, view_21, queries_7, linear_43, view_22, keys_7, linear_44, view_23, values_7, attn_output_28], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf164 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf161, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf162, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf163, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf161
            del buf162
            buf165 = buf164[0]
            assert_size_stride(buf165, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf165, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf164
            buf170 = buf163; del buf163  # reuse
            # Topologically Sorted Source Nodes: [transpose_32, reshape_7, attn_output_31], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg126_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf165, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg125_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf170)
            del arg125_1
            del arg126_1
            buf174 = reinterpret_tensor(buf165, (1, 256, 1152), (294912, 1152, 1), 0); del buf165  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_47, hidden_states_48, attn_output_31, hidden_states_50, hidden_states_51], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_7.run(buf149, buf156, buf170, arg127_1, arg128_1, buf174, 256, 1152, stream=stream0)
            del arg127_1
            del arg128_1
            buf175 = buf154; del buf154  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_47, hidden_states_48, attn_output_31, hidden_states_50, hidden_states_51, hidden_states_52], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg130_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf174, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg129_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf175)
            del arg129_1
            del arg130_1
            del buf174
            buf176 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_52, hidden_states_53], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf175, buf176, 1101824, stream=stream0)
            del buf175
            buf177 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_52, hidden_states_53, hidden_states_54], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg132_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf176, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg131_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf177)
            del arg131_1
            del arg132_1
            del buf176
            buf181 = empty_strided_cuda((1, 256, 1152), (294912, 1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_47, hidden_states_48, attn_output_31, hidden_states_50, hidden_states_54, hidden_states_55, hidden_states_56], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf149, buf156, buf170, buf177, arg133_1, arg134_1, buf181, 256, 1152, stream=stream0)
            del arg133_1
            del arg134_1
            buf182 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg136_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf181, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg135_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf182)
            del arg135_1
            del arg136_1
            buf183 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg138_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf181, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg137_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf183)
            del arg137_1
            del arg138_1
            buf184 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg140_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf181, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg139_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf184)
            del arg139_1
            del arg140_1
            del buf181
            # Topologically Sorted Source Nodes: [linear_48, view_24, queries_8, linear_49, view_25, keys_8, linear_50, view_26, values_8, attn_output_32], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf185 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf182, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf183, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf184, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf182
            del buf183
            buf186 = buf185[0]
            assert_size_stride(buf186, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf186, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf185
            buf191 = buf184; del buf184  # reuse
            # Topologically Sorted Source Nodes: [transpose_36, reshape_8, attn_output_35], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg142_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf186, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg141_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf191)
            del arg141_1
            del arg142_1
            buf192 = buf149; del buf149  # reuse
            buf196 = reinterpret_tensor(buf186, (1, 256, 1152), (294912, 1152, 1), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_47, hidden_states_48, attn_output_31, hidden_states_50, hidden_states_54, hidden_states_55, attn_output_35, hidden_states_57, hidden_states_58], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_9.run(buf192, buf156, buf170, buf177, buf191, arg143_1, arg144_1, buf196, 256, 1152, stream=stream0)
            del arg143_1
            del arg144_1
            del buf156
            del buf170
            del buf177
            buf197 = empty_strided_cuda((256, 4304), (4304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_58, hidden_states_59], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg146_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf196, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg145_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf197)
            del arg145_1
            del arg146_1
            buf198 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf197, buf198, 1101824, stream=stream0)
            buf199 = reinterpret_tensor(buf196, (256, 1152), (1152, 1), 0); del buf196  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60, hidden_states_61], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg148_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf198, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg147_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf199)
            del arg147_1
            del arg148_1
            del buf198
            buf203 = reinterpret_tensor(buf191, (1, 256, 1152), (294912, 1152, 1), 0); del buf191  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62, hidden_states_63], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf192, buf199, arg149_1, arg150_1, buf203, 256, 1152, stream=stream0)
            del arg149_1
            del arg150_1
            buf204 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg152_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf203, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg151_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf204)
            del arg151_1
            del arg152_1
            buf205 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg154_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf203, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg153_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf205)
            del arg153_1
            del arg154_1
            buf206 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg156_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf203, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg155_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf206)
            del arg155_1
            del arg156_1
            del buf203
            # Topologically Sorted Source Nodes: [linear_54, view_27, queries_9, linear_55, view_28, keys_9, linear_56, view_29, values_9, attn_output_36], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf207 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf204, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf205, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf206, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf204
            del buf205
            buf208 = buf207[0]
            assert_size_stride(buf208, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf208, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf207
            buf213 = buf206; del buf206  # reuse
            # Topologically Sorted Source Nodes: [transpose_40, reshape_9, attn_output_39], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg158_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf208, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg157_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf213)
            del arg157_1
            del arg158_1
            buf217 = reinterpret_tensor(buf208, (1, 256, 1152), (294912, 1152, 1), 0); del buf208  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62, attn_output_39, hidden_states_64, hidden_states_65], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_7.run(buf192, buf199, buf213, arg159_1, arg160_1, buf217, 256, 1152, stream=stream0)
            del arg159_1
            del arg160_1
            buf218 = buf197; del buf197  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62, attn_output_39, hidden_states_64, hidden_states_65, hidden_states_66], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg162_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf217, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg161_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf218)
            del arg161_1
            del arg162_1
            del buf217
            buf219 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_67], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf218, buf219, 1101824, stream=stream0)
            del buf218
            buf220 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_67, hidden_states_68], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg164_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf219, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg163_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf220)
            del arg163_1
            del arg164_1
            del buf219
            buf224 = empty_strided_cuda((1, 256, 1152), (294912, 1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62, attn_output_39, hidden_states_64, hidden_states_68, hidden_states_69, hidden_states_70], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf192, buf199, buf213, buf220, arg165_1, arg166_1, buf224, 256, 1152, stream=stream0)
            del arg165_1
            del arg166_1
            buf225 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg168_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf224, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg167_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf225)
            del arg167_1
            del arg168_1
            buf226 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg170_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf224, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg169_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf226)
            del arg169_1
            del arg170_1
            buf227 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg172_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf224, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg171_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf227)
            del arg171_1
            del arg172_1
            del buf224
            # Topologically Sorted Source Nodes: [linear_60, view_30, queries_10, linear_61, view_31, keys_10, linear_62, view_32, values_10, attn_output_40], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf228 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf225, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf226, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf227, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf225
            del buf226
            buf229 = buf228[0]
            assert_size_stride(buf229, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf229, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf228
            buf234 = buf227; del buf227  # reuse
            # Topologically Sorted Source Nodes: [transpose_44, reshape_10, attn_output_43], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg174_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf229, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg173_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf234)
            del arg173_1
            del arg174_1
            buf235 = buf192; del buf192  # reuse
            buf239 = reinterpret_tensor(buf229, (1, 256, 1152), (294912, 1152, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62, attn_output_39, hidden_states_64, hidden_states_68, hidden_states_69, attn_output_43, hidden_states_71, hidden_states_72], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_9.run(buf235, buf199, buf213, buf220, buf234, arg175_1, arg176_1, buf239, 256, 1152, stream=stream0)
            del arg175_1
            del arg176_1
            del buf199
            del buf213
            del buf220
            buf240 = empty_strided_cuda((256, 4304), (4304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_72, hidden_states_73], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg178_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf239, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg177_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf240)
            del arg177_1
            del arg178_1
            buf241 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_74], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf240, buf241, 1101824, stream=stream0)
            buf242 = reinterpret_tensor(buf239, (256, 1152), (1152, 1), 0); del buf239  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_74, hidden_states_75], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg180_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf241, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg179_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf242)
            del arg179_1
            del arg180_1
            del buf241
            buf246 = reinterpret_tensor(buf234, (1, 256, 1152), (294912, 1152, 1), 0); del buf234  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_75, hidden_states_76, hidden_states_77], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf235, buf242, arg181_1, arg182_1, buf246, 256, 1152, stream=stream0)
            del arg181_1
            del arg182_1
            buf247 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_66], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg184_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf246, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg183_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf247)
            del arg183_1
            del arg184_1
            buf248 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg186_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf246, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg185_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf248)
            del arg185_1
            del arg186_1
            buf249 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg188_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf246, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg187_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf249)
            del arg187_1
            del arg188_1
            del buf246
            # Topologically Sorted Source Nodes: [linear_66, view_33, queries_11, linear_67, view_34, keys_11, linear_68, view_35, values_11, attn_output_44], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf250 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf247, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf248, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf249, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf247
            del buf248
            buf251 = buf250[0]
            assert_size_stride(buf251, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf251, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf250
            buf256 = buf249; del buf249  # reuse
            # Topologically Sorted Source Nodes: [transpose_48, reshape_11, attn_output_47], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg190_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf251, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg189_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf256)
            del arg189_1
            del arg190_1
            buf260 = reinterpret_tensor(buf251, (1, 256, 1152), (294912, 1152, 1), 0); del buf251  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_75, hidden_states_76, attn_output_47, hidden_states_78, hidden_states_79], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_7.run(buf235, buf242, buf256, arg191_1, arg192_1, buf260, 256, 1152, stream=stream0)
            del arg191_1
            del arg192_1
            buf261 = buf240; del buf240  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_75, hidden_states_76, attn_output_47, hidden_states_78, hidden_states_79, hidden_states_80], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg194_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf260, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg193_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf261)
            del arg193_1
            del arg194_1
            del buf260
            buf262 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf261, buf262, 1101824, stream=stream0)
            del buf261
            buf263 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81, hidden_states_82], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg196_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf262, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg195_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf263)
            del arg195_1
            del arg196_1
            del buf262
            buf267 = empty_strided_cuda((1, 256, 1152), (294912, 1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_75, hidden_states_76, attn_output_47, hidden_states_78, hidden_states_82, hidden_states_83, hidden_states_84], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf235, buf242, buf256, buf263, arg197_1, arg198_1, buf267, 256, 1152, stream=stream0)
            del arg197_1
            del arg198_1
            buf268 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_72], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg200_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf267, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg199_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf268)
            del arg199_1
            del arg200_1
            buf269 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_73], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg202_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf267, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg201_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf269)
            del arg201_1
            del arg202_1
            buf270 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_74], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg204_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf267, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg203_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf270)
            del arg203_1
            del arg204_1
            del buf267
            # Topologically Sorted Source Nodes: [linear_72, view_36, queries_12, linear_73, view_37, keys_12, linear_74, view_38, values_12, attn_output_48], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf271 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf268, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf269, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf270, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf268
            del buf269
            buf272 = buf271[0]
            assert_size_stride(buf272, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf272, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf271
            buf277 = buf270; del buf270  # reuse
            # Topologically Sorted Source Nodes: [transpose_52, reshape_12, attn_output_51], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg206_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf272, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg205_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf277)
            del arg205_1
            del arg206_1
            buf278 = buf235; del buf235  # reuse
            buf282 = reinterpret_tensor(buf272, (1, 256, 1152), (294912, 1152, 1), 0); del buf272  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_75, hidden_states_76, attn_output_47, hidden_states_78, hidden_states_82, hidden_states_83, attn_output_51, hidden_states_85, hidden_states_86], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_9.run(buf278, buf242, buf256, buf263, buf277, arg207_1, arg208_1, buf282, 256, 1152, stream=stream0)
            del arg207_1
            del arg208_1
            del buf242
            del buf256
            del buf263
            buf283 = empty_strided_cuda((256, 4304), (4304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_86, hidden_states_87], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg210_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf282, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg209_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf283)
            del arg209_1
            del arg210_1
            buf284 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_87, hidden_states_88], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf283, buf284, 1101824, stream=stream0)
            buf285 = reinterpret_tensor(buf282, (256, 1152), (1152, 1), 0); del buf282  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_87, hidden_states_88, hidden_states_89], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg212_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf284, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg211_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf285)
            del arg211_1
            del arg212_1
            del buf284
            buf289 = reinterpret_tensor(buf277, (1, 256, 1152), (294912, 1152, 1), 0); del buf277  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_89, hidden_states_90, hidden_states_91], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf278, buf285, arg213_1, arg214_1, buf289, 256, 1152, stream=stream0)
            del arg213_1
            del arg214_1
            buf290 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_78], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg216_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf289, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg215_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf290)
            del arg215_1
            del arg216_1
            buf291 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_79], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg218_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf289, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg217_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf291)
            del arg217_1
            del arg218_1
            buf292 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_80], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg220_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf289, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg219_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf292)
            del arg219_1
            del arg220_1
            del buf289
            # Topologically Sorted Source Nodes: [linear_78, view_39, queries_13, linear_79, view_40, keys_13, linear_80, view_41, values_13, attn_output_52], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf293 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf290, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf291, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf292, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf290
            del buf291
            buf294 = buf293[0]
            assert_size_stride(buf294, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf294, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf293
            buf299 = buf292; del buf292  # reuse
            # Topologically Sorted Source Nodes: [transpose_56, reshape_13, attn_output_55], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg222_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf294, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg221_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf299)
            del arg221_1
            del arg222_1
            buf303 = reinterpret_tensor(buf294, (1, 256, 1152), (294912, 1152, 1), 0); del buf294  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_89, hidden_states_90, attn_output_55, hidden_states_92, hidden_states_93], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_7.run(buf278, buf285, buf299, arg223_1, arg224_1, buf303, 256, 1152, stream=stream0)
            del arg223_1
            del arg224_1
            buf304 = buf283; del buf283  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_89, hidden_states_90, attn_output_55, hidden_states_92, hidden_states_93, hidden_states_94], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg226_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf303, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg225_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf304)
            del arg225_1
            del arg226_1
            del buf303
            buf305 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf304, buf305, 1101824, stream=stream0)
            del buf304
            buf306 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg228_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf305, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg227_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf306)
            del arg227_1
            del arg228_1
            del buf305
            buf310 = empty_strided_cuda((1, 256, 1152), (294912, 1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_89, hidden_states_90, attn_output_55, hidden_states_92, hidden_states_96, hidden_states_97, hidden_states_98], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf278, buf285, buf299, buf306, arg229_1, arg230_1, buf310, 256, 1152, stream=stream0)
            del arg229_1
            del arg230_1
            buf311 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_84], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg232_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf310, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg231_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf311)
            del arg231_1
            del arg232_1
            buf312 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_85], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg234_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf310, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg233_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf312)
            del arg233_1
            del arg234_1
            buf313 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg236_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf310, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg235_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf313)
            del arg235_1
            del arg236_1
            del buf310
            # Topologically Sorted Source Nodes: [linear_84, view_42, queries_14, linear_85, view_43, keys_14, linear_86, view_44, values_14, attn_output_56], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf314 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf311, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf312, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf313, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf311
            del buf312
            buf315 = buf314[0]
            assert_size_stride(buf315, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf315, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf314
            buf320 = buf313; del buf313  # reuse
            # Topologically Sorted Source Nodes: [transpose_60, reshape_14, attn_output_59], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg238_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf315, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg237_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf320)
            del arg237_1
            del arg238_1
            buf321 = buf278; del buf278  # reuse
            buf325 = reinterpret_tensor(buf315, (1, 256, 1152), (294912, 1152, 1), 0); del buf315  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_89, hidden_states_90, attn_output_55, hidden_states_92, hidden_states_96, hidden_states_97, attn_output_59, hidden_states_99, hidden_states_100], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_9.run(buf321, buf285, buf299, buf306, buf320, arg239_1, arg240_1, buf325, 256, 1152, stream=stream0)
            del arg239_1
            del arg240_1
            del buf285
            del buf299
            del buf306
            buf326 = empty_strided_cuda((256, 4304), (4304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg242_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf325, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg241_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf326)
            del arg241_1
            del arg242_1
            buf327 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf326, buf327, 1101824, stream=stream0)
            buf328 = reinterpret_tensor(buf325, (256, 1152), (1152, 1), 0); del buf325  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102, hidden_states_103], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg244_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf327, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg243_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf328)
            del arg243_1
            del arg244_1
            del buf327
            buf332 = reinterpret_tensor(buf320, (1, 256, 1152), (294912, 1152, 1), 0); del buf320  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_103, hidden_states_104, hidden_states_105], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf321, buf328, arg245_1, arg246_1, buf332, 256, 1152, stream=stream0)
            del arg245_1
            del arg246_1
            buf333 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_90], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg248_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf332, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg247_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf333)
            del arg247_1
            del arg248_1
            buf334 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_91], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg250_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf332, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg249_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf334)
            del arg249_1
            del arg250_1
            buf335 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_92], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg252_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf332, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg251_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf335)
            del arg251_1
            del arg252_1
            del buf332
            # Topologically Sorted Source Nodes: [linear_90, view_45, queries_15, linear_91, view_46, keys_15, linear_92, view_47, values_15, attn_output_60], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf336 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf333, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf334, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf335, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf333
            del buf334
            buf337 = buf336[0]
            assert_size_stride(buf337, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf337, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf336
            buf342 = buf335; del buf335  # reuse
            # Topologically Sorted Source Nodes: [transpose_64, reshape_15, attn_output_63], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg254_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf337, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg253_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf342)
            del arg253_1
            del arg254_1
            buf346 = reinterpret_tensor(buf337, (1, 256, 1152), (294912, 1152, 1), 0); del buf337  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_103, hidden_states_104, attn_output_63, hidden_states_106, hidden_states_107], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_7.run(buf321, buf328, buf342, arg255_1, arg256_1, buf346, 256, 1152, stream=stream0)
            del arg255_1
            del arg256_1
            buf347 = buf326; del buf326  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_103, hidden_states_104, attn_output_63, hidden_states_106, hidden_states_107, hidden_states_108], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg258_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf346, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg257_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf347)
            del arg257_1
            del arg258_1
            del buf346
            buf348 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf347, buf348, 1101824, stream=stream0)
            del buf347
            buf349 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109, hidden_states_110], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg260_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf348, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg259_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf349)
            del arg259_1
            del arg260_1
            del buf348
            buf353 = empty_strided_cuda((1, 256, 1152), (294912, 1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_103, hidden_states_104, attn_output_63, hidden_states_106, hidden_states_110, hidden_states_111, hidden_states_112], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf321, buf328, buf342, buf349, arg261_1, arg262_1, buf353, 256, 1152, stream=stream0)
            del arg261_1
            del arg262_1
            buf354 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_96], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg264_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf353, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg263_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf354)
            del arg263_1
            del arg264_1
            buf355 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_97], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg266_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf353, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg265_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf355)
            del arg265_1
            del arg266_1
            buf356 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_98], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg268_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf353, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg267_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf356)
            del arg267_1
            del arg268_1
            del buf353
            # Topologically Sorted Source Nodes: [linear_96, view_48, queries_16, linear_97, view_49, keys_16, linear_98, view_50, values_16, attn_output_64], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf357 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf354, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf355, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf356, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf354
            del buf355
            buf358 = buf357[0]
            assert_size_stride(buf358, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf358, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf357
            buf363 = buf356; del buf356  # reuse
            # Topologically Sorted Source Nodes: [transpose_68, reshape_16, attn_output_67], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg270_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf358, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg269_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf363)
            del arg269_1
            del arg270_1
            buf364 = buf321; del buf321  # reuse
            buf368 = reinterpret_tensor(buf358, (1, 256, 1152), (294912, 1152, 1), 0); del buf358  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_103, hidden_states_104, attn_output_63, hidden_states_106, hidden_states_110, hidden_states_111, attn_output_67, hidden_states_113, hidden_states_114], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_9.run(buf364, buf328, buf342, buf349, buf363, arg271_1, arg272_1, buf368, 256, 1152, stream=stream0)
            del arg271_1
            del arg272_1
            del buf328
            del buf342
            del buf349
            buf369 = empty_strided_cuda((256, 4304), (4304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_114, hidden_states_115], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg274_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf368, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg273_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf369)
            del arg273_1
            del arg274_1
            buf370 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_115, hidden_states_116], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf369, buf370, 1101824, stream=stream0)
            buf371 = reinterpret_tensor(buf368, (256, 1152), (1152, 1), 0); del buf368  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_115, hidden_states_116, hidden_states_117], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg276_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf370, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg275_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf371)
            del arg275_1
            del arg276_1
            del buf370
            buf375 = reinterpret_tensor(buf363, (1, 256, 1152), (294912, 1152, 1), 0); del buf363  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_117, hidden_states_118, hidden_states_119], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf364, buf371, arg277_1, arg278_1, buf375, 256, 1152, stream=stream0)
            del arg277_1
            del arg278_1
            buf376 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_102], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg280_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf375, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg279_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf376)
            del arg279_1
            del arg280_1
            buf377 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_103], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg282_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf375, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg281_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf377)
            del arg281_1
            del arg282_1
            buf378 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_104], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg284_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf375, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg283_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf378)
            del arg283_1
            del arg284_1
            del buf375
            # Topologically Sorted Source Nodes: [linear_102, view_51, queries_17, linear_103, view_52, keys_17, linear_104, view_53, values_17, attn_output_68], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf379 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf376, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf377, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf378, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf376
            del buf377
            buf380 = buf379[0]
            assert_size_stride(buf380, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf380, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf379
            buf385 = buf378; del buf378  # reuse
            # Topologically Sorted Source Nodes: [transpose_72, reshape_17, attn_output_71], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg286_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf380, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg285_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf385)
            del arg285_1
            del arg286_1
            buf389 = reinterpret_tensor(buf380, (1, 256, 1152), (294912, 1152, 1), 0); del buf380  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_117, hidden_states_118, attn_output_71, hidden_states_120, hidden_states_121], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_7.run(buf364, buf371, buf385, arg287_1, arg288_1, buf389, 256, 1152, stream=stream0)
            del arg287_1
            del arg288_1
            buf390 = buf369; del buf369  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_117, hidden_states_118, attn_output_71, hidden_states_120, hidden_states_121, hidden_states_122], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg290_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf389, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg289_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf390)
            del arg289_1
            del arg290_1
            del buf389
            buf391 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_122, hidden_states_123], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf390, buf391, 1101824, stream=stream0)
            del buf390
            buf392 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_122, hidden_states_123, hidden_states_124], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg292_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf391, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg291_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf392)
            del arg291_1
            del arg292_1
            del buf391
            buf396 = empty_strided_cuda((1, 256, 1152), (294912, 1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_117, hidden_states_118, attn_output_71, hidden_states_120, hidden_states_124, hidden_states_125, hidden_states_126], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf364, buf371, buf385, buf392, arg293_1, arg294_1, buf396, 256, 1152, stream=stream0)
            del arg293_1
            del arg294_1
            buf397 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_108], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg296_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf396, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg295_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf397)
            del arg295_1
            del arg296_1
            buf398 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_109], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg298_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf396, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg297_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf398)
            del arg297_1
            del arg298_1
            buf399 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_110], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg300_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf396, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg299_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf399)
            del arg299_1
            del arg300_1
            del buf396
            # Topologically Sorted Source Nodes: [linear_108, view_54, queries_18, linear_109, view_55, keys_18, linear_110, view_56, values_18, attn_output_72], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf400 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf397, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf398, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf399, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf397
            del buf398
            buf401 = buf400[0]
            assert_size_stride(buf401, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf401, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf400
            buf406 = buf399; del buf399  # reuse
            # Topologically Sorted Source Nodes: [transpose_76, reshape_18, attn_output_75], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg302_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf401, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg301_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf406)
            del arg301_1
            del arg302_1
            buf407 = buf364; del buf364  # reuse
            buf411 = reinterpret_tensor(buf401, (1, 256, 1152), (294912, 1152, 1), 0); del buf401  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_117, hidden_states_118, attn_output_71, hidden_states_120, hidden_states_124, hidden_states_125, attn_output_75, hidden_states_127, hidden_states_128], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_9.run(buf407, buf371, buf385, buf392, buf406, arg303_1, arg304_1, buf411, 256, 1152, stream=stream0)
            del arg303_1
            del arg304_1
            del buf371
            del buf385
            del buf392
            buf412 = empty_strided_cuda((256, 4304), (4304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_128, hidden_states_129], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg306_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf411, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg305_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf412)
            del arg305_1
            del arg306_1
            buf413 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_129, hidden_states_130], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf412, buf413, 1101824, stream=stream0)
            buf414 = reinterpret_tensor(buf411, (256, 1152), (1152, 1), 0); del buf411  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_129, hidden_states_130, hidden_states_131], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg308_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf413, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg307_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf414)
            del arg307_1
            del arg308_1
            del buf413
            buf418 = reinterpret_tensor(buf406, (1, 256, 1152), (294912, 1152, 1), 0); del buf406  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_131, hidden_states_132, hidden_states_133], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf407, buf414, arg309_1, arg310_1, buf418, 256, 1152, stream=stream0)
            del arg309_1
            del arg310_1
            buf419 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_114], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg312_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf418, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg311_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf419)
            del arg311_1
            del arg312_1
            buf420 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_115], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg314_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf418, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg313_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf420)
            del arg313_1
            del arg314_1
            buf421 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_116], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg316_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf418, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg315_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf421)
            del arg315_1
            del arg316_1
            del buf418
            # Topologically Sorted Source Nodes: [linear_114, view_57, queries_19, linear_115, view_58, keys_19, linear_116, view_59, values_19, attn_output_76], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf422 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf419, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf420, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf421, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf419
            del buf420
            buf423 = buf422[0]
            assert_size_stride(buf423, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf423, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf422
            buf428 = buf421; del buf421  # reuse
            # Topologically Sorted Source Nodes: [transpose_80, reshape_19, attn_output_79], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg318_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf423, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg317_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf428)
            del arg317_1
            del arg318_1
            buf432 = reinterpret_tensor(buf423, (1, 256, 1152), (294912, 1152, 1), 0); del buf423  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_131, hidden_states_132, attn_output_79, hidden_states_134, hidden_states_135], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_7.run(buf407, buf414, buf428, arg319_1, arg320_1, buf432, 256, 1152, stream=stream0)
            del arg319_1
            del arg320_1
            buf433 = buf412; del buf412  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_131, hidden_states_132, attn_output_79, hidden_states_134, hidden_states_135, hidden_states_136], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg322_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf432, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg321_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf433)
            del arg321_1
            del arg322_1
            del buf432
            buf434 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_136, hidden_states_137], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf433, buf434, 1101824, stream=stream0)
            del buf433
            buf435 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_136, hidden_states_137, hidden_states_138], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg324_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf434, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg323_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf435)
            del arg323_1
            del arg324_1
            del buf434
            buf439 = empty_strided_cuda((1, 256, 1152), (294912, 1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_131, hidden_states_132, attn_output_79, hidden_states_134, hidden_states_138, hidden_states_139, hidden_states_140], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf407, buf414, buf428, buf435, arg325_1, arg326_1, buf439, 256, 1152, stream=stream0)
            del arg325_1
            del arg326_1
            buf440 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_120], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg328_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf439, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg327_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf440)
            del arg327_1
            del arg328_1
            buf441 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_121], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg330_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf439, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg329_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf441)
            del arg329_1
            del arg330_1
            buf442 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_122], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg332_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf439, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg331_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf442)
            del arg331_1
            del arg332_1
            del buf439
            # Topologically Sorted Source Nodes: [linear_120, view_60, queries_20, linear_121, view_61, keys_20, linear_122, view_62, values_20, attn_output_80], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf443 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf440, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf441, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf442, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf440
            del buf441
            buf444 = buf443[0]
            assert_size_stride(buf444, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf444, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf443
            buf449 = buf442; del buf442  # reuse
            # Topologically Sorted Source Nodes: [transpose_84, reshape_20, attn_output_83], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg334_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf444, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg333_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf449)
            del arg333_1
            del arg334_1
            buf450 = buf407; del buf407  # reuse
            buf454 = reinterpret_tensor(buf444, (1, 256, 1152), (294912, 1152, 1), 0); del buf444  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_131, hidden_states_132, attn_output_79, hidden_states_134, hidden_states_138, hidden_states_139, attn_output_83, hidden_states_141, hidden_states_142], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_9.run(buf450, buf414, buf428, buf435, buf449, arg335_1, arg336_1, buf454, 256, 1152, stream=stream0)
            del arg335_1
            del arg336_1
            del buf414
            del buf428
            del buf435
            buf455 = empty_strided_cuda((256, 4304), (4304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_142, hidden_states_143], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg338_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf454, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg337_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf455)
            del arg337_1
            del arg338_1
            buf456 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_143, hidden_states_144], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf455, buf456, 1101824, stream=stream0)
            buf457 = reinterpret_tensor(buf454, (256, 1152), (1152, 1), 0); del buf454  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_143, hidden_states_144, hidden_states_145], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg340_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf456, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg339_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf457)
            del arg339_1
            del arg340_1
            del buf456
            buf461 = reinterpret_tensor(buf449, (1, 256, 1152), (294912, 1152, 1), 0); del buf449  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_145, hidden_states_146, hidden_states_147], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf450, buf457, arg341_1, arg342_1, buf461, 256, 1152, stream=stream0)
            del arg341_1
            del arg342_1
            buf462 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_126], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg344_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf461, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg343_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf462)
            del arg343_1
            del arg344_1
            buf463 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_127], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg346_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf461, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg345_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf463)
            del arg345_1
            del arg346_1
            buf464 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_128], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg348_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf461, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg347_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf464)
            del arg347_1
            del arg348_1
            del buf461
            # Topologically Sorted Source Nodes: [linear_126, view_63, queries_21, linear_127, view_64, keys_21, linear_128, view_65, values_21, attn_output_84], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf465 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf462, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf463, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf464, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf462
            del buf463
            buf466 = buf465[0]
            assert_size_stride(buf466, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf466, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf465
            buf471 = buf464; del buf464  # reuse
            # Topologically Sorted Source Nodes: [transpose_88, reshape_21, attn_output_87], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg350_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf466, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg349_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf471)
            del arg349_1
            del arg350_1
            buf475 = reinterpret_tensor(buf466, (1, 256, 1152), (294912, 1152, 1), 0); del buf466  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_145, hidden_states_146, attn_output_87, hidden_states_148, hidden_states_149], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_7.run(buf450, buf457, buf471, arg351_1, arg352_1, buf475, 256, 1152, stream=stream0)
            del arg351_1
            del arg352_1
            buf476 = buf455; del buf455  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_145, hidden_states_146, attn_output_87, hidden_states_148, hidden_states_149, hidden_states_150], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg354_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf475, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg353_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf476)
            del arg353_1
            del arg354_1
            del buf475
            buf477 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_150, hidden_states_151], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf476, buf477, 1101824, stream=stream0)
            del buf476
            buf478 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_150, hidden_states_151, hidden_states_152], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg356_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf477, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg355_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf478)
            del arg355_1
            del arg356_1
            del buf477
            buf482 = empty_strided_cuda((1, 256, 1152), (294912, 1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_145, hidden_states_146, attn_output_87, hidden_states_148, hidden_states_152, hidden_states_153, hidden_states_154], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf450, buf457, buf471, buf478, arg357_1, arg358_1, buf482, 256, 1152, stream=stream0)
            del arg357_1
            del arg358_1
            buf483 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_132], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg360_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf482, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg359_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf483)
            del arg359_1
            del arg360_1
            buf484 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_133], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg362_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf482, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg361_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf484)
            del arg361_1
            del arg362_1
            buf485 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_134], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg364_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf482, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg363_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf485)
            del arg363_1
            del arg364_1
            del buf482
            # Topologically Sorted Source Nodes: [linear_132, view_66, queries_22, linear_133, view_67, keys_22, linear_134, view_68, values_22, attn_output_88], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf486 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf483, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf484, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf485, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf483
            del buf484
            buf487 = buf486[0]
            assert_size_stride(buf487, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf487, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf486
            buf492 = buf485; del buf485  # reuse
            # Topologically Sorted Source Nodes: [transpose_92, reshape_22, attn_output_91], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg366_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf487, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg365_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf492)
            del arg365_1
            del arg366_1
            buf493 = buf450; del buf450  # reuse
            buf497 = reinterpret_tensor(buf487, (1, 256, 1152), (294912, 1152, 1), 0); del buf487  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_145, hidden_states_146, attn_output_87, hidden_states_148, hidden_states_152, hidden_states_153, attn_output_91, hidden_states_155, hidden_states_156], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_9.run(buf493, buf457, buf471, buf478, buf492, arg367_1, arg368_1, buf497, 256, 1152, stream=stream0)
            del arg367_1
            del arg368_1
            del buf457
            del buf471
            del buf478
            buf498 = empty_strided_cuda((256, 4304), (4304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_156, hidden_states_157], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg370_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf497, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg369_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf498)
            del arg369_1
            del arg370_1
            buf499 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_157, hidden_states_158], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf498, buf499, 1101824, stream=stream0)
            buf500 = reinterpret_tensor(buf497, (256, 1152), (1152, 1), 0); del buf497  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_157, hidden_states_158, hidden_states_159], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg372_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf499, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg371_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf500)
            del arg371_1
            del arg372_1
            del buf499
            buf504 = reinterpret_tensor(buf492, (1, 256, 1152), (294912, 1152, 1), 0); del buf492  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_159, hidden_states_160, hidden_states_161], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf493, buf500, arg373_1, arg374_1, buf504, 256, 1152, stream=stream0)
            del arg373_1
            del arg374_1
            buf505 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_138], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg376_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf504, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg375_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf505)
            del arg375_1
            del arg376_1
            buf506 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_139], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg378_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf504, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg377_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf506)
            del arg377_1
            del arg378_1
            buf507 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_140], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg380_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf504, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg379_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf507)
            del arg379_1
            del arg380_1
            del buf504
            # Topologically Sorted Source Nodes: [linear_138, view_69, queries_23, linear_139, view_70, keys_23, linear_140, view_71, values_23, attn_output_92], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf508 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf505, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf506, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf507, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf505
            del buf506
            buf509 = buf508[0]
            assert_size_stride(buf509, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf509, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf508
            buf514 = buf507; del buf507  # reuse
            # Topologically Sorted Source Nodes: [transpose_96, reshape_23, attn_output_95], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg382_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf509, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg381_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf514)
            del arg381_1
            del arg382_1
            buf518 = reinterpret_tensor(buf509, (1, 256, 1152), (294912, 1152, 1), 0); del buf509  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_159, hidden_states_160, attn_output_95, hidden_states_162, hidden_states_163], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_7.run(buf493, buf500, buf514, arg383_1, arg384_1, buf518, 256, 1152, stream=stream0)
            del arg383_1
            del arg384_1
            buf519 = buf498; del buf498  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_159, hidden_states_160, attn_output_95, hidden_states_162, hidden_states_163, hidden_states_164], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg386_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf518, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg385_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf519)
            del arg385_1
            del arg386_1
            del buf518
            buf520 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_164, hidden_states_165], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf519, buf520, 1101824, stream=stream0)
            del buf519
            buf521 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_164, hidden_states_165, hidden_states_166], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg388_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf520, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg387_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf521)
            del arg387_1
            del arg388_1
            del buf520
            buf525 = empty_strided_cuda((1, 256, 1152), (294912, 1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_159, hidden_states_160, attn_output_95, hidden_states_162, hidden_states_166, hidden_states_167, hidden_states_168], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf493, buf500, buf514, buf521, arg389_1, arg390_1, buf525, 256, 1152, stream=stream0)
            del arg389_1
            del arg390_1
            buf526 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_144], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg392_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf525, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg391_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf526)
            del arg391_1
            del arg392_1
            buf527 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_145], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg394_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf525, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg393_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf527)
            del arg393_1
            del arg394_1
            buf528 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_146], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg396_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf525, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg395_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf528)
            del arg395_1
            del arg396_1
            del buf525
            # Topologically Sorted Source Nodes: [linear_144, view_72, queries_24, linear_145, view_73, keys_24, linear_146, view_74, values_24, attn_output_96], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf529 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf526, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf527, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf528, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf526
            del buf527
            buf530 = buf529[0]
            assert_size_stride(buf530, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf530, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf529
            buf535 = buf528; del buf528  # reuse
            # Topologically Sorted Source Nodes: [transpose_100, reshape_24, attn_output_99], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg398_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf530, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg397_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf535)
            del arg397_1
            del arg398_1
            buf536 = buf493; del buf493  # reuse
            buf540 = reinterpret_tensor(buf530, (1, 256, 1152), (294912, 1152, 1), 0); del buf530  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_159, hidden_states_160, attn_output_95, hidden_states_162, hidden_states_166, hidden_states_167, attn_output_99, hidden_states_169, hidden_states_170], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_9.run(buf536, buf500, buf514, buf521, buf535, arg399_1, arg400_1, buf540, 256, 1152, stream=stream0)
            del arg399_1
            del arg400_1
            del buf500
            del buf514
            del buf521
            buf541 = empty_strided_cuda((256, 4304), (4304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_170, hidden_states_171], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg402_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf540, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg401_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf541)
            del arg401_1
            del arg402_1
            buf542 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_171, hidden_states_172], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf541, buf542, 1101824, stream=stream0)
            buf543 = reinterpret_tensor(buf540, (256, 1152), (1152, 1), 0); del buf540  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_171, hidden_states_172, hidden_states_173], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg404_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf542, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg403_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf543)
            del arg403_1
            del arg404_1
            del buf542
            buf547 = reinterpret_tensor(buf535, (1, 256, 1152), (294912, 1152, 1), 0); del buf535  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_173, hidden_states_174, hidden_states_175], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf536, buf543, arg405_1, arg406_1, buf547, 256, 1152, stream=stream0)
            del arg405_1
            del arg406_1
            buf548 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_150], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg408_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf547, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg407_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf548)
            del arg407_1
            del arg408_1
            buf549 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_151], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg410_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf547, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg409_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf549)
            del arg409_1
            del arg410_1
            buf550 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_152], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg412_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf547, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg411_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf550)
            del arg411_1
            del arg412_1
            del buf547
            # Topologically Sorted Source Nodes: [linear_150, view_75, queries_25, linear_151, view_76, keys_25, linear_152, view_77, values_25, attn_output_100], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf551 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf548, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf549, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf550, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf548
            del buf549
            buf552 = buf551[0]
            assert_size_stride(buf552, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf552, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf551
            buf557 = buf550; del buf550  # reuse
            # Topologically Sorted Source Nodes: [transpose_104, reshape_25, attn_output_103], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg414_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf552, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg413_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf557)
            del arg413_1
            del arg414_1
            buf561 = reinterpret_tensor(buf552, (1, 256, 1152), (294912, 1152, 1), 0); del buf552  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_173, hidden_states_174, attn_output_103, hidden_states_176, hidden_states_177], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_7.run(buf536, buf543, buf557, arg415_1, arg416_1, buf561, 256, 1152, stream=stream0)
            del arg415_1
            del arg416_1
            buf562 = buf541; del buf541  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_173, hidden_states_174, attn_output_103, hidden_states_176, hidden_states_177, hidden_states_178], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg418_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf561, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg417_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf562)
            del arg417_1
            del arg418_1
            del buf561
            buf563 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_178, hidden_states_179], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf562, buf563, 1101824, stream=stream0)
            del buf562
            buf564 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_178, hidden_states_179, hidden_states_180], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg420_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf563, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg419_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf564)
            del arg419_1
            del arg420_1
            del buf563
            buf568 = empty_strided_cuda((1, 256, 1152), (294912, 1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_173, hidden_states_174, attn_output_103, hidden_states_176, hidden_states_180, hidden_states_181, hidden_states_182], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf536, buf543, buf557, buf564, arg421_1, arg422_1, buf568, 256, 1152, stream=stream0)
            del arg421_1
            del arg422_1
            buf569 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_156], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg424_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf568, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg423_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf569)
            del arg423_1
            del arg424_1
            buf570 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_157], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg426_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf568, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg425_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf570)
            del arg425_1
            del arg426_1
            buf571 = empty_strided_cuda((256, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_158], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg428_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf568, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg427_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf571)
            del arg427_1
            del arg428_1
            del buf568
            # Topologically Sorted Source Nodes: [linear_156, view_78, queries_26, linear_157, view_79, keys_26, linear_158, view_80, values_26, attn_output_104], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
            buf572 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf569, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf570, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), reinterpret_tensor(buf571, (1, 16, 256, 72), (294912, 72, 1152, 1), 0), scale=0.11785113019775792)
            del buf569
            del buf570
            buf573 = buf572[0]
            assert_size_stride(buf573, (1, 16, 256, 72), (294912, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf573, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf572
            buf578 = buf571; del buf571  # reuse
            # Topologically Sorted Source Nodes: [transpose_108, reshape_26, attn_output_107], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg430_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf573, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg429_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf578)
            del arg429_1
            del arg430_1
            buf579 = buf536; del buf536  # reuse
            buf583 = reinterpret_tensor(buf573, (1, 256, 1152), (294912, 1152, 1), 0); del buf573  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_173, hidden_states_174, attn_output_103, hidden_states_176, hidden_states_180, hidden_states_181, attn_output_107, hidden_states_183, hidden_states_184], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_9.run(buf579, buf543, buf557, buf564, buf578, arg431_1, arg432_1, buf583, 256, 1152, stream=stream0)
            del arg431_1
            del arg432_1
            del buf543
            del buf557
            del buf564
            del buf578
            buf584 = empty_strided_cuda((256, 4304), (4304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_184, hidden_states_185], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg434_1, (256, 4304), (0, 1), 0), reinterpret_tensor(buf583, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg433_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf584)
            del arg433_1
            del arg434_1
            buf585 = empty_strided_cuda((1, 256, 4304), (1114112, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_185, hidden_states_186], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_5.run(buf584, buf585, 1101824, stream=stream0)
            del buf584
            buf586 = reinterpret_tensor(buf583, (256, 1152), (1152, 1), 0); del buf583  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_185, hidden_states_186, hidden_states_187], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg436_1, (256, 1152), (0, 1), 0), reinterpret_tensor(buf585, (256, 4304), (4352, 1), 0), reinterpret_tensor(arg435_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf586)
            del arg435_1
            del arg436_1
            del buf585
            buf590 = buf579; del buf579  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_187, hidden_states_188, last_hidden_state], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_10.run(buf590, buf586, arg437_1, arg438_1, 256, 1152, stream=stream0)
            del arg437_1
            del arg438_1
            del buf586
            buf591 = empty_strided_cuda((256, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_187, hidden_states_188, last_hidden_state, hidden_states_189], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg440_1, (256, 2048), (0, 1), 0), reinterpret_tensor(buf590, (256, 1152), (1152, 1), 0), reinterpret_tensor(arg439_1, (1152, 2048), (1, 1152), 0), alpha=1, beta=1, out=buf591)
            del arg439_1
            del arg440_1
            del buf590
            buf592 = reinterpret_tensor(buf591, (1, 256, 2048), (524288, 2048, 1), 0); del buf591  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_189, features], Original ATen: [aten.view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_view_11.run(buf592, 524288, stream=stream0)
        return (buf592, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 3, 224, 224), (3, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1152, 3, 14, 14), (588, 196, 14, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.int64)
    arg4_1 = rand_strided((256, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg26_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg38_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg42_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg45_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg46_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg49_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg50_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg53_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg54_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg56_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg57_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg58_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg59_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg60_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg61_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg62_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg63_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg64_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg65_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg66_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg67_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg68_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg69_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg76_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg77_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg78_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg79_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg80_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg81_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg82_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg83_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg84_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg85_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg86_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg87_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg88_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg89_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg90_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg91_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg93_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg94_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg95_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg96_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg97_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg98_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg99_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg100_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg101_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg102_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg103_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg104_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg105_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg106_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg107_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg108_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg109_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg110_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg111_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg112_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg113_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg114_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg115_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg116_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg117_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg118_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg119_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg120_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg121_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg122_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg123_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg124_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg125_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg126_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg127_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg128_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg129_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg130_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg131_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg132_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg133_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg134_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg135_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg136_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg137_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg138_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg139_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg140_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg141_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg142_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg143_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg144_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg145_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg146_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg147_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg148_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg149_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg150_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg151_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg152_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg153_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg154_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg155_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg156_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg157_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg158_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg159_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg160_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg161_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg162_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg163_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg164_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg165_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg166_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg167_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg168_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg169_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg170_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg171_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg172_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg173_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg174_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg175_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg176_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg177_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg178_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg179_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg180_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg181_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg182_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg183_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg184_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg185_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg186_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg187_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg188_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg189_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg190_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg191_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg192_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg193_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg194_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg195_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg196_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg197_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg198_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg199_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg200_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg201_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg202_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg203_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg204_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg205_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg206_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg207_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg208_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg209_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg210_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg211_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg212_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg213_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg214_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg215_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg216_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg217_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg218_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg219_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg220_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg221_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg222_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg223_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg224_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg225_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg226_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg227_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg228_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg229_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg230_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg231_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg232_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg233_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg234_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg235_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg236_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg237_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg238_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg239_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg240_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg241_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg242_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg243_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg244_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg245_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg246_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg247_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg248_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg249_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg250_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg251_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg252_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg253_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg254_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg255_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg256_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg257_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg258_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg259_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg260_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg261_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg262_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg263_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg264_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg265_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg266_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg267_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg268_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg269_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg270_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg271_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg272_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg273_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg274_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg275_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg276_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg277_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg278_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg279_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg280_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg281_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg282_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg283_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg284_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg285_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg286_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg287_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg288_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg289_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg290_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg291_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg292_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg293_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg294_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg295_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg296_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg297_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg298_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg299_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg300_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg301_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg302_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg303_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg304_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg305_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg306_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg307_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg308_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg309_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg310_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg311_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg312_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg313_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg314_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg315_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg316_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg317_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg318_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg319_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg320_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg321_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg322_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg323_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg324_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg325_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg326_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg327_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg328_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg329_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg330_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg331_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg332_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg333_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg334_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg335_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg336_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg337_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg338_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg339_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg340_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg341_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg342_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg343_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg344_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg345_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg346_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg347_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg348_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg349_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg350_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg351_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg352_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg353_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg354_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg355_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg356_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg357_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg358_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg359_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg360_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg361_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg362_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg363_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg364_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg365_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg366_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg367_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg368_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg369_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg370_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg371_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg372_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg373_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg374_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg375_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg376_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg377_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg378_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg379_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg380_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg381_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg382_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg383_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg384_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg385_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg386_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg387_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg388_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg389_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg390_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg391_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg392_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg393_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg394_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg395_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg396_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg397_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg398_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg399_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg400_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg401_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg402_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg403_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg404_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg405_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg406_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg407_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg408_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg409_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg410_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg411_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg412_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg413_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg414_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg415_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg416_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg417_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg418_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg419_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg420_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg421_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg422_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg423_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg424_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg425_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg426_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg427_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg428_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg429_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg430_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg431_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg432_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg433_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg434_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg435_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg436_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg437_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg438_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg439_1 = rand_strided((2048, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg440_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
