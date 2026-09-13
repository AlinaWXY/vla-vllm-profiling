"""CPU-only FLOP and tensor-I/O estimates for the 37 audited VLM groups of 0913/30.

Logical work is separate from hardware counters (padding, compiler arithmetic
and transcendental implementations). Single-read/write tensor I/O is a model,
never a measurement of DRAM bytes or a cache-miss estimate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from profiling.experiments import file_record, figure_record, timestamp
from profiling.full_roofline import csv_write


DRAM_GBPS = 273.0


def models():
    L, D, H = 918, 2048, 16384
    S, V, F, A, d = 256, 1152, 4304, 16, 72
    E = S*V
    specs = {}

    def put(i, shape, tensor, scalar, formula, io=None, weight=0, special='', note=''):
        specs[i] = dict(shape=shape, logical_tensor_flops=tensor,
                        scalar_flops_estimate=scalar, formula=formula,
                        tensor_io_once_bytes=io, weight_bytes=weight,
                        special_ops=special, estimate_note=note)

    def gemm(i, m, k, n, residual=False, bias=False, scalar_extra=0, extra_bytes=0):
        put(i, f'[{m},{k}] @ [{k},{n}]', 2*m*k*n,
            m*n*(int(residual)+int(bias))+scalar_extra,
            f'2*{m}*{k}*{n} tensor; {m*n*(int(residual)+int(bias))+scalar_extra} scalar',
            2*(m*k+k*n+m*n)+2*m*n*int(residual)+2*n*int(bias)+extra_bytes,
            2*k*n, note='FMA=2; logical dimensions; excludes MMA tile padding and compiler epilogue overhead')

    put(1, f'gate/up [{L},{2*H}] -> [{L},{H}]', 0, 9*L*H,
        '9*L*H add/mul + L*H tanh', 6*L*H, special=f'{L*H} tanh')
    gemm(2, L, H, D, residual=True)
    gemm(3, L, D, 2*H)
    # Prefix has 8 Q heads and one shared KV head, all of width 256.
    put(4, '[7344,918] @ [918,256]', 2*8*L*L*256, 0, '2*(8*L)*L*256',
        2*(8*L*L+L*256+8*L*256))
    put(5, '[7344,256] @ [256,918]', 2*8*L*256*L, 8*L*L,
        '2*(8*L)*256*L tensor + 8*L*L scale', 2*(8*L*256+L*256)+4*8*L*L)
    gemm(6, L, D, D, residual=True)
    put(7, f'positions [{L}], half-dim 128', 0, 3*L*128,
        '3*L*128 mul before folding unit attention_scaling',
        8*L+4*128+2*2*L*256, special=f'{L*128} sin + {L*128} cos',
        note='Unit scaling may be folded; trig implementation arithmetic is excluded')
    put(8, '[7344,918] softmax', 0, 3*8*L*L-8*L,
        'R*(3*C-1), R=8*L, C=L: subtract + reduction + reciprocal multiply',
        6*8*L*L+4*L, special=f'{8*L*L} exp + {8*L} reciprocal + max comparisons',
        note='Logical reduction, excludes padded 1024-key lanes and transcendental lowering')
    gemm(9, L, D, 2560, scalar_extra=3*L*9*256, extra_bytes=2*L*256)
    for i in (10, 11):
        put(i, f'[{L},{D}]', 0, L*(5*D+1),
            'L*(5*D+1): x^2, sum, mean, epsilon, (1+w), two output multiplies',
            4*L*D+2*D, 2*D, special=f'{L} rsqrt',
            note='Logical reduction; source rereads x in its second pass, which can hit cache')
    put(12, '[1,968,2048] concatenate, then view first 918', 0, 0,
        '0 (copy)', 4*968*D,
        note='Three image sequences plus 200 padded text positions; prefix encoder uses 918 valid positions')
    put(13, '[918] mask conversion', 0, 0, '0 (integer/copy)')
    put(14, 'one valid-length integer', 0, 0, '0 (integer fill)')
    put(15, '[200] -> [200,2048] embedding', 0, 0, '0 (gather)',
        4*200*D+8*200, note='Includes padded text positions; repeated token IDs may reuse weights')
    put(16, '[200,2048] scale', 0, 200*D, '200*D mul', 4*200*D)
    put(17, f'[{S},{F}] GELU', 0, 8*S*F, '8*S*F add/mul + S*F tanh',
        4*S*F, special=f'{S*F} tanh')
    gemm(18, S, F, V, bias=True)
    gemm(19, S, V, F, bias=True)
    put(20, '[1,16,256,72] Q/K/V, full attention', 4*A*S*S*d,
        4*A*S*S-A*S, '4*heads*S^2*d tensor; ~heads*S*(4*S-1) scalar',
        8*A*S*d, special=f'{A*S*S} exp + {A*S} reciprocal + max comparisons',
        note='Algorithmic full attention estimate, no materialized score matrix; online-softmax rescaling/reductions add work. Kernel trait pads d=72 to 96.')
    for i, shape, count in ((21, '[1,3,224,224]', 3*224*224),
                            (22, '[1152,3,14,14]', V*3*14*14),
                            (23, '[1,1152,16,16]', E)):
        put(i, shape, 0, 0, '0 (layout copy)', 4*count,
            note='NCU counts one FP32 operation/element in the cuDNN conversion kernel; no useful arithmetic is required by layout conversion')
    gemm(24, S, V, V, bias=True)
    put(25, f'[{S},{D}] scale', 0, S*D, 'S*D mul', 4*S*D)
    gemm(26, S, V, D, bias=True)
    put(27, '[1,3,224,224] F32 -> BF16', 0, 0, '0 (cast/layout)', 6*3*224*224)
    put(28, '[1,3,224,224], 1152 filters 14x14, stride14', 2*S*V*3*14*14, 0,
        '2*256*1152*(3*14*14)', 2*(3*224*224+V*3*14*14+E), 2*V*3*14*14,
        note='Convolution bias is deferred into the patch/LayerNorm fusion')
    # Welford combines two (mean,m2,count) triples: approx 9 add/mul + one
    # division per merge. Real compiler reductions/padding are counted by NCU.
    put(29, '[256,1152] split into 2304 groups of 128', 0, 2*E+9*2304*127,
        '2*S*V + 9*2304*(128-1)', special='~2304*127 Welford divisions',
        note='Approximate source work: bias+position additions and partial Welford moments')
    put(30, '256 rows x 9 partial Welford triples', 0, 9*S*8,
        '9*256*(9-1)', special='~256*8 Welford divisions',
        note='Logical merges; padded reduction lanes and replicated lane work excluded')
    put(31, '[256,1152] patch/LayerNorm normalize', 0, 8*E,
        '8*S*V including broadcast variance scaling', special=f'up to {E} rsqrt before hoisting',
        note='Source elementwise count, including recomputation of bias/position additions')
    # Generated kernels sometimes recompute the residual sum in a second pass.
    for i, additions, variant in ((32, 2, 10), (33, 3, 4), (34, 2, 6),
                                  (35, 4, 7), (36, 6, 8), (37, 4, 9)):
        put(i, f'[256,1152], LayerNorm fusion {variant}', 0,
            (additions+4)*E+9*S*(V-1)+2*S,
            f'({additions}+4)*S*V + 9*S*(V-1) + 2*S',
            special=f'~{S*(V-1)} Welford divisions + {S} rsqrt',
            note='Approximate source arithmetic, logical Welford merges; repeated residual sums included; compiler lowering may differ')
    assert set(specs) == set(range(1, 38))
    return specs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('results/processed/0913/30/vlm'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require_empty = args.output/'operator_costs.csv'
    if require_empty.exists():
        raise FileExistsError('Allocate a new dated experiment')
    measured = list(csv.DictReader((args.input/'operators.csv').open()))
    legends = {int(r['group_id']):r for r in csv.DictReader((args.input/'operator_legend.csv').open())}
    if not (len(legends) == 37 and legends[3]['label'] == 'Prefix FFN: gate/up projection'
            and legends[20]['label'] == 'Vision: FlashAttention'):
        raise ValueError('Wrong group map')
    rows = []
    total_ms = sum(float(r['duration_ns']) for r in measured if r['domain']=='bf16_tensor')/1e6
    for ident, spec in models().items():
        by_domain = {r['domain']:r for r in measured if int(r['group_id']) == ident}
        t, f = by_domain['bf16_tensor'], by_domain['fp32_simt']
        count, ns = int(t['invocations']), float(t['duration_ns'])/int(t['invocations'])
        tensor, scalar = float(t['flops'])/count, float(f['flops'])/count
        traffic = float(t['memory_bytes'])/count
        logical = spec['logical_tensor_flops']+spec['scalar_flops_estimate']
        io = spec['tensor_io_once_bytes']
        row = {'id':ident, 'label':legends[ident]['label'], 'calls':count, **spec,
            'logical_flops_per_call':logical, 'logical_gflop_per_call':logical/1e9,
            'ncu_bf16_gflop_per_call':tensor/1e9, 'ncu_fp32_gflop_per_call':scalar/1e9,
            'mean_ncu_us':ns/1e3, 'total_ncu_ms':ns*count/1e6,
            'time_percent':ns*count/1e6/total_ms*100,
            'logical_tensor_tflops':spec['logical_tensor_flops']/ns/1000,
            'ncu_tensor_tflops':tensor/ns/1000, 'ncu_scalar_tflops':scalar/ns/1000,
            'tensor_counter_over_logical':tensor/spec['logical_tensor_flops'] if spec['logical_tensor_flops'] else None,
            'ncu_l2_bytes_per_call':traffic, 'ncu_l2_gbps':traffic/ns,
            'ncu_tensor_ai_l2':tensor/traffic if traffic else None,
            'logical_tensor_ai_io_once':spec['logical_tensor_flops']/io if io else None,
            'io_once_equivalent_gbps':io/ns if io else None,
            'io_once_time_at_273gbps_us':io/DRAM_GBPS/1000 if io else None,
            'l2_bytes_over_io_once':traffic/io if io else None,
            'dram_crossover_tensor_tflops':DRAM_GBPS*spec['logical_tensor_flops']/io/1000 if io else None,
            'operator':legends[ident]['operator']}
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_write(args.output/'operator_costs.csv', rows)
    tensor_total = sum(r['logical_tensor_flops']*r['calls'] for r in rows)
    scalar_total = sum(r['scalar_flops_estimate']*r['calls'] for r in rows)
    ncu_tensor = sum(r['ncu_bf16_gflop_per_call']*1e9*r['calls'] for r in rows)
    summary = {'created_at':timestamp(), 'scope':'CPU-only estimates from existing 0913/30 data',
        'dram_spec_gbps':DRAM_GBPS, 'logical_tensor_flops_total':tensor_total,
        'scalar_flops_estimate_total':scalar_total, 'ncu_tensor_flops_total':ncu_tensor,
        'ncu_tensor_over_logical':ncu_tensor/tensor_total, 'kernel_time_sum_ms':total_ms,
        'prefix_ffn_projections_time_percent':sum(r['time_percent'] for r in rows if r['id'] in (2,3)),
        'prefix_ffn_projections_tensor_flops_percent':sum(r['logical_tensor_flops']*r['calls'] for r in rows if r['id'] in (2,3))/tensor_total*100,
        'input_files':[file_record(args.input/n) for n in ('operators.csv','operator_legend.csv','summary.json')],
        'assumptions':['FMA=2; zero FLOPs for casts/indexing/copies as mathematical operators',
            'Scalar estimates exclude explicit sin/cos/exp/tanh/rsqrt/division costs listed separately',
            'Welford estimates use logical merges, not actual GPU reduction instruction counts',
            'One-read/write tensor I/O is not measured DRAM traffic; caches, rereads and deferred writeback can change it',
            'The 273 GB/s specification is shared LPDDR5X peak, not sustained bandwidth or L2 bandwidth',
            'NCU time sums are isolated cache-flushed kernel replay, not original pipeline latency']}
    # Explain the prefix QKV L2 traffic by counting explicit per-CTA loads.
    qkv = next(r for r in rows if r['id']==9)
    tiles, heads, L, D = 29, 10, 918, 2048
    tiled_weights = tiles*heads*D*256*2
    tiled_inputs = heads*L*D*2
    tiled_rope = 9*L*256*2
    outputs = L*2560*2
    summary['prefix_qkv_tile_model'] = {
        'rows_per_tile':32, 'row_tiles':tiles, 'head_tiles':heads,
        'weight_bytes_unique':D*2560*2, 'weight_bytes_repeated_per_row_tile':tiled_weights,
        'input_bytes_repeated_per_head':tiled_inputs, 'rope_bytes_repeated_per_rotated_head':tiled_rope,
        'output_bytes':outputs, 'explicit_tile_bytes':tiled_weights+tiled_inputs+tiled_rope+outputs,
        'measured_l2_bytes_per_call':qkv['ncu_l2_bytes_per_call'],
        'note':'Counts explicit source loads; L1 hits/sector rounding mean it is not an exact L2 prediction. Repeated L2 requests do not imply repeated DRAM fetches.'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    table = ['| ID | 算子 | 次数 | 逻辑 FLOPs/次 (GF) | NCU BF16 / FP32 (GF/次) | 平均耗时 μs | 耗时占比 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        table.append(f"| {r['id']} | {r['label']} | {r['calls']} | {r['logical_gflop_per_call']:.6g} | {r['ncu_bf16_gflop_per_call']:.6g} / {r['ncu_fp32_gflop_per_call']:.6g} | {r['mean_ncu_us']:.3f} | {r['time_percent']:.2f}% |")
    (args.output/'operator_table.md').write_text('\n'.join(table)+'\n')
    formulas = ['| ID | 形状 | FLOPs 估算公式 | 单独列出的特殊运算 |', '| --- | --- | --- | --- |']
    for r in rows:
        formulas.append(f"| {r['id']} | `{r['shape']}` | `{r['formula']}` | {r['special_ops']} |")
    (args.output/'formulas.md').write_text('L=918, D=2048, H=16384; S=256, V=1152, F=4304。\n\n'+'\n'.join(formulas)+'\n')
    plot(rows, args.output, args.input)
    print(json.dumps({k:v for k,v in summary.items() if k!='input_files'},indent=2))


def plot(rows, output, input_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    inputs = [output/'operator_costs.csv', input_dir/'operators.csv', Path(__file__)]
    chosen = [r for r in rows if r['logical_tensor_flops']]
    chosen.sort(key=lambda r:r['total_ncu_ms'], reverse=True)
    stem = output/'tensor_flops_and_intensity'
    stamp = figure_record(stem, inputs)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    y = np.arange(len(chosen))
    names = [f"[{r['id']}] {r['label']}" for r in chosen]
    ax=axes[0]
    ax.barh(y-.17, [r['logical_tensor_flops']/1e9 for r in chosen], height=.32, label='Logical 2MNK / attention / convolution')
    ax.barh(y+.17, [r['ncu_bf16_gflop_per_call'] for r in chosen], height=.32, label='NCU BF16 Tensor counter')
    for i,r in enumerate(chosen):
        if r['tensor_counter_over_logical'] > 1.1:
            ax.text(r['ncu_bf16_gflop_per_call']*1.07,i+.17,
                    f"{r['tensor_counter_over_logical']:.2f}×",fontsize=8,va='center')
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8);ax.invert_yaxis()
    ax.set_ylim(len(chosen)+1.3,-.75)
    ax.set(xscale='log',xlabel='GFLOP per invocation',title='Useful Tensor work vs executed Tensor work')
    ax.legend(fontsize=8);ax.grid(axis='x',alpha=.2)
    ax=axes[1]
    for i,r in enumerate(chosen):
        vals=[r['ncu_tensor_ai_l2'],r['logical_tensor_ai_io_once']]
        ax.plot(vals,[i,i],color='#adb5bd',lw=1)
    ax.scatter([r['ncu_tensor_ai_l2'] for r in chosen],y,label='NCU counted FLOPs / measured L2 bytes',color='#086788',s=30)
    ax.scatter([r['logical_tensor_ai_io_once'] for r in chosen],y,label='Logical FLOPs / one-read/write tensor I/O',color='#e8871e',s=30)
    ax.set_yticks(y);ax.set_yticklabels([f"[{r['id']}]" for r in chosen]);ax.invert_yaxis()
    ax.set_ylim(len(chosen)+1.3,-.75)
    ax.set(xscale='log',xlabel='FLOP/byte',title='Memory-level choice changes arithmetic intensity')
    ax.legend(fontsize=8,loc='lower right');ax.grid(axis='x',alpha=.2)
    fig.suptitle('π0.5 VLM, batch=1: 3 cameras × 256 tokens; prefix length 918',fontsize=14)
    fig.text(.5,.045,'Right-panel orange: tensor-I/O model, NOT measured DRAM traffic. Neither panel assumes a BF16 hardware ceiling.',ha='center',fontsize=9)
    fig.text(.5,.017,stamp,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.96))
    for ext in ('png','pdf'):fig.savefig(str(stem)+'.'+ext,dpi=180)
    plt.close(fig)
    chosen=[r for r in rows if r['id'] in (1,2,3,4,5,6,9,17,18,19,20,24,26,28)]
    chosen.sort(key=lambda r:r['total_ncu_ms'],reverse=True)
    stem=output/'bandwidth_scenarios';stamp=figure_record(stem,inputs)
    fig,axes=plt.subplots(1,2,figsize=(17,7));y=np.arange(len(chosen))
    for ax,field,title,reference in ((axes[0],'io_once_equivalent_gbps','If every tensor read/write reached DRAM',273),
                                     (axes[1],'ncu_l2_gbps','Actual L2 byte counter / NCU kernel time',954.7155332067927)):
        ax.barh(y,[r[field] for r in chosen],color='#e8871e' if ax is axes[0] else '#086788')
        ax.axvline(reference,color='#943126',ls='--',label='273 GB/s official LPDDR5X peak' if ax is axes[0] else '954.7 GB/s earlier L2-copy reference')
        ax.set_yticks(y);ax.set_yticklabels([f"[{r['id']}] {r['label']}" if ax is axes[0] else f"[{r['id']}]" for r in chosen],fontsize=8)
        ax.invert_yaxis();ax.set_ylim(len(chosen)+.8,-.75)
        ax.set(xlabel='GB/s',title=title);ax.legend(fontsize=8,loc='lower right');ax.grid(axis='x',alpha=.2)
    fig.suptitle('VLM bandwidth: explicit DRAM scenario and measured L2 traffic',fontsize=14)
    fig.text(.5,.045,'Left: single-read/write tensor model; cache hits, rereads and deferred writeback are unknown. Right: L2 is not DRAM.',ha='center',fontsize=9)
    fig.text(.5,.017,stamp,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.96))
    for ext in ('png','pdf'):fig.savefig(str(stem)+'.'+ext,dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
