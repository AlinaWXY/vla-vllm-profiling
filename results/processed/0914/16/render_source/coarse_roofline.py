"""Coarse L20 operators, with audited semantic reuse and utilization evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from profiling.experiments import file_record, figure_record, timestamp
from profiling.full_roofline import csv_write
from profiling.roofline import analyze, attach_operators, load_ncu, metric_value

MAIN = ('QKV', 'QK', 'PV', 'O projection', 'FFN up', 'FFN down', 'Vision attention', 'Attention (fused)')
PCTS = {
    'dram_util_pct': 'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'tensor_active_pct': 'sm__pipe_tensor_cycles_active_v2.avg.pct_of_peak_sustained_elapsed',
    'sm_util_pct': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'l2_util_pct': 'lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'occupancy_pct': 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'issue_active_pct': 'smsp__issue_active.avg.pct_of_peak_sustained_active',
    'bf16_counter_peak_pct': 'sm__ops_path_tensor_src_bf16_dst_fp32.sum.pct_of_peak_sustained_elapsed',
}
SPEC_URL = 'https://www.hpe.com/es/es/collaterals/collateral.c04123180.html'
GUIDE_URL = 'https://docs.nvidia.com/nsight-compute/ProfilingGuide/#roofline-charts'
NOMINAL_TF = 119.0
NOMINAL_GBS = 864.0


def family(label):
    if 'FFN' in label and ('gate/up' in label or 'up +' in label or 'GELU product' in label):
        return 'FFN up'
    if 'FFN down' in label or 'FFN: down' in label:
        return 'FFN down'
    if 'QKᵀ' in label:
        return 'QK'
    if 'P × V' in label:
        return 'PV'
    if 'QKV projection' in label or any(f'Vision: {x} projection' in label for x in 'QKV'):
        return 'QKV'
    if 'output' in label and ('Attention' in label or 'attention' in label):
        return 'O projection'
    if 'FlashAttention' in label:
        return 'Vision attention'
    if 'softmax' in label:
        return 'Softmax'
    if 'Norm' in label or 'AdaRMS' in label:
        return 'Norm / residual'
    if 'Action input' in label or 'Action head' in label:
        return 'Action IO'
    return 'Other'


def load_capture(cap, semantic):
    suite = json.loads((cap/'suite.json').read_text())
    if suite['state'] != 'completed':
        raise ValueError('Capture has not completed all replay audits')
    report = json.loads((cap/'workload/full_scopes.json').read_text())
    reference = json.loads((semantic/'summary.json').read_text())
    oldcap = Path(reference['capture'])
    original = json.loads((oldcap/'workload/full_scopes.json').read_text())
    for key in ('input_audit', 'model_parameter_fingerprint', 'operator_manifest', 'framework'):
        if report[key] != original[key]:
            raise ValueError(f'Semantic source workload changed: {key}')
    oldchoices = json.loads((oldcap/'workload/preprofile.json').read_text())['compiled_launch_choices']
    choices = json.loads((cap/'workload/preprofile.json').read_text())['compiled_launch_choices']
    if choices != oldchoices:
        raise ValueError('Frozen launch configurations differ from semantic source')
    import numpy as np
    with np.load(oldcap/'workload/actions.npz') as old, np.load(cap/'workload/actions.npz') as new:
        for key in ('noise', 'optimized', 'scoped', 'scoped_graph'):
            if not np.array_equal(old[key], new[key]):
                raise ValueError(f'Actions changed from semantic source: {key}')
    kernels = load_ncu(cap/'ncu/raw.csv')
    attach_operators(kernels, load_ncu(cap/'ncu/operators.csv'))
    with (semantic/'kernels.csv').open() as f:
        labels = [r for r in csv.DictReader(f) if r['domain'] == 'bf16_tensor']
    if len(kernels) != len(labels):
        raise ValueError('Kernel count differs from audited semantic source')
    for k, ref in zip(kernels, labels):
        for key in ('id', 'kernel', 'grid', 'block'):
            if str(k[key]) != ref[key]:
                raise ValueError(f'Kernel identity changed: {k["id"]} {key}')
        if k['operator'] != ref['nvtx_export']:
            raise ValueError(f'NVTX sequence changed: {k["id"]}')
        for key in ('stage', 'semantic_label', 'nvtx', 'tensor_shapes', 'source_line'):
            k[key] = ref[key]
        k['family'] = family(k['semantic_label'])
    contract = json.loads((cap/'metrics.json').read_text())
    for k in kernels:
        for name in contract['metrics']:
            if name not in k['metrics'] or k['metrics'][name][0] is None:
                raise ValueError(f'Missing counter {k["id"]}: {name}')
    domains = analyze(kernels, contract)
    domain_map = {(r['id'], r['domain']): r for r in domains}
    rows = []
    for k in kernels:
        bf = domain_map[k['id'], 'bf16_tensor']
        fp = domain_map[k['id'], 'fp32_simt']
        r = {key: k[key] for key in ('id', 'stage', 'family', 'semantic_label', 'kernel', 'nvtx')}
        r.update(duration_ns=bf['duration_ns'], dram_bytes=bf['memory_bytes'],
                 bf16_flops=bf['flops'], fp32_flops=fp['flops'],
                 l2_bytes=metric_value(k, 'lts__t_bytes.sum', 'bytes'),
                 ncu_bf16_peak_ops=metric_value(k, 'sm__ops_path_tensor_src_bf16_dst_fp32.sum.peak_sustained_elapsed', 'count'),
                 ncu_dram_peak_bytes=metric_value(k, 'dram__bytes_read.sum.peak_sustained_elapsed', 'bytes'))
        for short, metric in PCTS.items():
            value, unit = k['metrics'][metric]
            if unit != '%':
                raise ValueError(f'Unexpected utilization unit: {metric} {unit}')
            r[short] = value
        rows.append(r)
    return rows, suite, report, reference


def aggregate(rows, ceilings):
    groups = defaultdict(list)
    for r in rows:
        groups[r['stage'], r['family']].append(r)
    result = []
    for (stage, op), members in groups.items():
        r = {'stage': stage, 'operator': op, 'kernel_calls': len(members)}
        for key in ('duration_ns', 'dram_bytes', 'bf16_flops', 'fp32_flops', 'l2_bytes',
                    'ncu_bf16_peak_ops', 'ncu_dram_peak_bytes'):
            r[key] = sum(m[key] for m in members) if all(m.get(key) is not None for m in members) else None
        for key in PCTS:
            r[key] = (sum(m[key]*m['duration_ns'] for m in members)/r['duration_ns']
                      if all(m.get(key) is not None for m in members) else None)
        r.update(ncu_time_ms=r['duration_ns']/1e6,
                 dram_gbps=r['dram_bytes']/r['duration_ns'],
                 bf16_tflops=r['bf16_flops']/r['duration_ns']/1e3,
                 fp32_tflops=r['fp32_flops']/r['duration_ns']/1e3,
                 bf16_ai=r['bf16_flops']/r['dram_bytes'],
                 fp32_ai=r['fp32_flops']/r['dram_bytes'])
        r['nominal_compute_min_ms'] = r['bf16_flops']/NOMINAL_TF/1e9
        r['nominal_dram_min_ms'] = r['dram_bytes']/NOMINAL_GBS/1e6
        r['roofline_side'] = 'compute' if r['bf16_ai'] >= NOMINAL_TF*1000/NOMINAL_GBS else 'memory'
        empirical_ridge = ceilings['compute_tflops']['bf16_tensor']*1000/ceilings['dram_bandwidth_gbps']
        r['empirical_roofline_side'] = 'compute' if r['bf16_ai'] >= empirical_ridge else 'memory'
        r['nominal_roof_pct'] = 100*r['bf16_tflops']/min(NOMINAL_TF, r['bf16_ai']*NOMINAL_GBS/1000) if r['bf16_flops'] else 0
        r['diagnosis'] = 'not classified on BF16 roof'
        if op in MAIN:
            # An explicit reporting heuristic, not a causal sensitivity test.
            if r['roofline_side'] == 'compute' and r['tensor_active_pct'] is not None and r['tensor_active_pct'] >= 75:
                r['diagnosis'] = 'compute-bound supported'
            elif r['roofline_side'] == 'memory' and r['dram_util_pct'] is not None and r['dram_util_pct'] >= 70:
                r['diagnosis'] = 'DRAM-bound supported'
            else:
                r['diagnosis'] = r['roofline_side'] + '-side; saturation not established'
        result.append(r)
    for key in ('duration_ns', 'dram_bytes', 'bf16_flops', 'fp32_flops', 'l2_bytes'):
        if not all(r.get(key) is not None for r in rows):
            continue
        assert abs(sum(r[key] for r in result)-sum(r[key] for r in rows)) <= max(1, sum(r[key] for r in rows)*1e-12), key
    order = {name:i for i,name in enumerate((*MAIN, 'Softmax', 'Norm / residual', 'Action IO', 'Other'))}
    return sorted(result, key=lambda r: (r['stage'] != 'vlm', order[r['operator']]))


def plot(rows, out, ceilings, inputs, collected, *, caption=None, footnote=None,
         xbounds=(10,2048), ybounds=(4,180), label_offsets=None, region_label_y=6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    stamp = figure_record(out/'major_operators', inputs)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 7.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.98, bottom=.26, top=.79, wspace=.12)
    colors = {'compute': '#c46716', 'memory': '#147baf'}
    ridge = NOMINAL_TF*1000/NOMINAL_GBS
    offsets = {
        'vlm': {'QKV':(3,-26), 'QK':(-15,-25), 'PV':(-38,12),
                'O projection':(10,-12), 'FFN up':(-43,16), 'FFN down':(12,12), 'Vision attention':(-5,-27)},
        'action_expert': {'QKV':(9,-17), 'QK':(12,8), 'PV':(10,-22),
                          'O projection':(-5,-26), 'FFN up':(9,15), 'FFN down':(25,-6)}}
    if label_offsets:
        for stage, values in label_offsets.items():
            offsets[stage].update(values)
    names = {'QK':r'$QK^{T}$', 'PV':r'$PV$', 'Vision attention':'Vision attention\n(fused)',
             'Attention (fused)':'Attention\n(QK + softmax + PV)'}
    xs = np.geomspace(*xbounds, 300)
    for ax, stage, title in zip(axes, ('vlm','action_expert'), ('VLM','Action Expert')):
        ax.set_xscale('log', base=10); ax.set_yscale('log', base=10)
        ax.set_xlim(*xbounds); ax.set_ylim(*ybounds)
        ax.axvspan(xbounds[0], ridge, color=colors['memory'], alpha=.055)
        ax.axvspan(ridge,xbounds[1],color=colors['compute'],alpha=.055)
        ax.plot(xs,np.minimum(NOMINAL_TF,xs*NOMINAL_GBS/1000),color='#252b32',lw=2,label='Nominal L20 roof')
        ax.plot(xs,np.minimum(ceilings['compute_tflops']['bf16_tensor'],xs*ceilings['dram_bandwidth_gbps']/1000),
                color='#717b85',lw=1.4,ls='--',label='Measured GEMM / copy reference')
        ax.axvline(ridge,color='#8b939b',lw=.8,ls=':')
        ax.text(65,region_label_y,'Memory side',ha='center',color=colors['memory'])
        ax.text(390,region_label_y,'Compute side',ha='center',color=colors['compute'])
        for r in rows:
            if r['stage'] != stage or r['operator'] not in MAIN:
                continue
            supported = r['diagnosis'].endswith('supported')
            color = colors[r['roofline_side']]
            ax.scatter(r['bf16_ai'],r['bf16_tflops'],s=115 if r['operator'].startswith('FFN') else 76,
                       facecolors=color if supported else 'white',edgecolors=color,linewidths=1.8,zorder=4)
            ax.annotate(names.get(r['operator'],r['operator']), (r['bf16_ai'],r['bf16_tflops']),
                        xytext=offsets[stage].get(r['operator'],(10,12)),textcoords='offset points',fontsize=10,
                        color=color,fontweight='bold' if r['operator'].startswith('FFN') else 'normal',
                        arrowprops={'arrowstyle':'-','color':color,'lw':.7},zorder=5)
        ax.set_title(title,fontsize=15,fontweight='bold',pad=15)
        ax.set_xlabel('DRAM arithmetic intensity (BF16 FLOP / byte)',labelpad=10)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, subs=(1,)))
            axis.set_major_formatter(FuncFormatter(
                lambda value, position: r'$10$' if np.isclose(value, 10)
                else rf'$10^{{{int(round(np.log10(value)))}}}$'))
            axis.set_minor_locator(LogLocator(base=10, subs=range(2,10)))
            axis.set_minor_formatter(NullFormatter())
        ax.grid(which='major',alpha=.16)
        for side in ('top','right'): ax.spines[side].set_visible(False)
    axes[0].set_ylabel('Achieved BF16 Tensor throughput (TFLOP/s)')
    axes[0].legend(loc='upper left',fontsize=9,frameon=False)
    fig.suptitle('L20 DRAM Roofline — major operators',fontsize=20,fontweight='bold',y=.97)
    fig.text(.5,.91,caption or 'π0.5 / Omni 6bdbf97  |  batch 1  |  3 valid cameras  |  prefix 918  |  10 denoising steps',ha='center',fontsize=11)
    fig.text(.5,.865,f'Nominal: 119 TFLOP/s BF16, 864 GB/s DRAM  |  ridge = {ridge:.1f} FLOP/byte',ha='center',fontsize=11)
    fig.text(.07,.125,'Filled: utilization supports the bottleneck.  Hollow: roofline side only; saturation is not established.',fontsize=10)
    fig.text(.07,.09,footnote or 'FFN up includes gate / activation; down includes residual. VLM QK / PV: prefix only; vision attention is fused.',fontsize=9.5)
    fig.text(.07,.055,f'Collection: {collected}  |  NCU kernel times are not end-to-end latency.',fontsize=8.5,color='#59616a')
    fig.text(.07,.025,stamp,fontsize=8.5,color='#59616a')
    for ext in ('png','pdf'): fig.savefig(out/f'major_operators.{ext}',dpi=190)
    plt.close(fig)
    return stamp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--semantic-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); cap=a.capture.resolve(); semantic=a.semantic_source.resolve(); out=a.output.resolve()
    if (out/'summary.json').exists(): raise FileExistsError('Allocate a new experiment')
    rows,suite,report,reference=load_capture(cap,semantic)
    ceilings=json.loads((cap/'ceilings.json').read_text())
    collection=json.loads((cap/'ncu/collection.json').read_text())
    grouped=aggregate(rows,ceilings)
    csv_write(out/'kernels.csv',rows); csv_write(out/'operators.csv',grouped)
    csv_write(out/'mapping.csv',[{'stage':s,'semantic_label':label,'coarse_operator':family(label),'kernel_calls':n}
        for (s,label),n in sorted(Counter((r['stage'],r['semantic_label']) for r in rows).items())])
    inputs=[cap/'ncu/raw.csv',cap/'ncu/operators.csv',cap/'ncu/collection.json',cap/'workload/full_scopes.json',
            cap/'suite.json',cap/'source.json',cap/'ceilings.json',cap/'metrics.json',
            cap/'compute_ncu/raw.csv',cap/'compute_validation.json',semantic/'kernels.csv',semantic/'summary.json',Path(__file__).resolve()]
    stamp=plot(grouped,out,ceilings,inputs,collection['started_at']+' — '+collection['finished_at'])
    validation = next(k for k in load_ncu(cap/'compute_ncu/raw.csv')
        if metric_value(k,'sm__ops_path_tensor_src_bf16_dst_fp32.sum','count') > 0)
    vf = metric_value(validation,'sm__ops_path_tensor_src_bf16_dst_fp32.sum','count')
    assert vf == 2*8192**3
    counter_validation = {'known_bf16_flops':vf,
        'bf16_tflops':vf/metric_value(validation,'gpu__time_duration.sum','time')/1e12,
        'tensor_active_pct':validation['metrics'][PCTS['tensor_active_pct']][0],
        'raw_bf16_counter_peak_pct':validation['metrics'][PCTS['bf16_counter_peak_pct']][0]}
    summary={'analyzed_at':timestamp(),'capture':str(cap),'semantic_source':str(semantic),
        'slurm_job_id':suite['slurm_job_id'],'collection_started_at':collection['started_at'],
        'collection_finished_at':collection['finished_at'],'plot_stamp':stamp,
        'framework':report['framework'],'input_audit':report['input_audit'],'operators':grouped,
        'counter_validation':counter_validation,
        'empirical_ceilings':ceilings,'nominal_ceilings':{'bf16_tflops':NOMINAL_TF,'dram_gbps':NOMINAL_GBS,'source':SPEC_URL},
        'verification':{'exact_semantic_source_launch_sequence_and_geometry':True,'identical_inputs_model_actions_launch_choices':True,
            'all_requested_metrics_present':True,'aggregation_conserves_flops_bytes_time':True,
            'main_classification_agrees_for_nominal_and_empirical_roofs':all(r['roofline_side']==r['empirical_roofline_side'] for r in grouped if r['operator'] in MAIN),
            'kernel_count':len(rows),'stages':dict(Counter(r['stage'] for r in rows)),
            'cross_pass_audits':suite['application_passes'],'baseline_vs_profile_max_abs':suite['baseline_vs_profile_max_abs']},
        'inputs':[file_record(f) for f in inputs]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    write_readme(out,summary,grouped)
    print(json.dumps({'output':str(out),'operators':grouped},indent=2))


def write_readme(out,s,rows):
    ff = {(r['stage'],r['operator']):r for r in rows if r['operator'] in ('FFN up','FFN down')}
    vu,vd = ff['vlm','FFN up'],ff['vlm','FFN down']
    eu,ed = ff['action_expert','FFN up'],ff['action_expert','FFN down']
    lines=[f'# L20 大算子 DRAM Roofline — {out.parent.name}/{out.name}', '',
        f'采集：{s["collection_started_at"]} 至 {s["collection_finished_at"]}（Asia/Hong_Kong）；Slurm {s["slurm_job_id"]}。',
        f'绘图：{s["plot_stamp"]}。新采集目录：{s["capture"]}；0914/06 仅提供已审计语义映射，不复用其测量值。', '',
        'π0.5 / Omni 6bdbf97，真实 pi05_base 权重，BF16，batch=1，3 路有效合成相机输入，prefix=918，10 步去噪。',
        '![Major operators](major_operators.png)', '',
        f'**FFN up/down 分开后：VLM 两者均有 compute-bound 证据**，Tensor active 分别为 {vu["tensor_active_pct"]:.1f}% / {vd["tensor_active_pct"]:.1f}%；**Action Expert 两者均有 DRAM memory-bound 证据**，DRAM 利用率为 {eu["dram_util_pct"]:.1f}% / {ed["dram_util_pct"]:.1f}%。',
        'QK、PV 等其余点需要区分模型分类和饱和证据。例如 Expert QK 位于 compute side，但 Tensor active 仅约 30%；Expert PV 位于 memory side，但 DRAM 利用率约 24%。本次数据不支持将这些点宣称为已证实的计算/DRAM 带宽饱和。', '',
        '## 算子边界', '',
        '- QKV：Q/K/V 投影，保留已融合的 RoPE/bias；O projection：attention 输出投影及已融合残差。',
        '- FFN up：gate/up 升维 + 激活/乘积，含独立的 prefix GELU kernel；FFN down：降维 + 已融合残差。up 与 down 分开。',
        '- VLM 中 QK 和 PV 指 prefix Transformer。视觉编码器使用 FlashAttention，QK、softmax、PV 与 split-KV combine 合在 Vision attention，无法独立分配其实测 DRAM/时间。',
        '- VLM 的投影/FFN 合并视觉与 prefix 的同类操作；重复层、相机、去噪步聚合，保留每个成员的 mapping.csv。', '',
        'FFN 的规模差异：Expert 的升维输入为 [50,1024]、gate/up 权重各为 [1024,4096]，down 输入为 [50,4096]；VLM prefix 为 918 tokens，视觉每路为 256 tokens。较小 token 数降低权重复用，这是 Expert FFN 算术强度更低的结构性原因。实际 AI 仍用硬件 FLOP/DRAM 计数计算，包含实现中的 padding/recomputation。', '',
        '## 证据与判定', '',
        'AI = ΣBF16 FLOPs / ΣDRAM bytes；性能 = ΣBF16 FLOPs / ΣNCU kernel time；每个 kernel 的时间/流量只算一次。',
        f'标称 roof 使用 BF16 {NOMINAL_TF:g} TFLOP/s、DRAM {NOMINAL_GBS:g} GB/s，拐点 {NOMINAL_TF*1000/NOMINAL_GBS:.2f} FLOP/byte。[HPE L20 规格]({SPEC_URL})',
        f'虚线是本次作业内的实际 GEMM / copy 参考：BF16 {s["empirical_ceilings"]["compute_tflops"]["bf16_tensor"]:.2f} TFLOP/s，DRAM {s["empirical_ceilings"]["dram_bandwidth_gbps"]:.2f} GB/s；经验参考不是硬上限。',
        'roofline side 由两项理想下界比较：t_compute = F/P_peak；t_DRAM = B/BW_peak。此分类描述该模型中的较紧约束；低利用率时不等于硬件已饱和。',
        '实心点要求：compute side 且 Tensor 管线 active ≥75%，或 memory side 且 DRAM utilization ≥70%。这是明确的报告筛选标准，不能代替频率/工作量扰动的因果实验。',
        '利用率是各 kernel 指标按 NCU 时间加权后的均值，不是整段并发运行的硬件利用率。SM throughput 可能包含非 Tensor 管线，L2 throughput 也不等同 DRAM 带宽。',
        f'计数器验证：8192³ dense BF16 GEMM 的计数严格等于 2×8192³ FLOPs，达到 {s["counter_validation"]["bf16_tflops"]:.2f} TFLOP/s；Tensor active {s["counter_validation"]["tensor_active_pct"]:.2f}%，而 BF16 ops 原始 peak 百分比只有 {s["counter_validation"]["raw_bf16_counter_peak_pct"]:.2f}%。后者的分母与 dense 路径上限不一致，本报告保留原始值但不把它当作 dense 计算利用率。',
        f'Roofline 与利用率解释参考 [NVIDIA Nsight Compute Profiling Guide]({GUIDE_URL})。', '',
        '| 模块 | 算子 | AI FLOP/B | TFLOP/s | DRAM GB/s | DRAM % | Tensor active % | L2 % | Roofline side | 证据判定 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        if r['operator'] not in MAIN: continue
        lines.append(f'| {r["stage"]} | {r["operator"]} | {r["bf16_ai"]:.1f} | {r["bf16_tflops"]:.2f} | {r["dram_gbps"]:.1f} | {r["dram_util_pct"]:.1f} | {r["tensor_active_pct"]:.1f} | {r["l2_util_pct"]:.1f} | {r["roofline_side"]} | {r["diagnosis"]} |')
    lines+=['','## 覆盖与复现','','所有测量值来自本次完整采集；与旧语义映射逐调用核对 kernel 名、顺序、NVTX、grid、block，模型/输入/动作及冻结 launch 配置均一致。',
        '附表保留 Softmax、Norm/residual、Action IO 和 Other；它们的 BF16、FP32 分开统计，不把归一化/softmax 的 FP32 操作放到 BF16 计算 roof 上。',
        '| 模块 | 总 kernel 数 | 主图 kernel 数 | 主图 NCU 时间覆盖 | 主图 BF16 FLOP 覆盖 |',
        '|---|---:|---:|---:|---:|']
    for stage in ('vlm','action_expert'):
        allrows=[r for r in rows if r['stage']==stage]; main=[r for r in allrows if r['operator'] in MAIN]
        lines.append(f'| {stage} | {sum(r["kernel_calls"] for r in allrows)} | {sum(r["kernel_calls"] for r in main)} | {100*sum(r["duration_ns"] for r in main)/sum(r["duration_ns"] for r in allrows):.2f}% | {100*sum(r["bf16_flops"] for r in main)/sum(r["bf16_flops"] for r in allrows):.3f}% |')
    lines+=['', '其余操作仅附表，避免主图过细；FP32 指 add/mul/2×FMA，未计超越函数和整数指令，不能据此单独证明 softmax/norm 的算力瓶颈。',
        '| 模块 | 附属操作 | NCU 合计 ms | BF16 GFLOP | FP32 GFLOP | DRAM GB/s | DRAM % | SM % |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['operator'] in MAIN: continue
        lines.append(f'| {r["stage"]} | {r["operator"]} | {r["ncu_time_ms"]:.3f} | {r["bf16_flops"]/1e9:.3f} | {r["fp32_flops"]/1e9:.3f} | {r["dram_gbps"]:.1f} | {r["dram_util_pct"]:.1f} | {r["sm_util_pct"]:.1f} |')
    lines+=['', 'NCU application replay、cache-control=all、未锁频，按 kernel 采集会改变缓存及调度条件；这里的流量/时间用于算子 roofline，不代表生产缓存状态。真实模块及 VLA 时长见采集目录 baseline/full_scopes.json，不能由 kernel 时间求和替代。',
        '范围含图像编码、文字 embedding、prefix Transformer、Expert 输入投影/10步解码/动作头及更新；预处理、tokenization、D2H 与静态 timestep/AdaRMS 条件准备在范围外。当前框架的既有数值差异不在本任务修复，不声称数值等价加速。',
        '', '[PDF](major_operators.pdf) · [全部大算子 CSV](operators.csv) · [逐 kernel 证据](kernels.csv) · [归组映射](mapping.csv) · [来源及校验](summary.json)', '']
    (out/'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
