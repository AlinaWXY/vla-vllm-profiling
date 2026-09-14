"""Coarse L20 operators, with audited semantic reuse and utilization evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from profiling.experiments import file_record, figure_record, timestamp

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
