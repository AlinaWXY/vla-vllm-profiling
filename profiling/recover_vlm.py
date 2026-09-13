"""Recover the completed VLM prefix of cancelled capture 0913/28, using CPU only.

This audit is deliberately specific to Omni 6bdbf97 and its frozen harness.
It does not reconstruct a missing end-of-run action array or combine the
new VLM measurements with the historical Action Expert workload.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from profiling.experiments import file_record, timestamp
from profiling.full_roofline import csv_write, plot_ops, semantic
from profiling.roofline import load_ncu, attach_operators, analyze, aggregate_operators


COMMIT = "6bdbf97e357232b8aa3b6caf5a6c8b0fed713c25"
PREFIX_SITES = [("_rms_norm_kernel", 1429), ("_matmul_gemma_rope_qkv", 1437),
                ("_matmul_abt_scale", 1455), ("_softmax_mask_vector", 1467),
                ("_matmul_small", 1476), ("_matmul_small_res", 1487),
                ("_rms_norm_kernel", 1499)]
FFN = ["nvjet_sm110_tst_256x232_64x3_1x2_h_bz_NNT",
       "triton_poi_fused_gelu_mul_split_0",
       "nvjet_sm110_tst_256x184_64x3_2x1_v_badd_NNT"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recover(kernels, contract, source):
    """Reject an incomplete stage, missing metrics or a mismatched launch trace."""
    require(len({(k['process_id'], k['device']) for k in kernels}) == 1,
            "Expected one profiled process/device")
    require([int(k['id']) for k in kernels] == list(range(len(kernels))),
            "Report IDs are not contiguous from zero")
    for k in kernels:
        # NCU 2026.1 appends a demangled kernel name to the innermost NVTX range.
        k['nvtx_export'] = k['operator']
        k['nvtx'] = k['operator'].split('/', 1)[0]
    vlm = [k for k in kernels if k['nvtx'].startswith('vlm')]
    require(vlm and len(vlm) < len(kernels), "Need a measured transition into Action Expert")
    require([int(k['id']) for k in vlm] == list(range(len(vlm))),
            "VLM is not a contiguous complete prefix of the report")
    require(all(k['nvtx'].startswith('action_expert') for k in kernels[len(vlm):]),
            "Unexpected stage after VLM")
    missing = [(k['id'], n) for k in vlm for n in contract['metrics']
               if k['metrics'].get(n, (None,))[0] is None]
    require(not missing, f"Missing VLM counters: {missing[:10]}")

    cameras, sequences = [], []
    for camera in range(3):
        calls = [k for k in vlm if k['nvtx'] == f'vlm.vision.camera{camera}']
        require(len(calls) == 280, f"Camera {camera} launch count differs from the audited trace")
        require(sum('flash_fwd_kernel' in k['kernel'] for k in calls) == 27,
                f"Camera {camera} does not cover all 27 vision attention layers")
        require(calls[-1]['kernel'] == 'triton_poi_fused_mul_view_11',
                f"Camera {camera} final embedding scale is missing")
        sequences.append([(k['kernel'], k['grid'], k['block']) for k in calls])
        cameras.append({'camera':camera, 'calls':len(calls), 'attention_layers':27,
                        'first_id':calls[0]['id'], 'last_id':calls[-1]['id']})
    require(sequences[0] == sequences[1] == sequences[2],
            "Camera launch sequences/geometries differ")

    observed = []
    for k in vlm:
        m = re.fullmatch(r'vlm\.prefix\.step-1\.layer(-?\d+)\.(\w+)\.line(\d+)', k['nvtx'])
        if m:
            layer, name, line = int(m[1]), m[2], int(m[3])
            require(k['kernel'] == name, "NVTX/kernel name mismatch")
            require(source[line-1].strip().startswith(name+'['), "Source launch site changed")
            observed.append((layer, name, line))
            k['source_line'] = line
    expected = [(-1, '_build_gemma_rope_from_positions', 1414)]
    for layer in range(18):
        expected.extend((layer, name, line) for name, line in
                        (PREFIX_SITES if layer < 17 else PREFIX_SITES[:2]))
    require(observed == expected, "Prefix Triton launch order/coverage mismatch")
    for layer in range(17):
        calls = [k for k in vlm if k['nvtx'] == f'vlm.prefix.layer{layer:02d}.FFN']
        require([k['kernel'] for k in calls] == FFN, f"Prefix FFN layer {layer} is incomplete")
    require(len(vlm) == 1018, "Unexpected VLM launch count")
    require(Counter(k['nvtx'] for k in vlm)['vlm.text_embedding'] == 2,
            "Text embedding lookup/scale missing")
    require(Counter(k['nvtx'] for k in vlm)['vlm.concat'] == 1, "Prefix concatenation missing")
    return vlm, {'recovered_vlm_calls':len(vlm), 'first_id':vlm[0]['id'],
        'last_id':vlm[-1]['id'], 'first_expert_id':kernels[len(vlm)]['id'],
        'ignored_partial_expert_calls':len(kernels)-len(vlm), 'cameras':cameras,
        'camera_sequences_and_geometries_equal':True, 'prefix_triton_launches':len(observed),
        'prefix_layers':18, 'prefix_ffn_layers':17, 'prefix_ffn_launches':51,
        'final_prefix_layer_note':'The selected source computes only RMSNorm/QKV for the last KV-producing layer.',
        'all_six_counters_present':True,
        'coverage_evidence':'Contiguous IDs, transition into expert, 3 equal camera traces, exact prefix source order; no reconstructed runtime manifest.'}


def label_kernel(k, source):
    name, nvtx = k['kernel'], k['nvtx']
    if 'source_line' in k:
        return 'Prefix: '+semantic(name, k['source_line'], source)
    if '.FFN' in nvtx:
        return 'Prefix FFN: '+dict(zip(FFN, ['gate/up projection', 'GELU product', 'down + residual']))[name]
    if nvtx == 'vlm.text_embedding':
        return 'Text: '+('embedding lookup' if 'gather' in name else 'embedding scale')
    if nvtx == 'vlm.concat':
        return 'Prefix: concatenate image/text embeddings'
    if nvtx == 'vlm.prefix':
        return 'Prefix: '+('valid-length fill' if 'FillFunctor' in name else 'mask conversion/copy')
    require(nvtx.startswith('vlm.vision.camera'), f"Unknown NVTX scope: {nvtx}")
    known = {
        'nvjet_sm110_tst_128x128_64x6_1x2_h_bz_bias_TNT':'attention Q/K/V/O projections',
        'nvjet_sm110_tst_224x128_64x7_1x2_2cta_h_bz_bias_TNN':'FFN expansion + bias',
        'nvjet_sm110_tst_144x128_64x6_4x1_v_bz_bias_TNN':'FFN contraction + bias',
        'nvjet_sm110_tst_512x64_64x2_1x4_h_bz_bias_TNT':'image-to-text projection + bias',
        'triton_poi_fused__to_copy_convolution_0':'input cast/layout',
        'triton_poi_fused_gelu_view_5':'FFN GELU',
        'triton_poi_fused_mul_view_11':'image embedding scale',
    }
    if name in known:
        label = known[name]
    elif 'nchwToNhwc' in name:
        label = 'NCHW → NHWC ('+('input' if k['grid'] == '(1568, 1, 1)' else 'weights')+')'
    elif 'nhwcToNchw' in name:
        label = 'NHWC → NCHW'
    elif 'cutlass_tensorop_bf16_s16816fprop' in name:
        label = 'patch embedding convolution'
    elif 'flash_fwd_kernel' in name:
        label = 'FlashAttention'
    elif 'native_layer_norm' in name:
        variant = name.rsplit('_', 1)[-1]
        label = ('patch/LayerNorm fusion ' if int(variant) <= 3 else 'residual/LayerNorm fusion ')+variant
    else:
        raise ValueError('Unmapped vision kernel: '+name)
    return 'Vision: '+label


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--framework', type=Path, default=Path('vllm-omni'))
    p.add_argument('--ceilings', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(not (args.output/'summary.json').exists(), 'Allocate a new dated result before replotting')
    basepath = args.baseline/'workload/full_scopes.json'
    baseline = json.loads(basepath.read_text())
    require(baseline['framework']['commit'] == COMMIT, 'Wrong baseline source')
    sourcepath = args.framework/'vllm_omni/diffusion/models/pi05/realtime_triton.py'
    sha = hashlib.sha256(sourcepath.read_bytes()).hexdigest()
    require(sha == next(r['sha256'] for r in baseline['sources'] if r['path'].endswith('realtime_triton.py')),
            'Source differs from baseline')
    frozen = json.loads((args.capture/'source.json').read_text())
    for item in frozen['files']:
        require(file_record(item['path'])['sha256'] == item['sha256'], 'Frozen capture source changed')
    collection = json.loads((args.capture/'ncu/collection.json').read_text())
    require('--launch-count' not in collection['command'], 'Capture was limited by launch count')
    require(collection['command'][collection['command'].index('--nvtx-include')+1] == 'vla/',
            'Capture scope does not include every VLA launch')
    contract = json.loads((args.capture/'metrics.json').read_text())
    raw, annotated = args.export/'raw.csv', args.export/'operators.csv'
    kernels = load_ncu(raw)
    attach_operators(kernels, load_ncu(annotated))
    source = sourcepath.read_text().splitlines()
    vlm, audit = recover(kernels, contract, source)
    labels = {}
    for k in vlm:
        label = label_kernel(k, source)
        key = label+' | '+k['kernel']+' | grid='+k['grid']+' block='+k['block']
        k.update(operator=key, stage='vlm', group_id=0)
        labels[key] = label
    rows = analyze(vlm, contract)
    ids = {key:i+1 for i,key in enumerate(sorted(labels))}
    for r in rows:
        r['group_id'] = ids[r['operator']]
    grouped = aggregate_operators(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_write(args.output/'kernels.csv', rows)
    csv_write(args.output/'operators.csv', grouped)
    mapping = [{'group_id':ids[key], 'label':labels[key], 'operator':key} for key in sorted(labels)]
    csv_write(args.output/'operator_legend.csv', mapping)
    domain0 = next(iter(contract['domains']))
    one = [r for r in rows if r['domain'] == domain0]
    inputs = [raw, annotated, basepath, args.capture/'source.json', args.capture/'metrics.json',
              args.capture/'ncu/collection.json', args.capture/'ncu/ncu.log', sourcepath, args.ceilings]
    summary = {'analyzed_at':timestamp(), 'framework_commit':COMMIT, 'coverage':audit,
        'operator_groups':len(ids), 'collection_started_at':collection['started_at'],
        'collection_failed_at':'2026-09-13T20:19:35+08:00',
        'original_report':file_record(args.capture/'ncu/vla.ncu-rep'),
        'inputs':[file_record(i) for i in inputs], 'input_audit':baseline['input_audit'],
        'input_audit_provenance':'Same frozen harness/config as 27; 28 ncu.log independently confirms 3 true masks and prefix=918.',
        'output_validation':'Capture reached the profiling marker after model-fingerprint and <=1e-5 decomposition guards. End-of-run output/manifest absent after interruption; baseline 27 independently records exact output agreement.',
        'kernel_time_sum_ms':sum(r['duration_ns'] for r in one)/1e6,
        'kernel_l2_bytes_sum':sum(r['memory_bytes'] for r in one),
        'flops':{d:sum(r['flops'] for r in rows if r['domain']==d) for d in contract['domains']},
        'whole_stage_roofline_available':False,
        'note':'Complete measured VLM kernel subset recovered from an incomplete full-VLA report. Kernel time/traffic sums are isolated replay sums, not whole-stage timing/traffic.'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    ceilings = json.loads(args.ceilings.read_text())
    caption = 'Omni 6bdbf97 | 3 valid cameras, prefix=918 | complete VLM subset from cancelled capture 0913/28'
    for invocations, stem in ((False, 'roofline_by_operator'), (True, 'roofline')):
        plot_ops(rows if invocations else grouped, args.output/stem, ceilings,
                 'π0.5 VLM — '+('1,018 measured kernel invocations' if invocations else 'roofline by operator'),
                 inputs, invocations=invocations, label_map=labels, caption=caption)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('inputs','input_audit')},indent=2))


if __name__ == '__main__':
    main()
