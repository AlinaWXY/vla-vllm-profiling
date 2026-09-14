"""Check the two retained results against their bundled NCU and replay evidence."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

from profiling.coarse_roofline import aggregate, family, PCTS
from profiling.roofline import analyze, attach_operators, load_ncu, metric_value

RESULTS = ('0914/15', '0914/16')


def read_json(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return json.load(handle)


def equal(actual, expected, label):
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-6), (label, actual, expected)
    else:
        assert actual == expected, (label, actual, expected)


def csv_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def verify(result):
    import numpy as np
    summary = read_json(result / 'summary.json')
    publication = read_json(result / 'publication.json')
    for entry in publication['files']:
        data = (result / entry['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry['published_sha256'], entry['path']
        if not entry.get('publication_edit'):
            original = gzip.decompress(data) if entry['encoding'] == 'gzip' else data
            assert hashlib.sha256(original).hexdigest() == entry['original_sha256'], entry['path']
    capture = result / 'capture'
    contract = read_json(capture / 'metrics.json')
    assert contract['memory']['level'] == 'dram'
    raw = load_ncu(capture / 'ncu/raw.csv.gz')
    attach_operators(raw, load_ncu(capture / 'ncu/operators.csv.gz'))
    measured = {(r['id'], r['domain']): r for r in analyze(raw, contract)}
    rows = csv_rows(result / 'kernels.csv')
    assert len(rows) == len(raw) == summary['verification']['kernel_count']
    assert len({row['id'] for row in rows}) == len(rows)
    sglang = (result / 'nvtx_parent_audit.json').exists()
    report_name = 'report.json' if sglang else 'full_scopes.json'
    report = read_json(capture / 'workload' / report_name)
    audit = {r['kernel_id']: r for r in read_json(result / 'nvtx_parent_audit.json')} if sglang else {}
    numeric = {'duration_ns', 'dram_bytes', 'bf16_flops', 'fp32_flops', 'l2_bytes',
               'ncu_bf16_peak_ops', 'ncu_dram_peak_bytes', *PCTS}
    for row, kernel in zip(rows, raw):
        equal(row['id'], kernel['id'], 'kernel ID')
        equal(row['kernel'], kernel['kernel'], 'kernel name')
        for key in numeric & row.keys():
            row[key] = None if row[key] == '' else float(row[key])
        if sglang:
            for key in ('grid', 'block'):
                equal(row[key], kernel[key], key)
            resolved = kernel['operator'].split('/', 1)[0]
            if kernel['id'] in audit:
                equal(audit[kernel['id']]['cli_label'], kernel['operator'], 'shadowed NVTX')
                resolved = audit[kernel['id']]['resolved_label']
            equal(row['nvtx'], resolved, 'NVTX')
        else:
            equal(row['nvtx'], kernel['operator'].split('/', 1)[0], 'NVTX')
            equal(row['family'], family(row['semantic_label']), 'operator family')
        bf, fp = measured[row['id'], 'bf16_tensor'], measured[row['id'], 'fp32_simt']
        for key, value in {'duration_ns': bf['duration_ns'], 'dram_bytes': bf['memory_bytes'],
                           'bf16_flops': bf['flops'], 'fp32_flops': fp['flops']}.items():
            equal(row[key], value, key)
        for key, metric in PCTS.items():
            if key in row:
                equal(row[key], kernel['metrics'][metric][0], key)
    grouped = aggregate(rows, summary['empirical_ceilings'])
    assert len(grouped) == len(summary['operators'])
    published = csv_rows(result / 'operators.csv')
    assert len(published) == len(grouped)
    for calculated, saved, csv_row in zip(grouped, summary['operators'], published):
        for key, value in calculated.items():
            equal(value, saved[key], key)
            text_value = csv_row[key]
            parsed = None if text_value == '' else float(text_value) if isinstance(value, (int,float)) else text_value
            equal(parsed, value, 'operator CSV ' + key)
    known = load_ncu(capture / 'compute_ncu/raw.csv')
    counted = sum(metric_value(k, 'sm__ops_path_tensor_src_bf16_dst_fp32.sum', 'count') or 0 for k in known)
    equal(counted, 2 * 8192**3, 'known BF16 GEMM FLOPs')
    baseline = read_json(capture / 'baseline' / report_name)
    for key in ('framework', 'input_audit', 'model_parameter_fingerprint'):
        equal(baseline[key], report[key], 'baseline ' + key)
    with np.load(capture / 'workload/actions.npz') as reference, np.load(capture / 'baseline/actions.npz') as base:
        for key in ('noise', 'optimized'):
            assert np.array_equal(reference[key], base[key]), 'baseline actions ' + key
        passes = sorted((capture / 'passes').iterdir())
        assert len(passes) == len(read_json(capture / 'suite.json')['application_passes'])
        for entry in passes:
            current = read_json(entry / (report_name + '.gz'))
            for key in ('framework', 'input_audit', 'model_parameter_fingerprint', 'operator_manifest'):
                equal(current[key], report[key], 'pass ' + key)
            with np.load(entry / 'actions.npz') as actions:
                for key in (('noise','optimized','eager','measured') if sglang else ('noise','optimized','scoped','scoped_graph')):
                    assert np.array_equal(reference[key], actions[key]), (entry, key)
    print(f'{result.name}: {len(raw)} kernels, {len(grouped)} operator groups, {len(passes)} replay passes; verified')


def main():
    root = Path(__file__).resolve().parents[1] / 'results/processed'
    for name in RESULTS:
        verify(root / name)


if __name__ == '__main__':
    main()
