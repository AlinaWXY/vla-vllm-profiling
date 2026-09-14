"""One Slurm allocation: reference rates, independent latency, complete NCU capture."""
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

from profiling.experiments import file_record, timestamp
from profiling.roofline import load_ncu, attach_operators, analyze

OUT = Path(__file__).resolve().parent
ROOT = Path('/fact_data/xinyaowang/vla_vllm_profiling')
OMNI = Path('/fact_data/xinyaowang/vllm-omni-pi05-numerical-fixes')
ASSETS = Path('/scratch/xinyaowang/vla-vllm-profiling/assets')
CHECKPOINT = ASSETS / 'pi05_base/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba'
PY = sys.executable
NCU = os.environ.get('PI05_NCU', '/opt/nvidia/nsight-compute/2025.1.1/ncu')
state = {'started_at': timestamp(), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'phases': []}


def save():
    (OUT / 'suite.json').write_text(json.dumps(state, indent=2) + '\n')


def run(name, command):
    phase = {'name': name, 'started_at': timestamp(), 'command': list(map(str, command))}
    state['phases'].append(phase)
    state['state'] = name
    save()
    print(timestamp(), name, flush=True)
    with (OUT / f'{name}.log').open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    phase.update(finished_at=timestamp(), returncode=result.returncode)
    save()
    result.check_returncode()


def main():
    save()
    commit = subprocess.check_output(['git', '-C', str(OMNI), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != '6bdbf97e357232b8aa3b6caf5a6c8b0fed713c25':
        raise ValueError(f'Framework changed: {commit}')
    model_files = sorted((OMNI / 'vllm_omni/diffusion/models/pi05').glob('*.py'))
    original_sources = [file_record(p) for p in model_files]
    source = {'recorded_at': timestamp(), 'framework_commit': commit,
              'framework_status': subprocess.check_output(['git', '-C', str(OMNI), 'status', '--short'], text=True),
              'framework_files': original_sources,
              'harness_files': [file_record(p) for p in sorted((OUT/'harness/profiling').glob('*.py'))],
              'sources_lock': json.loads((ROOT/'sources.lock.json').read_text()),
              'input_files': [file_record(CHECKPOINT/'config.json'),
                              file_record(ASSETS/'paligemma_tokenizer/tokenizer_config.json')],
              'launch_policy': {'source': str(ROOT/'results/processed/0914/03/baseline/preprofile.json'), 'fixed_choices': file_record(OUT/'launch_config.json')},
              'synthetic_inputs': {'seed':17, 'batch_size':1, 'cameras':3, 'image_size':[224,224],
                                   'denoising_steps':10, 'state':'zeros',
                                   'prompt':'pick up the red block and place it in the bin'}}
    (OUT/'source.json').write_text(json.dumps(source, indent=2)+'\n')
    run('gpu_before', ['nvidia-smi', '-q'])
    run('discovery', [PY, '-m', 'profiling.ncu', '--ncu', NCU, 'discover',
                      '--output', str(OUT/'inventory'), '--memory-level', 'dram'])
    contract = json.loads((OUT/'inventory/metrics.json').read_text())
    extras = [
        'dram__throughput.avg.pct_of_peak_sustained_elapsed',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'lts__throughput.avg.pct_of_peak_sustained_elapsed',
        'sm__pipe_tensor_cycles_active_v2.avg.pct_of_peak_sustained_elapsed',
        'sm__warps_active.avg.pct_of_peak_sustained_active',
        'smsp__issue_active.avg.pct_of_peak_sustained_active',
        'sm__ops_path_tensor_src_bf16_dst_fp32.sum.pct_of_peak_sustained_elapsed',
        'sm__ops_path_tensor_src_bf16_dst_fp32.sum.peak_sustained_elapsed',
        'dram__bytes_read.sum.peak_sustained_elapsed',
        'lts__t_bytes.sum',
    ]
    bases = sorted({m.split('.')[0] for m in extras})
    query = subprocess.check_output([NCU, '--query-metrics-mode', 'suffix', '--metrics', ','.join(bases)], text=True)
    (OUT/'extra_metric_inventory.txt').write_text(query)
    available = {line.split()[0] for line in query.splitlines() if line.split()}
    missing = set(extras) - available
    if missing:
        raise ValueError(f'Unsupported evidence counters: {missing}')
    contract['metrics'] += extras
    contract['evidence_metrics'] = extras
    (OUT/'metrics.json').write_text(json.dumps(contract, indent=2)+'\n')
    run('calibration', [PY, '-m', 'profiling.calibrate', '--output', str(OUT/'ceilings.json'),
                       '--memory-level', 'dram', '--iterations', '30',
                       '--power-clock-note', 'Same Slurm L20 allocation; default clocks/power; gpu_before.log and gpu_after.log; no clock changes'])
    run('compute_validation', [PY, '-m', 'profiling.ncu', '--ncu', NCU, 'capture',
        '--contract', str(OUT/'metrics.json'), '--output', str(OUT/'compute_ncu'),
        '--report-name', 'compute', '--replay-mode', 'application', '--nvtx-scope', 'compute_reference',
        '--scope-description', 'Known-work BF16 and FP32 GEMMs, n=8192, TF32 disabled',
        '--', PY, '-m', 'profiling.validate_compute_counters', '--output', str(OUT/'compute_workload.json')])
    kernels = load_ncu(OUT/'compute_ncu/raw.csv')
    attach_operators(kernels, load_ncu(OUT/'compute_ncu/operators.csv'))
    rows = analyze(kernels, contract)
    reference = {domain: sum(r['flops'] for r in rows if r['domain']==domain and r['operator'].split('/', 1)[0]==domain)
                 for domain in ('bf16_tensor', 'fp32_simt')}
    n = 8192
    assert reference['bf16_tensor'] == 2*n**3, reference
    assert 2*n**3 <= reference['fp32_simt'] <= 2*n**3+2*n**2, reference
    (OUT/'compute_validation.json').write_text(json.dumps(reference, indent=2)+'\n')
    bench = [PY, '-u', '-m', 'profiling.bench_pi05', '--checkpoint', str(CHECKPOINT),
             '--tokenizer', str(ASSETS/'paligemma_tokenizer'), '--buffered-original',
             '--cameras', '3', '--steps', '10', '--warmup', '3', '--iterations', '30', '--scope', 'all',
             '--launch-config', str(OUT/'launch_config.json')]
    run('baseline', [*bench, '--output', str(OUT/'baseline')])
    run('capture', [PY, '-u', '-m', 'profiling.ncu', '--ncu', NCU, 'capture',
        '--contract', str(OUT/'metrics.json'), '--output', str(OUT/'ncu'), '--report-name', 'vla',
        '--cache-control', 'all', '--replay-mode', 'application', '--nvtx-scope', 'vla',
        '--scope-description', 'VLM and Action Expert all kernels; batch 1, 3 valid cameras, 10 denoising steps; static setup/preprocessing excluded',
        '--', PY, '-u', '-m', 'profiling.application_replay', '--output-root', str(OUT/'passes'),
        *bench[4:], '--profile', '--profile-kind', 'kernels',
        '--expected-fingerprint', str(OUT/'baseline/full_scopes.json')])
    import numpy as np
    passes = sorted(p for p in (OUT/'passes').iterdir() if p.is_dir())
    if not passes:
        raise ValueError('No completed application passes')
    first_report = json.loads((passes[0]/'workload/full_scopes.json').read_text())
    state['application_passes'] = []
    with np.load(passes[0]/'workload/actions.npz') as first_actions:
        for attempt in passes:
            report = json.loads((attempt/'workload/full_scopes.json').read_text())
            for key in ('input_audit', 'model_parameter_fingerprint', 'operator_manifest', 'framework'):
                if report[key] != first_report[key]:
                    raise ValueError(f'Application passes differ: {key}')
            choices = json.loads((attempt/'workload/preprofile.json').read_text())['compiled_launch_choices']
            first_choices = json.loads((passes[0]/'workload/preprofile.json').read_text())['compiled_launch_choices']
            if choices != first_choices:
                raise ValueError('Compiled launch choices differ across passes')
            with np.load(attempt/'workload/actions.npz') as current:
                for key in ('optimized', 'noise', 'scoped', 'scoped_graph'):
                    if not np.array_equal(current[key], first_actions[key]):
                        raise ValueError(f'Application pass arrays differ: {key}')
            state['application_passes'].append({'path': str(attempt), 'status': 'identical',
                'started_at': report['started_at'], 'finished_at': report['finished_at']})
    shutil.copytree(passes[0]/'workload', OUT/'workload')
    if [file_record(p) for p in model_files] != original_sources:
        raise ValueError('Framework files changed during collection')
    import numpy as np
    with np.load(OUT/'baseline/actions.npz') as base, np.load(OUT/'workload/actions.npz') as measured:
        assert np.array_equal(base['noise'], measured['noise'])
        delta = float(np.max(np.abs(base['optimized']-measured['optimized'])))
        if not np.isfinite(delta) or delta > 1e-5:
            raise ValueError(f'Baseline and NCU outputs differ: {delta}')
    state['baseline_vs_profile_max_abs'] = delta
    run('gpu_after', ['nvidia-smi', '-q'])
    state.update(state='completed', finished_at=timestamp())
    save()
    print(timestamp(), 'Suite completed', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        state.update(state='failed', finished_at=timestamp(), error=repr(exc))
        save()
        raise
