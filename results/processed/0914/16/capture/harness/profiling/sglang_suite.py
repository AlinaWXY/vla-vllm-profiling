"""Calibrate, independently time, capture and audit SGLang pi0.5 in one Slurm job."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from profiling.experiments import timestamp, file_record
from profiling.roofline import load_ncu, metric_value
from profiling.sglang_pi05 import SOURCE, COMMIT, CHECKPOINT, TOKENIZER


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();py=sys.executable
    ncu='/opt/nvidia/nsight-compute/2025.1.1/ncu'
    state={'started_at':timestamp(),'slurm_job_id':os.environ['SLURM_JOB_ID'],'phases':[]}
    def save():
        (out/'suite.json').write_text(json.dumps(state,indent=2)+'\n')
    def run(name,command):
        command=list(map(str,command))
        phase={'name':name,'started_at':timestamp(),'command':command}
        state['phases'].append(phase);state['state']=name;save()
        print(timestamp(),name,flush=True)
        with (out/f'{name}.log').open('x') as f:
            result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        phase.update(finished_at=timestamp(),returncode=result.returncode);save()
        result.check_returncode()
    try:
        assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==COMMIT
        assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip()
        source_files=[SOURCE/'python'/path for path in [
            'sglang/multimodal_gen/runtime/models/vlas/pi05_policy.py',
            'sglang/multimodal_gen/runtime/models/vlas/pi05_core.py',
            'sglang/multimodal_gen/runtime/vla/cuda_graph.py',
            'sglang/multimodal_gen/runtime/layers/attention/layer.py',
            'sglang/multimodal_gen/runtime/layers/linear.py',
            'sglang/srt/models/siglip.py',
            'sglang/srt/layers/attention/vision.py',
            'sglang/srt/layers/activation.py',
            'sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/pi05_preprocess.py']]
        records=[file_record(p) for p in source_files]
        (out/'source.json').write_text(json.dumps({'commit':COMMIT,'source':str(SOURCE),
            'files':records,'harness':[file_record(p) for p in Path(__file__).parent.glob('*.py')],
            'checkpoint_config':file_record(CHECKPOINT/'config.json'),
            'checkpoint_revision':CHECKPOINT.name,
            'checkpoint_sha256':'0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f',
            'tokenizer_config':file_record(TOKENIZER/'tokenizer_config.json')},indent=2)+'\n')
        run('gpu_before',['nvidia-smi','-q'])
        run('discovery',[py,'-m','profiling.ncu','--ncu',ncu,'discover','--output',out/'inventory'])
        contract=json.loads((out/'inventory/metrics.json').read_text())
        extra=['dram__throughput.avg.pct_of_peak_sustained_elapsed',
               'sm__pipe_tensor_cycles_active_v2.avg.pct_of_peak_sustained_elapsed']
        contract['metrics']+=extra;contract['evidence_metrics']=extra
        (out/'metrics.json').write_text(json.dumps(contract,indent=2)+'\n')
        run('calibration',[py,'-m','profiling.calibrate','--output',out/'ceilings.json',
            '--memory-level','dram','--iterations','30','--power-clock-note','Same L20 allocation; default clocks; no clock changes'])
        run('counter_validation',[py,'-m','profiling.ncu','--ncu',ncu,'capture',
            '--contract',out/'metrics.json','--output',out/'compute_ncu','--report-name','compute',
            '--replay-mode','application','--nvtx-scope','compute_reference',
            '--',py,'-m','profiling.validate_compute_counters','--output',out/'compute_workload.json'])
        kernels=load_ncu(out/'compute_ncu/raw.csv')
        k=next(k for k in kernels if metric_value(k,'sm__ops_path_tensor_src_bf16_dst_fp32.sum','count')>0)
        assert metric_value(k,'sm__ops_path_tensor_src_bf16_dst_fp32.sum','count')==2*8192**3
        state['counter_validation_bf16_exact']=True;save()
        run('baseline',[py,'-u','-m','profiling.sglang_pi05','--output',out/'baseline'])
        run('capture',[py,'-u','-m','profiling.ncu','--ncu',ncu,'capture',
            '--contract',out/'metrics.json','--output',out/'ncu','--report-name','vla',
            '--cache-control','all','--replay-mode','application','--nvtx-scope','vla',
            '--scope-description','SGLang native fused operators; VLM and 10 Expert steps; eager dispatch for NVTX; independent native graph latency in baseline',
            '--',py,'-u','-m','profiling.sglang_pi05','--profile','--output-root',out/'passes',
            '--expected',out/'baseline/report.json'])
        import numpy as np
        baseline=json.loads((out/'baseline/report.json').read_text())
        passes=sorted((out/'passes').iterdir())
        assert passes
        state['application_passes']=[]
        with np.load(out/'baseline/actions.npz') as ref:
            for entry in passes:
                report=json.loads((entry/'report.json').read_text())
                for key in ('framework','input_audit','model_parameter_fingerprint','operator_manifest'):
                    assert baseline[key]==report[key],key
                with np.load(entry/'actions.npz') as current:
                    for key in ('noise','optimized','eager','measured'):
                        assert np.array_equal(ref[key],current[key]),(entry,key)
                state['application_passes'].append({'path':str(entry),'status':'exact model/input/actions/manifest match',
                    'started_at':report['started_at'],'finished_at':report['finished_at']})
        assert [file_record(p) for p in source_files]==records
        shutil.copytree(passes[0],out/'workload')
        run('gpu_after',['nvidia-smi','-q'])
        state.update(state='completed',finished_at=timestamp());save()
        print(timestamp(),'SGLang full capture completed',flush=True)
    except BaseException as e:
        state.update(state='failed',finished_at=timestamp(),error=repr(e));save();raise


if __name__=='__main__':
    main()
