"""Pinned SGLang native pi0.5 kernels; Slurm-only GPU profiling and graph timing.

NCU uses eager dispatch to retain module NVTX labels. Fused projections,
activations and attention are upstream implementations. Native CUDA graphs are
independently timed and checked against eager execution, outside NCU scopes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from profiling.experiments import timestamp, file_record
from profiling.ncu import require_gpu_execution

SOURCE = Path('/fact_data/xinyaowang/sglang-pi05-profiling')
ROOT = Path('/fact_data/xinyaowang/vla_vllm_profiling')
COMMIT = '5200508b0fd25733752c2c5e3af5539508023c27'
ASSETS = Path('/scratch/xinyaowang/vla-vllm-profiling/assets')
CHECKPOINT = ASSETS/'pi05_base/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba'
TOKENIZER = ASSETS/'paligemma_tokenizer'
PROMPT = 'pick up the red block and place it in the bin'


def tensor_record(t):
    import torch
    array = t.detach().contiguous().view(torch.uint8).cpu().numpy()
    return {'shape': list(t.shape), 'dtype': str(t.dtype),
            'sha256': hashlib.sha256(memoryview(array)).hexdigest()}


def module_family(name, module):
    cls = type(module).__name__
    leaf = name.rsplit('.', 1)[-1]
    if leaf in ('qkv_proj', 'q_proj', 'k_proj', 'v_proj', 'qkv'):
        return 'QKV'
    if leaf in ('o_proj', 'out_proj', 'proj') and 'self_attn' in name:
        return 'O projection'
    if leaf in ('down_proj', 'fc2') and '.mlp.' in name:
        return 'FFN down'
    if leaf == 'mlp':
        # Parent includes gate/up + activation; the down child overrides it.
        return 'FFN up'
    if cls == 'LocalAttention':
        return 'Attention (fused)'
    if cls.startswith('Vision') and 'Attention' in cls:
        return 'Vision attention'
    if 'norm' in leaf.lower():
        return 'Norm / residual'
    if leaf in ('action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out'):
        return 'Action IO'
    return None


@contextmanager
def annotate(model, manifest):
    import torch
    handles = []
    for name, module in model.named_modules():
        family = module_family(name, module)
        if family is None:
            continue
        stage = 'action_expert' if ('gemma_expert' in name or family == 'Action IO') else 'vlm'
        label = '|'.join((stage, family.replace(' / ', '-'), name))
        def pre(m, args, kwargs, label=label, family=family, stage=stage, name=name):
            def shapes(value):
                if isinstance(value, torch.Tensor):
                    return {'shape': list(value.shape), 'dtype': str(value.dtype)}
                if isinstance(value, (tuple, list)):
                    return [shapes(x) for x in value]
                return None
            manifest.append({'label': label, 'stage': stage, 'family': family, 'module': name,
                             'class': type(m).__module__+'.'+type(m).__name__,
                             'args': shapes(args), 'kwargs': {k: shapes(v) for k,v in kwargs.items()}})
            torch.cuda.nvtx.range_push(label)
        def post(m, args, result):
            torch.cuda.nvtx.range_pop()
        handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(module.register_forward_hook(post, always_call=True))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def stats(values):
    import numpy as np
    return {'count': len(values), 'p50_ms': float(np.median(values)),
            'mean_ms': float(np.mean(values)), 'p95_ms': float(np.percentile(values,95)), 'samples_ms': values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--output-root', type=Path, help='Allocate a unique directory for each application replay')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--expected', type=Path)
    a = p.parse_args()
    require_gpu_execution()
    if a.output_root:
        a.output_root.mkdir(parents=True, exist_ok=True)
        for index in range(1000):
            a.output = a.output_root/f'{index:02d}'
            try:
                a.output.mkdir()
                break
            except FileExistsError:
                continue
    elif a.output:
        a.output.mkdir(parents=True, exist_ok=False)
    else:
        p.error('output or output-root is required')
    out = a.output.resolve()
    started = timestamp()
    def log(message):
        print(timestamp(), message, flush=True)
    def save(name, data):
        (out/name).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str)+'\n')
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip() == COMMIT
    log('Importing SGLang policy')
    import numpy as np
    import torch
    from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment, initialize_model_parallel)
    from sglang.multimodal_gen.runtime.models.vlas.pi05_policy import Pi05PolicyModel
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.pi05_preprocess import Pi05Preprocessor

    torch.manual_seed(17)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = Pi05PipelineConfig(tokenizer_name=str(TOKENIZER), enable_global_prefix_cache=False,
                                prompt_token_buckets=[], materialize_dtype='bf16')
    server_args = ServerArgs(model_path=str(CHECKPOINT), pipeline_config=config, num_gpus=1, warmup_mode='off')
    set_global_server_args(server_args)
    init_distributed_environment(distributed_init_method='file://'+str(Path(os.environ['TMPDIR'])/f'sglang-dist-{os.getpid()}'),
                                 device_id=torch.device('cuda:0'))
    initialize_model_parallel(sequence_parallel_degree=1, backend='nccl')
    # Exact shared SRT context setup used by the upstream diffusion GPUWorker.
    from sglang.srt.server_args import ServerArgs as SrtServerArgs
    from sglang.srt.runtime_context import publish
    publish(SrtServerArgs(model_path='dummy', tp_size=1), role='diffusion_gpu_worker')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        log('Loading pinned checkpoint')
        model = Pi05PolicyModel.from_pretrained(str(CHECKPOINT), config, dtype=torch.bfloat16)
        log('Weights loaded; preparing identical raw observations')
        rng = np.random.default_rng(17)
        images = {key: rng.integers(0,256,(224,224,3),dtype=np.uint8) for key in config.image_keys}
        noise_np = rng.standard_normal((1,50,32)).astype(np.float32)
        old_actions = ROOT/'results/processed/0914/07/baseline/actions.npz'
        if old_actions.is_file():
            with np.load(old_actions) as ref:
                assert np.array_equal(noise_np, ref['noise'])
        obs = Pi05Preprocessor(config)({'prompt': PROMPT, 'state': np.zeros(32,dtype=np.float32),
                                      'images': images, 'noise': noise_np})
        audit = {'camera_order': list(config.image_keys), 'camera_masks': [bool(x.item()) for x in obs.image_masks.values()],
                 'language_valid_tokens': int(obs.token_masks.sum()),
                 'images': {k: tensor_record(v) for k,v in obs.images.items()},
                 'tokens': tensor_record(obs.tokens), 'token_masks': tensor_record(obs.token_masks),
                 'noise': tensor_record(obs.noise), 'state': tensor_record(obs.state)}
        assert audit['camera_masks'] == [True,True,True]
        assert audit['language_valid_tokens'] == 150, audit
        # H2D and preprocessing are outside all GPU operator scopes.
        obs.images = {k: v.cuda() for k,v in obs.images.items()}
        noise = obs.noise.cuda()
        context = None
        def vlm(graph=True):
            nonlocal context
            context = model.encode_prefix(obs, use_cuda_graph=graph, bucket_prompt=False)
            return context
        def expert(graph=True):
            return model.sample_actions(obs, context, noise=noise, num_steps=10, use_cuda_graph=graph)
        def full(graph=True):
            vlm(graph)
            return expert(graph)
        log('Warming native eager and CUDA graph paths')
        eager = full(False).clone()
        for _ in range(3):
            optimized = full(True).clone()
        torch.cuda.synchronize()
        audit['prefix_len'] = context.prefix_len
        audit['prefix_valid_len'] = int(context.prefix_pad_masks.sum())
        audit['prefix_layout'] = context.layout
        assert audit['prefix_len'] == audit['prefix_valid_len'] == 918, audit
        delta = float((eager-optimized).abs().max())
        assert torch.isfinite(optimized).all() and torch.isfinite(eager).all()
        log(f'Prefix audit passed; eager vs native graph max abs {delta}')
        from profiling.model_fingerprint import fingerprint
        fp = fingerprint(model.core_model)
        packages = {d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
        save('modules.json',[{'name':n,'class':type(m).__module__+'.'+type(m).__name__,
                             'family':module_family(n,m),
                             'backend':str(getattr(m,'qkv_backend_name',getattr(m,'attn_backend',getattr(m,'backend',None)))),
                             'impl':type(getattr(m,'attn_impl',None)).__name__}
                            for n,m in model.core_model.named_modules()])
        report = {'started_at':started, 'framework':{'path':str(SOURCE),'commit':COMMIT},
                  'slurm_job_id':os.environ['SLURM_JOB_ID'], 'host':socket.gethostname(),
                  'packages':packages, 'config':vars(config), 'input_audit':audit,
                  'model_parameter_fingerprint':fp, 'graph_vs_eager_max_abs':delta,
                  'parameter_dtypes':dict(Counter(str(x.dtype) for x in model.parameters())),
                  'profile_dispatch':'eager with upstream fused kernels; native graphs timed separately',
                  'tf32_allowed':{'matmul':torch.backends.cuda.matmul.allow_tf32,
                                  'cudnn':torch.backends.cudnn.allow_tf32}}
        from dataclasses import asdict
        report['graph_cache_info'] = {'prefix':asdict(model.prefix_graph_runner.cache_info()),
                                      'action':asdict(model.graph_runner.cache_info())}
        assert all(x['captures'] > 0 and x['failures'] == 0 for x in report['graph_cache_info'].values()), report['graph_cache_info']
        save('preprofile.json',report)
        if a.expected:
            reference = json.loads(a.expected.read_text())
            for key in ('framework','input_audit','model_parameter_fingerprint'):
                assert report[key] == reference[key], key
        timings = {}
        if not a.profile:
            for name, fn in [('vlm',vlm), ('action_expert',expert), ('vla',full)]:
                gpu, wall = [], []
                for _ in range(3 if a.smoke else a.iterations):
                    torch.cuda.synchronize()
                    st, ed = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start = time.perf_counter(); st.record(); fn(True); ed.record(); ed.synchronize()
                    gpu.append(st.elapsed_time(ed)); wall.append((time.perf_counter()-start)*1000)
                timings[name] = {'cuda_event':stats(gpu), 'synchronized_wall':stats(wall)}
            log(f'Native graph timing: {timings["vla"]["cuda_event"]["p50_ms"]:.3f} ms')
        # Warm eager again after graph allocations, then instrument one full VLA.
        for _ in range(2):
            eager = full(False).clone()
        torch.cuda.synchronize()
        manifest = []
        with annotate(model.core_model, manifest):
            if a.profile:
                torch.cuda.cudart().cudaProfilerStart()
            with torch.cuda.nvtx.range('vla'):
                with torch.cuda.nvtx.range('vlm|Other|prefix'):
                    vlm(False)
                with torch.cuda.nvtx.range('action_expert|Other|denoise'):
                    measured = expert(False)
                torch.cuda.synchronize()
            if a.profile:
                torch.cuda.cudart().cudaProfilerStop()
        measured = measured.clone()
        assert torch.equal(measured,eager), 'NVTX instrumentation changed eager outputs'
        np.savez(out/'actions.npz',noise=noise_np,optimized=optimized.float().cpu().numpy(),
                 eager=eager.float().cpu().numpy(),measured=measured.float().cpu().numpy())
        report.update(finished_at=timestamp(), timings=timings, operator_manifest=manifest,
                      instrumented_vs_eager_max_abs=0.0)
        save('report.json', report)
        log('SGLang workload completed')
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
