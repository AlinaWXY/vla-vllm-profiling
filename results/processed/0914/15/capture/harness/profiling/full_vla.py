"""Profile VLM, Action Expert and combined GPU computation of the selected Omni source.

Observations and static timestep conditioning are prepared outside these GPU
scopes. Independent pipeline wall timing includes preprocessing and output copy.
The harness preserves the selected source's numerical semantics.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import time
import subprocess

from profiling.experiments import timestamp, clock_snapshot, file_record


@contextmanager
def annotate_kernels(rt, manifest):
    """Name direct prefix/expert Triton launches by stage, layer and source site."""
    import torch
    state = {"stage": "vlm", "step": -1}
    original_step = rt.Pi05RealtimeTritonDecoder._step
    original_prefix = rt.Pi05RealtimePrefixEncoder.__call__
    original_decoder = rt.Pi05RealtimeTritonDecoder.__call__
    original_mlp = rt.Pi05RealtimePrefixEncoder._run_prefix_mlp
    originals = {}

    def step(self, *args, **kwargs):
        state["step"] += 1
        return original_step(self, *args, **kwargs)

    def prefix(self, *args, **kwargs):
        state["stage"] = "vlm.prefix"
        with torch.cuda.nvtx.range("vlm.prefix"):
            return original_prefix(self, *args, **kwargs)

    def decoder(self, *args, **kwargs):
        state.update(stage="action_expert", step=-1)
        with torch.cuda.nvtx.range("action_expert"):
            return original_decoder(self, *args, **kwargs)

    def prefix_mlp(self, buffers, layer_idx, prefix_len):
        with torch.cuda.nvtx.range(f"vlm.prefix.layer{layer_idx:02d}.FFN"):
            return original_mlp(self, buffers, layer_idx, prefix_len)

    class Kernel:
        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __getitem__(self, grid):
            launch = self.kernel[grid]
            def run(*args, **kwargs):
                frame = inspect.currentframe().f_back
                layer = frame.f_locals.get("layer_idx", -1)
                line = frame.f_lineno
                if state["stage"] == "action_expert" and line >= final_norm_line:
                    layer = -1
                del frame
                label = (f"{state['stage']}.step{state['step']:02d}.layer{layer:02d}."
                         f"{self.name}.line{line}")
                manifest.append({"index": len(manifest), "operator": label, "stage": state["stage"],
                                 "triton_kernel": self.name, "source_line": line, "grid": str(grid),
                                 "tensors": [{"shape": list(t.shape), "dtype": str(t.dtype)}
                                             for t in args if isinstance(t, torch.Tensor)]})
                with torch.cuda.nvtx.range(label):
                    return launch(*args, **kwargs)
            return run

    source, start = inspect.getsourcelines(original_step)
    final_norm_line = start + next(i for i, line in enumerate(source)
                                  if line.startswith("        _adarms_norm_kernel["))
    try:
        rt.Pi05RealtimeTritonDecoder._step = step
        rt.Pi05RealtimePrefixEncoder.__call__ = prefix
        rt.Pi05RealtimeTritonDecoder.__call__ = decoder
        rt.Pi05RealtimePrefixEncoder._run_prefix_mlp = prefix_mlp
        for name, kernel in list(vars(rt).items()):
            if name.startswith("_") and type(kernel).__module__.startswith("triton") and hasattr(kernel, "__getitem__"):
                originals[name] = kernel
                setattr(rt, name, Kernel(name, kernel))
        yield
    finally:
        rt.Pi05RealtimeTritonDecoder._step = original_step
        rt.Pi05RealtimePrefixEncoder.__call__ = original_prefix
        rt.Pi05RealtimeTritonDecoder.__call__ = original_decoder
        rt.Pi05RealtimePrefixEncoder._run_prefix_mlp = original_mlp
        for name, kernel in originals.items():
            setattr(rt, name, kernel)


def run(pipeline, robot_obs, optimized_req, expected_pipeline_actions, observed, args):
    import numpy as np
    import torch
    def stats(values):
        return {"count": len(values), "mean_ms": float(np.mean(values)),
                "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
                "samples_ms": values}
    from vllm_omni.diffusion.models.pi05.processor_pi05 import build_model_inputs
    from vllm_omni.diffusion.models.pi05 import realtime_triton as rt

    started = timestamp()
    model = pipeline.model
    from profiling.model_fingerprint import fingerprint
    print(timestamp(), "Fingerprinting actual model parameters before profiling", flush=True)
    parameter_fingerprint = fingerprint(model)
    expected_fingerprint = getattr(args, "expected_fingerprint", None)
    if expected_fingerprint:
        baseline = json.loads(expected_fingerprint.read_text())
        if parameter_fingerprint != baseline["model_parameter_fingerprint"]:
            raise ValueError("Loaded model parameters differ from the original-checkpoint baseline")
    images, masks, tokens, lang_masks, metadata = build_model_inputs(
        robot_obs, pipeline.config, pipeline.tokenizer, pipeline._device,
        max_cameras=args.cameras, return_metadata=True)
    if not all(bool(mask.item()) for mask in masks) or not metadata["prefix_masks_contiguous"]:
        raise ValueError("Full VLA workload requires all requested cameras and a contiguous valid prefix")
    prefix_len = metadata["prefix_valid_len"]
    decoder = observed["decoder"]
    saved = observed["kwargs"]
    if saved.get("prefix_len") != prefix_len or saved["valid_prefix_len"] != prefix_len:
        raise ValueError("Prepared VLM input and production expert prefix differ")
    executor = model._get_or_create_realtime_triton_executor()
    if executor.decoder is not decoder:
        raise ValueError("Observed decoder differs from the production executor")
    noise = saved["x_t"].clone()
    buffer = decoder.prepare_prefix_buffers(prefix_len=prefix_len, valid_prefix_len=prefix_len,
                                             dtype=noise.dtype, prefix_pad_masks=saved["prefix_pad_masks"])
    prefix_mask = saved["prefix_pad_masks"].clone()
    positions = torch.arange(prefix_len, device=noise.device, dtype=torch.long)[None]
    expert_kwargs = {**saved, "x_t": noise, "decoder_buffers": buffer, "prefix_kv": None,
                     "prefix_len": prefix_len, "prefix_pad_masks": prefix_mask}
    profile_labels = False
    # Follow the selected framework's camera ownership contract, including the
    # corrected source branch. Do not change semantics during decomposition.
    preserve_camera_outputs = "img_emb.clone() if cuda_graph_image_embed" in inspect.getsource(model.embed_prefix)
    framework_root = Path(rt.__file__).resolve().parents[4]
    framework_commit = subprocess.check_output(
        ["git", "-C", str(framework_root), "rev-parse", "HEAD"], text=True).strip()
    image_fn = model._image_embed_callable(True, torch_compile_fullgraph=True)
    image_inputs = []
    for pixels in images:
        key = model._image_cuda_graph_key(pixel_values=pixels, torch_compile_image_embed=True,
                                         torch_compile_image_embed_fullgraph=True)
        image_inputs.append(model._image_cuda_graph_caches[key].static_pixels)

    @contextmanager
    def label(name):
        if profile_labels:
            with torch.cuda.nvtx.range(name):
                yield
        else:
            yield

    def vlm():
        # Recapture the exact compiled vision callable used in production graphs.
        # PyTorch forbids replaying an existing graph while capturing a new one.
        # Preserve its static-input copy and the source's per-camera output clone.
        with label("vlm"):
            embs = []
            for index, pixels in enumerate(images):
                with label(f"vlm.vision.camera{index}"):
                    image_inputs[index].copy_(pixels)
                    image_emb = image_fn(image_inputs[index])
                    embs.append(image_emb.clone() if preserve_camera_outputs else image_emb)
            with label("vlm.text_embedding"):
                embs.append(model.paligemma_with_expert.embed_language_tokens(tokens))
            with label("vlm.concat"):
                prefix_embs = torch.cat(embs, dim=1)[:, :prefix_len]
            executor.prefix_encoder(prefix_embs=prefix_embs, prefix_pad_masks=prefix_mask,
                prefix_position_ids=positions, valid_prefix_len=prefix_len,
                output_k=buffer.k[:, :prefix_len], output_v=buffer.v[:, :prefix_len], return_prefix_kv=False)

    def expert():
        with label("action_expert"):
            return decoder(**expert_kwargs)

    def vla():
        vlm()
        return expert()

    print(timestamp(), "Warming VLM/expert/combined scopes and checking production replay", flush=True)
    for _ in range(args.warmup):
        actual = vla()
    torch.cuda.synchronize()
    expected = torch.as_tensor(expected_pipeline_actions, device=noise.device)[None]
    direct_delta = float((actual.float() - expected.float()).abs().max())
    if direct_delta > 1e-5 or not bool(torch.isfinite(actual).all()):
        raise RuntimeError(f"Scoped replay changes pinned PR output: max_abs={direct_delta}")
    graphs, graph_outputs = {}, {}
    for name, fn in (("vlm", vlm), ("action_expert", expert), ("vla", vla)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn()
        graphs[name], graph_outputs[name] = graph, output
    graphs["vla"].replay()
    torch.cuda.synchronize()
    graph_delta = float((graph_outputs["vla"].float() - expected.float()).abs().max())
    if graph_delta > 1e-5:
        raise RuntimeError(f"Combined scoped CUDA Graph changes PR output: max_abs={graph_delta}")

    report = {"started_at": started, "clock_before": clock_snapshot(),
        "model_parameter_fingerprint": parameter_fingerprint,
        "scopes": {"vlm": "3 image encoders, text embedding, concatenation and prefix Transformer",
                   "action_expert": "Action input projection, 10 denoising steps (or configured count), action head/Euler",
                   "vla": "Combined VLM and Action Expert GPU computation"},
        "excluded_from_gpu_scopes": "Observation preprocessing, tokenization, output D2H copy and static timestep/AdaRMS setup",
        "pipeline_wall_scope": "Original Pi05Pipeline.forward, including preprocessing and output copy; no service transport",
        "framework": {"path": str(framework_root), "commit": framework_commit,
                      "preserve_camera_outputs": preserve_camera_outputs},
        "upstream_semantics": "Selected framework source; no numerical substitutions by the profiling harness",
        "scope_graph_structure": "VLM/Expert/VLA graphs recapture the selected source's compiled kernels, static image input copies and camera output clones; original pipeline wall time is measured separately",
        "input_audit": {"camera_masks": [bool(mask.item()) for mask in masks], "prefix_len": prefix_len,
                        "language_valid_tokens": int(lang_masks.sum().item()), "metadata": metadata},
        "scoped_direct_vs_pipeline_max_abs": direct_delta, "scoped_graph_vs_pipeline_max_abs": graph_delta,
        "profiled": args.profile, "profile_kind": args.profile_kind if args.profile else None,
        "range_order": list(graphs),
        "operator_manifest": []}
    # Save launch choices before NCU can abort a mismatched application pass.
    # Only inspect already-loaded modules; do not walk the compile cache.
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    launch_choices = {}
    for module in PyCodeCache.modules:
        for name, value in vars(module).items():
            if isinstance(value, CachingAutotuner) and value.launchers:
                launch_choices[name] = [str(launcher.config) for launcher in value.launchers]
    from profiling.freeze_launches import audit as audit_launch_policy
    report["fixed_launch_policy"] = audit_launch_policy(launch_choices)
    (args.output / "preprofile.json").write_text(json.dumps({
        "recorded_at": timestamp(), "input_audit": report["input_audit"],
        "compiled_launch_choices": launch_choices}, indent=2) + "\n")
    if not args.profile:
        print(timestamp(), "Measuring original pipeline wall time and isolated/combined CUDA Graph time", flush=True)
        wall_times = []
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result = pipeline.forward(optimized_req)
            if getattr(result, "error", None):
                raise RuntimeError(result.error)
            torch.cuda.synchronize()
            wall_times.append((time.perf_counter() - begin) * 1000)
        report["pipeline_wall"] = stats(wall_times)
        report["gpu_graph_timings"] = {}
        for name, graph in graphs.items():
            for _ in range(args.warmup):
                graph.replay()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(args.iterations):
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            report["gpu_graph_timings"][name] = stats(samples)
    elif args.profile_kind == "kernels":
        print(timestamp(), "NCU: all VLA kernels with VLM/Action Expert attribution", flush=True)
        profile_labels = True
        with annotate_kernels(rt, report["operator_manifest"]):
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            try:
                with torch.cuda.nvtx.range("vla"):
                    actual = vla()
                    torch.cuda.synchronize()
            finally:
                torch.cuda.cudart().cudaProfilerStop()
        profile_labels = False
    else:
        # NCU --graph-profiling graph measures complete CUDA Graphs, preserving
        # intra-graph dependencies/cache reuse. Never substitute kernel time sums.
        for name, graph in graphs.items():
            print(timestamp(), f"NCU: whole {name} CUDA Graph", flush=True)
            for _ in range(args.warmup):
                graph.replay()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            try:
                with torch.cuda.nvtx.range(name):
                    graph.replay()
                    torch.cuda.synchronize()
            finally:
                torch.cuda.cudart().cudaProfilerStop()
    report.update(finished_at=timestamp(), clock_after=clock_snapshot(),
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    report["sources"] = [file_record(Path(__file__)), file_record(Path(rt.__file__))]
    # Inspect only modules loaded by this process; never scan the compile cache.
    from torch._inductor.codecache import PyCodeCache
    generated = args.output / "compiled_code"
    generated.mkdir(exist_ok=True)
    report["compiled_code"] = []
    for path in sorted({Path(m.__file__) for m in PyCodeCache.modules if getattr(m, "__file__", None)}):
        destination = generated / path.name
        if path.is_file():
            destination.write_bytes(path.read_bytes())
            report["compiled_code"].append({**file_record(destination), "original_path": str(path)})
    np.savez(args.output / "actions.npz", optimized=expected_pipeline_actions,
             scoped=actual.float().cpu().numpy(), scoped_graph=graph_outputs["vla"].float().cpu().numpy(),
             noise=noise.cpu().numpy())
    (args.output / "full_scopes.json").write_text(json.dumps(report, indent=2) + "\n")
    print(timestamp(), "Full scopes completed", flush=True)
    return report
