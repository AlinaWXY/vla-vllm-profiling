"""Real-checkpoint pipeline benchmark plus isolated Action Expert replay.

Run in the environment for pinned vLLM-Omni PR 4419, inside Slurm. This exercises
Pi05Pipeline directly, not the websocket transport or the serving scheduler.
GPU execution and Thor compatibility remain pending until actually measured.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

from profiling.ncu import require_slurm
from profiling.instrumentation import annotate_decoder, observe_decoder


@contextmanager
def strict_checkpoint_load(model_class):
    original = model_class.load_weights
    audit = {}

    def load(self, weights):
        loaded = original(self, weights)
        expected = {name for name, _ in self.named_parameters()
                    if "rotary_emb" not in name and not name.endswith(".inv_freq")}
        missing = sorted(expected - loaded)
        audit.update(loaded_parameter_names=len(loaded), missing_parameter_names=missing)
        if missing:
            raise RuntimeError(f"Refusing partially initialized checkpoint: {missing[:10]}")
        return loaded

    model_class.load_weights = load
    try:
        yield audit
    finally:
        model_class.load_weights = original


def stats(values):
    import numpy as np
    return {"count": len(values), "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
            "samples_ms": values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cameras", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    require_slurm()
    if min(args.steps, args.warmup, args.iterations) < 1:
        p.error("steps, warmup and iterations must be positive")
    raw_config = json.loads((args.checkpoint / "config.json").read_text())
    if raw_config.get("type") != "pi05":
        raise ValueError("A real LeRobot pi05 checkpoint with type=pi05 is required.")
    if not ((args.checkpoint / "model.safetensors").is_file()
            or list(args.checkpoint.glob("model-*-of-*.safetensors"))):
        raise FileNotFoundError("Checkpoint weights absent; random initialization is forbidden.")

    import numpy as np
    import torch
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline
    from vllm_omni.diffusion.models.pi05.modeling_pi05 import Pi05ForActionPrediction
    from vllm_omni.diffusion.models.pi05 import realtime_triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this allocation.")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    camera_keys = list(raw_config["input_features"])
    camera_keys = [k for k in camera_keys if k.startswith("observation.images.")][:args.cameras]
    # Real weights, synthetic observations. This is a performance workload,
    # not a robot success-rate or action-space evaluation.
    robot_obs = {"prompt": "pick up the red block and place it in the bin",
                 "state": np.zeros(raw_config["max_state_dim"], dtype=np.float32),
                 "images": {k.removeprefix("observation.images."):
                            rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for k in camera_keys}}
    noise = rng.standard_normal((1, raw_config["chunk_size"], raw_config["max_action_dim"])).astype(np.float32)
    model_config = {**raw_config, "tokenizer": args.tokenizer, "dtype": "bfloat16"}
    config = OmniDiffusionConfig(model=str(args.checkpoint), model_class_name="Pi05Pipeline", dtype=torch.bfloat16,
                                model_config=model_config)

    with torch.inference_mode(), strict_checkpoint_load(Pi05ForActionPrediction) as load_audit:
        pipeline = Pi05Pipeline(od_config=config)

        def request(backend):
            return OmniDiffusionRequest(
                prompts=[robot_obs["prompt"]], request_id="pi05-roofline",
                sampling_params=OmniDiffusionSamplingParams(extra_args={
                    "robot_obs": robot_obs, "noise": noise, "num_inference_steps": args.steps,
                    "pi05_execution_backend": backend, "pi05_max_cameras": args.cameras,
                    "pi05_realtime_max_cameras": args.cameras}))

        def forward(req):
            result = pipeline.forward(req)
            if getattr(result, "error", None):
                raise RuntimeError(result.error)
            actions = result.output["actions"]
            if actions.shape != (raw_config["chunk_size"], raw_config["max_action_dim"]) or not np.isfinite(actions).all():
                raise RuntimeError("Invalid action chunk.")
            return actions

        safe_req, optimized_req = request("safe"), request("realtime_triton_prefix")
        safe_actions = forward(safe_req)
        safe_times = []
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            forward(safe_req)
            torch.cuda.synchronize()
            safe_times.append((time.perf_counter() - begin) * 1000)
        with observe_decoder(realtime_triton) as observed:
            for _ in range(args.warmup):
                optimized_actions = forward(optimized_req)
        if not observed:
            raise RuntimeError("Optimized decoder was never invoked.")
        optimized_times = []
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            optimized_actions = forward(optimized_req)
            torch.cuda.synchronize()
            optimized_times.append((time.perf_counter() - begin) * 1000)
        np.savez(args.output / "actions.npz", safe=safe_actions, optimized=optimized_actions, noise=noise)

        decoder, kwargs = observed["decoder"], observed["kwargs"]
        # Snapshot input noise separately; the decoder owns mutable scratch buffers.
        kwargs["x_t"] = kwargs["x_t"].clone()
        def expert():
            return decoder(**kwargs)
        for _ in range(args.warmup):
            expert()
        torch.cuda.synchronize()
        expected = expert().clone()
        # Measure the same optimized decoder kernels with and without CUDA Graph.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_actions = expert()
        graph.replay()
        torch.cuda.synchronize()
        delta = float((expected.float() - graph_actions.float()).abs().max())
        if not torch.isfinite(graph_actions).all() or delta > 1e-5:
            raise RuntimeError(f"Decoder graph replay changed actions: max_abs={delta}")
        times = {}
        for name, fn in (("expert_diagnostic_launches", expert), ("expert_cuda_graph", graph.replay)):
            samples = []
            for _ in range(args.iterations):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            times[name] = stats(samples)

        manifest = []
        if args.profile:
            with annotate_decoder(realtime_triton, decoder, manifest):
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.nvtx.range_push("action_expert")
                try:
                    expert()
                    torch.cuda.synchronize()
                finally:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.cudart().cudaProfilerStop()
        report = {
            "slurm_job_id": os.environ["SLURM_JOB_ID"], "host": platform.node(),
            "gpu": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "packages": {name: importlib.metadata.version(name) for name in ("vllm", "vllm-omni", "transformers", "triton")},
            "upstream_model_file": str(Path(realtime_triton.__file__).resolve()),
            "workload": {"checkpoint": str(args.checkpoint), "dtype": "bfloat16 blanket cast (PR4419)",
                         "batch": 1, "cameras": args.cameras, "steps": args.steps, "seed": args.seed,
                         "input": "Synthetic images and zero state; repeated observation; cross-request caching disabled",
                         "expert_scope": "Action input projection, 10-step denoising (or configured steps), final head and Euler; prefix and AdaRMS precompute excluded"},
            "load_audit": load_audit, "pipeline_safe_wall": stats(safe_times),
            "pipeline_optimized_wall": stats(optimized_times), **times,
            "comparison": {"reference": "same PR safe backend; external LeRobot parity NOT yet established",
                           "safe_vs_optimized_max_abs": float(np.max(np.abs(safe_actions - optimized_actions))),
                           "safe_vs_optimized_mean_abs": float(np.mean(np.abs(safe_actions - optimized_actions))),
                           "expert_eager_vs_graph_max_abs": delta},
            "profiled": args.profile,
            "timing_warning": "If launched under NCU, use a separate uninstrumented invocation for latency claims.",
            "operator_manifest": manifest,
        }
        (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "operator_manifest"}, indent=2))


if __name__ == "__main__":
    main()
