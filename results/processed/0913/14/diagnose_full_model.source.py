"""Run numeric ablations, then validate complete Pi05Pipeline graph executions.

Retain the loaded model to avoid a second checkpoint load. Corrections remain
process-local and graph caches are rebuilt after every decoder substitution.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

from profiling import diagnose_numerics as diagnosis
from profiling.experiments import timestamp, file_record
from profiling.ncu import require_gpu_execution
from profiling.runtime import require_headroom


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=10)
    args, _ = parser.parse_known_args()
    require_gpu_execution()
    require_headroom(32)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline
    from vllm_omni.diffusion.models.pi05 import realtime_triton as rt
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    # Keep the diagnostic half-split kernel within L20's 99 KiB shared-memory
    # budget. This changes launch tiling only; pinned model source is untouched.
    gemma_kernel = rt._matmul_gemma_rope_qkv
    class BoundedGemmaKernel:
        def __getitem__(self, grid):
            def launch(*a, **kw):
                if kw.get("block_k", 0) > 64:
                    kw.update(block_k=32, num_stages=2)
                return gemma_kernel[grid](*a, **kw)
            return launch
    rt._matmul_gemma_rope_qkv = BoundedGemmaKernel()

    pipelines = []
    init = Pi05Pipeline.__init__
    def retain(self, *a, **kw):
        init(self, *a, **kw)
        pipelines.append(self)
    with diagnosis.replace(Pi05Pipeline, "__init__", retain):
        diagnosis.main()
    if len(pipelines) != 1:
        raise RuntimeError("Expected one model instance across both diagnostic phases")
    pipe = pipelines[0]
    model = pipe.model
    raw = json.loads((args.checkpoint / "config.json").read_text())
    first = json.loads((args.output / "numerics.json").read_text())
    records, arrays = [], {}
    started = timestamp()
    print(started, "Beginning full Pi05Pipeline verification", flush=True)
    original_decoder = rt.Pi05RealtimeTritonDecoder.__call__

    def euler_decoder(self, *a, **kw):
        state = {"x": kw["x_t"][0].clone()}
        def head(x, w, out, b, res, n, features, hidden, **blocks):
            velocity = F.linear(x, model.action_out_proj.weight, model.action_out_proj.bias)
            state["x"] = state["x"] + (-1.0/kw["num_steps"]) * velocity
            out.copy_(state["x"])
        with diagnosis.replace(rt, "_matmul_small_bias_res", diagnosis.LaunchAdapter(head)):
            original_decoder(self, *a, **kw)
        return state["x"][None]

    def error(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        d = a-b
        rmse, rms = float(np.mean(d*d)**0.5), float(np.mean(b*b)**0.5)
        return {"rmse": rmse, "reference_rms": rms, "relative_rmse": rmse/rms if rms else None,
                "max_abs": float(np.abs(d).max()), "mean_abs": float(np.abs(d).mean())}

    with torch.inference_mode():
        for case in first["cases"]:
            name = case["input"]
            legacy = name == "legacy_masked_images"
            rng = np.random.default_rng(args.seed)
            obs = {"prompt": "pick up the red block and place it in the bin",
                   "state": np.zeros(raw["max_state_dim"], dtype=np.float32),
                   "images": {(k.removeprefix("observation.images.") if legacy else k):
                              rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
                              for k in diagnosis.camera_features(raw, 3)}}
            noise = rng.standard_normal((1, raw["chunk_size"], raw["max_action_dim"])).astype(np.float32)
            def run(backend):
                request = OmniDiffusionRequest(
                    prompts=[obs["prompt"]], request_id="pi05-complete-model-numerics",
                    sampling_params=OmniDiffusionSamplingParams(extra_args={
                        "robot_obs": obs, "noise": noise, "num_inference_steps": args.steps,
                        "pi05_execution_backend": backend, "pi05_max_cameras": 3,
                        "pi05_realtime_max_cameras": 3}))
                out = pipe.forward(request)
                if getattr(out, "error", None):
                    raise RuntimeError(out.error)
                value = out.output["actions"].copy()
                if value.shape != (raw["chunk_size"], raw["max_action_dim"]) or not np.isfinite(value).all():
                    raise ValueError("Invalid full-model actions")
                return value

            reference = run("safe")
            repeat = run("safe")
            arrays[name+"__safe"] = reference
            metrics = {"safe_repeat": error(repeat, reference)}
            rotary = model.paligemma_with_expert.gemma_expert.model.rotary_emb
            positions = torch.arange(case["valid_prefix_len"], case["valid_prefix_len"]+raw["chunk_size"],
                                     device=pipe._device)[None]
            cos, sin = rotary(torch.empty(1, raw["chunk_size"], 1024, device=pipe._device,
                                          dtype=torch.bfloat16), positions)
            cos, sin = cos[0].contiguous(), sin[0].contiguous()
            def rope(x, n, features, dim, heads, w, table, q, k, v, **blocks):
                rt._matmul_gemma_rope_qkv[((n+31)//32, heads+2)](
                    x, n, features, dim, heads, w, cos, sin, q, k, v,
                    block_m=32, block_half=dim//2, block_k=blocks["block_k"])
            def rope_eager(x, n, features, dim, heads, w, table, q, k, v, **blocks):
                qr, kr, vr = (x @ w).split((heads*dim, dim, dim), -1)
                qr = qr.reshape(n, heads, dim)
                def rot(z):
                    return torch.cat((-z[..., dim//2:], z[..., :dim//2]), -1)
                q.copy_((qr*cos[:, None] + rot(qr)*sin[:, None]).reshape_as(q))
                k.copy_(kr*cos + rot(kr)*sin)
                v.copy_(vr)

            for mode in ("original", "rope_model", "rope_model_euler", "rope_model_eager_euler"):
                # Captured graphs retain old kernels: explicitly rebuild the affected graph.
                model._prefix_realtime_triton_cuda_graph_caches = {}
                with ExitStack() as stack:
                    if "rope" in mode:
                        stack.enter_context(diagnosis.replace(rt, "_matmul_rope_qkv",
                            diagnosis.LaunchAdapter(rope_eager if "eager" in mode else rope)))
                    if "euler" in mode:
                        stack.enter_context(diagnosis.replace(rt.Pi05RealtimeTritonDecoder,
                                                               "__call__", euler_decoder))
                    print(timestamp(), name, "Full pipeline warmup/capture", mode, flush=True)
                    warmup = run("realtime_triton_prefix")
                    actual = run("realtime_triton_prefix")
                metrics[mode] = error(actual, reference)
                metrics[mode]["capture_vs_replay"] = error(actual, warmup)
                arrays[name+"__"+mode] = actual
                print(timestamp(), name, "FULL_MODEL", mode, json.dumps(metrics[mode]), flush=True)
            records.append({"input": name, "image_masks": case["image_masks"], "metrics": metrics})
            np.savez(args.output / "full_pipeline_actions.npz", **arrays)
            (args.output / "full_pipeline_partial.json").write_text(json.dumps(records, indent=2)+"\n")
    report = {"started_at": started, "finished_at": timestamp(), "cases": records,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "scope": "Pi05Pipeline.forward including preprocessing, image encoder, prefix encoder, "
                       "all 10 denoising steps, action output; original production backend graph settings",
              "sources": [file_record(Path(__file__)), file_record(args.output / "numerics.json")],
              "notes": "No latency claim. Each substitution rebuilds prefix+decoder graphs. "
                       "Report contains normalized-space actions from synthetic observations and fixed real weights."}
    (args.output / "full_pipeline.json").write_text(json.dumps(report, indent=2)+"\n")
    print(timestamp(), "Full model verification complete", flush=True)


if __name__ == "__main__":
    main()
