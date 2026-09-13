"""Validate actual corrected framework source against complete-pipeline ablations.

The actual-source measurements install no numerical patches. A separate control
can reproduce the original decoder plus the verified correction in this process.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from profiling.bench_pi05 import strict_checkpoint_load
from profiling.experiments import timestamp, file_record
from profiling.inputs import camera_features
from profiling.ncu import require_gpu_execution
from profiling.runtime import offline_assets, require_headroom, stream_checkpoint_weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "tokenizer", "output", "diagnostic"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--baseline-source", type=Path,
                   help="Original PR4419 realtime_triton.py for a same-process corrected-decoder control")
    args = p.parse_args()
    require_gpu_execution()
    require_headroom(32)
    offline_assets(args.checkpoint, args.tokenizer)
    if (args.output / "source_validation.json").exists():
        raise FileExistsError(args.output)
    started = timestamp()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "collection.json").write_text(json.dumps({"started_at": started}, indent=2) + "\n")
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline
    from vllm_omni.diffusion.models.pi05.modeling_pi05 import Pi05ForActionPrediction
    from vllm_omni.diffusion.models.pi05 import realtime_triton as rt
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    class LocalPipeline(Pi05Pipeline):
        def _load_tokenizer(self):
            return AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)

        def _load_checkpoint(self, model):
            model.load_weights(stream_checkpoint_weights(self.model_dir))

    def error(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        d = a-b
        rmse, rms = float(np.mean(d*d)**0.5), float(np.mean(b*b)**0.5)
        return {"rmse": rmse, "reference_rms": rms, "relative_rmse": rmse/rms,
                "max_abs": float(np.abs(d).max())}

    raw = json.loads((args.checkpoint / "config.json").read_text())
    config = OmniDiffusionConfig(model=str(args.checkpoint), model_class_name="Pi05Pipeline",
        dtype=torch.bfloat16, model_config={**raw, "tokenizer": str(args.tokenizer), "dtype": "bfloat16"})
    records, arrays = [], {}
    torch.manual_seed(17)
    with torch.inference_mode(), strict_checkpoint_load(Pi05ForActionPrediction) as audit:
        pipe = LocalPipeline(od_config=config)
        print(timestamp(), "Loaded actual framework", rt.__file__, flush=True)
        baseline_decoder = None
        if args.baseline_source:
            from profiling.check_rope import load_kernels
            old_rt = load_kernels(args.baseline_source)
            baseline_decoder = old_rt.Pi05RealtimeTritonDecoder(pipe.model)
        with np.load(args.diagnostic / "full_pipeline_actions.npz") as expected:
            for legacy in (True, False):
                name = "legacy_masked_images" if legacy else "three_valid_images"
                rng = np.random.default_rng(17)
                obs = {"prompt": "pick up the red block and place it in the bin",
                    "state": np.zeros(raw["max_state_dim"], dtype=np.float32),
                    "images": {(k.removeprefix("observation.images.") if legacy else k):
                        rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for k in camera_features(raw, 3)}}
                noise = rng.standard_normal((1, raw["chunk_size"], raw["max_action_dim"])).astype(np.float32)

                def run(backend):
                    request = OmniDiffusionRequest(prompts=[obs["prompt"]], request_id="pi05-source-regression",
                        sampling_params=OmniDiffusionSamplingParams(extra_args={"robot_obs": obs, "noise": noise,
                            "num_inference_steps": 10, "pi05_execution_backend": backend,
                            "pi05_max_cameras": 3, "pi05_realtime_max_cameras": 3}))
                    result = pipe.forward(request)
                    if getattr(result, "error", None):
                        raise RuntimeError(result.error)
                    value = result.output["actions"].copy()
                    if value.shape != (50, 32) or not np.isfinite(value).all():
                        raise ValueError("Invalid complete-pipeline output")
                    return value

                safe = run("safe")
                warmup = run("realtime_triton_prefix")
                optimized = run("realtime_triton_prefix")
                metrics = {"optimized_vs_safe": error(optimized, safe),
                    "optimized_vs_diagnostic_fix": error(optimized, expected[name+"__rope_model_image_snapshot"]),
                    "safe_vs_original_safe": error(safe, expected[name+"__safe"]),
                    "graph_repeat": error(optimized, warmup)}
                if baseline_decoder is not None:
                    from profiling.diagnose_numerics import LaunchAdapter, replace
                    from vllm_omni.diffusion.models.pi05.processor_pi05 import build_model_inputs
                    inputs = build_model_inputs(obs, pipe.config, pipe.tokenizer, pipe._device, max_cameras=3)
                    valid = int(inputs[3].sum()) + sum(int(m.item()) * 256 for m in inputs[1])
                    rotary = pipe.model.paligemma_with_expert.gemma_expert.model.rotary_emb
                    cos, sin = rotary(torch.empty(1, 50, 1024, device=pipe._device, dtype=torch.bfloat16),
                                      torch.arange(valid, valid+50, device=pipe._device)[None])
                    cos, sin = cos[0].contiguous(), sin[0].contiguous()
                    def corrected_rope(x, n, features, dim, heads, w, table, q, k, v, **blocks):
                        rt._matmul_gemma_rope_qkv[((n+31)//32, heads+2)](
                            x, n, features, dim, heads, w, cos, sin, q, k, v,
                            block_m=32, block_half=dim//2, block_k=32, num_stages=2)
                    executor = rt.Pi05RealtimeExecutor(
                        pipe.model._get_or_create_realtime_triton_prefix_encoder(), baseline_decoder)
                    pipe.model._prefix_realtime_triton_cuda_graph_caches = {}
                    with replace(old_rt, "_matmul_rope_qkv", LaunchAdapter(corrected_rope)), \
                         replace(pipe.model, "_pi05_realtime_triton_executor", executor):
                        control = run("realtime_triton_prefix")
                        control_repeat = run("realtime_triton_prefix")
                    pipe.model._prefix_realtime_triton_cuda_graph_caches = {}
                    metrics["same_process_diagnostic_control"] = error(optimized, control)
                    metrics["control_graph_repeat"] = error(control_repeat, control)
                    arrays[name+"__same_process_control"] = control
                print(timestamp(), name, json.dumps(metrics), flush=True)
                arrays.update({name+"__safe": safe, name+"__optimized": optimized, name+"__noise": noise})
                records.append({"input": name, "metrics": metrics})
                np.savez(args.output / "source_validation_actions.npz", **arrays)
                (args.output / "partial_cases.json").write_text(json.dumps(records, indent=2)+"\n")
                exact = ["safe_vs_original_safe", "graph_repeat"]
                if baseline_decoder is not None:
                    exact += ["same_process_diagnostic_control", "control_graph_repeat"]
                for key in exact:
                    if metrics[key]["max_abs"] != 0:
                        raise AssertionError(f"{name}: actual source did not reproduce diagnostic: {key} {metrics[key]}")
    framework = Path(rt.__file__).resolve().parents[4]
    commit = subprocess.check_output(["git", "-C", str(framework), "rev-parse", "HEAD"], text=True).strip()
    report = {"started_at": started, "finished_at": timestamp(), "gpu": torch.cuda.get_device_name(),
        "framework_commit": commit, "framework_path": str(framework), "load_audit": audit,
        "scope": "Actual Pi05Pipeline.forward measured without numerical substitutions; separate same-process "
                 "original-decoder correction control; fixed real weights, seed 17, 10 steps",
        "cases": records, "sources": [file_record(Path(__file__)), file_record(Path(rt.__file__)),
            file_record(Path(rt.__file__).with_name("modeling_pi05.py")),
            file_record(args.diagnostic / "full_pipeline_actions.npz")]}
    if args.baseline_source:
        report["sources"].append(file_record(args.baseline_source))
    np.savez(args.output / "source_validation_actions.npz", **arrays)
    (args.output / "source_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(timestamp(), "Actual framework regression checks passed", flush=True)


if __name__ == "__main__":
    main()
