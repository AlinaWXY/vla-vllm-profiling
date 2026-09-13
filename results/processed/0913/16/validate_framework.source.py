"""Validate actual corrected framework source against complete-pipeline ablations.

No numerical monkey patches are installed. Only offline checkpoint/tokenizer
loading and strict parameter-load auditing replace environment-specific defaults.
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
                for key in ("optimized_vs_diagnostic_fix", "safe_vs_original_safe", "graph_repeat"):
                    if metrics[key]["max_abs"] != 0:
                        raise AssertionError(f"{name}: actual source did not reproduce diagnostic: {key} {metrics[key]}")
                arrays.update({name+"__safe": safe, name+"__optimized": optimized, name+"__noise": noise})
                records.append({"input": name, "metrics": metrics})
                print(timestamp(), name, json.dumps(metrics), flush=True)
    framework = Path(rt.__file__).resolve().parents[4]
    commit = subprocess.check_output(["git", "-C", str(framework), "rev-parse", "HEAD"], text=True).strip()
    report = {"started_at": started, "finished_at": timestamp(), "gpu": torch.cuda.get_device_name(),
        "framework_commit": commit, "framework_path": str(framework), "load_audit": audit,
        "scope": "Actual Pi05Pipeline.forward; no numerical substitutions; same fixed real weights, seed 17, 10 steps",
        "cases": records, "sources": [file_record(Path(__file__)), file_record(Path(rt.__file__)),
            file_record(Path(rt.__file__).with_name("modeling_pi05.py")),
            file_record(args.diagnostic / "full_pipeline_actions.npz")]}
    np.savez(args.output / "source_validation_actions.npz", **arrays)
    (args.output / "source_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(timestamp(), "Actual framework matches diagnostic corrections exactly", flush=True)


if __name__ == "__main__":
    main()
