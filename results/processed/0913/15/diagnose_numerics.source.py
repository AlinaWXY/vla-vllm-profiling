"""Controlled real-weight ablations of PR4419 numerical differences on Thor.

Changes are process-local diagnostic substitutions, never edits to pinned source.
No latency claims: the same eager image embeddings feed both prefix backends.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from profiling.bench_pi05 import strict_checkpoint_load
from profiling.experiments import timestamp, file_record, clock_snapshot
from profiling.inputs import camera_features
from profiling.ncu import require_gpu_execution
from profiling.runtime import offline_assets, require_headroom, stream_checkpoint_weights


class LaunchAdapter:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        return self.fn


@contextmanager
def replace(module, name, value):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def receive_weights(stream, size, expected_sha256):
    """Verify existing checkpoint bytes into anonymous RAM, without disk files."""
    fd = os.memfd_create("pi05-verified-checkpoint", flags=os.MFD_CLOEXEC)
    digest, total = hashlib.sha256(), 0
    try:
        while total < size:
            block = stream.read(min(8*1024**2, size-total))
            if not block:
                raise ValueError(f"Truncated checkpoint transport: {total}/{size} bytes")
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("Incomplete write to anonymous checkpoint memory")
                view = view[written:]
            total += len(block)
        if stream.read(1):
            raise ValueError("Checkpoint transport contains unexpected trailing bytes")
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Checkpoint transport SHA256 mismatch")
        return fd, {"bytes": total, "sha256": digest.hexdigest(), "storage": "anonymous RAM memfd; no disk copy"}
    except BaseException:
        os.close(fd)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--weights-stdin", action="store_true",
                   help="Receive existing pinned weights into verified anonymous RAM before offline inference")
    args = p.parse_args()
    require_gpu_execution()
    require_headroom(64 if args.weights_stdin else 32)
    offline_assets(args.checkpoint, args.tokenizer)
    if (args.output / "numerics.json").exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    started = timestamp()
    (args.output / "diagnose_numerics.source.py").write_text(Path(__file__).read_text())
    (args.output / "collection.json").write_text(json.dumps({"started_at": started,
        "script": file_record(Path(__file__)), "checkpoint": str(args.checkpoint),
        "seed": args.seed, "steps": args.steps}, indent=2)+"\n")
    checkpoint_fd, transport = None, {"storage": "existing shared checkpoint"}
    if args.weights_stdin:
        lock = json.loads(Path("sources.lock.json").read_text())["checkpoint"]
        print(started, "Receiving pinned checkpoint into anonymous RAM", flush=True)
        checkpoint_fd, transport = receive_weights(sys.stdin.buffer, lock["safetensors_file_bytes"],
                                                    lock["safetensors_file_sha256"])
        print(timestamp(), "Checkpoint transport verified", json.dumps(transport), flush=True)
    print(timestamp(), "Importing runtime", flush=True)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline
    from vllm_omni.diffusion.models.pi05.modeling_pi05 import (
        Pi05ForActionPrediction, make_att_2d_masks, prepare_attention_masks_4d,
    )
    from vllm_omni.diffusion.models.pi05.processor_pi05 import build_model_inputs
    from vllm_omni.diffusion.models.pi05 import realtime_triton as rt

    def error(actual, reference):
        a, b = actual.float(), reference.float()
        d = a - b
        rms, rmse = float(b.square().mean().sqrt()), float(d.square().mean().sqrt())
        return {"rmse": rmse, "reference_rms": rms, "relative_rmse": rmse / rms if rms else None,
                "max_abs": float(d.abs().max()), "mean_abs": float(d.abs().mean())}

    class LocalPipeline(Pi05Pipeline):
        def _load_tokenizer(self):
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)

        def _load_checkpoint(self, model):
            print(timestamp(), "Streaming fixed checkpoint", flush=True)
            def weights():
                count, size, last = 0, 0, time.monotonic()
                def source():
                    if checkpoint_fd is None:
                        yield from stream_checkpoint_weights(self.model_dir)
                    else:
                        from safetensors import safe_open
                        with safe_open(f"/proc/self/fd/{checkpoint_fd}", framework="pt", device="cpu") as handle:
                            for key in handle.keys():
                                yield key, handle.get_tensor(key)
                for name, tensor in source():
                    yield name, tensor
                    count += 1
                    size += tensor.numel()*tensor.element_size()
                    if time.monotonic()-last > 30:
                        print(timestamp(), "Loaded tensors", count, "GiB", round(size/1024**3, 3), flush=True)
                        last = time.monotonic()
            try:
                model.load_weights(weights())
            finally:
                if checkpoint_fd is not None:
                    os.close(checkpoint_fd)

    raw = json.loads((args.checkpoint / "config.json").read_text())
    config = OmniDiffusionConfig(model=str(args.checkpoint), model_class_name="Pi05Pipeline",
                                dtype=torch.bfloat16,
                                model_config={**raw, "tokenizer": str(args.tokenizer), "dtype": "bfloat16"})
    torch.manual_seed(args.seed)
    all_results, arrays = [], {}
    with torch.inference_mode(), strict_checkpoint_load(Pi05ForActionPrediction) as audit:
        pipe = LocalPipeline(od_config=config)
        model = pipe.model
        expert_rotary = model.paligemma_with_expert.gemma_expert.model.rotary_emb
        prefix_rotary = model.paligemma_with_expert.paligemma.model.language_model.rotary_emb
        canonical_inv = 1.0 / (10000 ** (torch.arange(0, 256, 2, device=pipe._device, dtype=torch.float32)/256))
        rotary_audit = {"expert_inv_freq_dtype": str(expert_rotary.inv_freq.dtype),
                        "prefix_inv_freq_dtype": str(prefix_rotary.inv_freq.dtype),
                        "expert_vs_recomputed_fp32": error(expert_rotary.inv_freq, canonical_inv)}
        print(timestamp(), "RoPE frequency audit", json.dumps(rotary_audit), flush=True)
        print(timestamp(), "Model loaded", flush=True)
        for legacy in (True, False):
            name = "legacy_masked_images" if legacy else "three_valid_images"
            rng = np.random.default_rng(args.seed)
            obs = {"prompt": "pick up the red block and place it in the bin",
                   "state": np.zeros(raw["max_state_dim"], dtype=np.float32),
                   "images": {(k.removeprefix("observation.images.") if legacy else k):
                              rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
                              for k in camera_features(raw, 3)}}
            noise_np = rng.standard_normal((1, raw["chunk_size"], raw["max_action_dim"])).astype(np.float32)
            noise = torch.from_numpy(noise_np).to(pipe._device)
            images, image_masks, tokens, masks, meta = build_model_inputs(
                obs, pipe.config, pipe.tokenizer, pipe._device, max_cameras=3, return_metadata=True)
            embs, pads, att = model.embed_prefix(images, image_masks, tokens, masks)
            original_length = int(pads.shape[1])
            valid = int(pads.sum())
            print(timestamp(), name, "prefix", original_length, "valid", valid, flush=True)

            def prefix(e, m, a):
                return model._run_prefix_forward(
                    prefix_embs=e, attention_mask=prepare_attention_masks_4d(make_att_2d_masks(m, a)),
                    position_ids=m.long().cumsum(1)-1, torch_compile_prefix=False,
                    torch_compile_prefix_fullgraph=False, use_packed_prefix_qkv=False,
                    use_packed_prefix_mlp=False)[1]

            def safe(kv, m, static=None):
                x = noise.clone()
                trajectory = []
                for i in range(args.steps):
                    t = torch.full((1,), 1-i/args.steps, device=noise.device, dtype=torch.float32)
                    v = model.denoise_step(prefix_pad_masks=m, past_key_values=kv, x_t=x,
                                           timestep=t, static_context=static, step_index=i)
                    x = x + (-1.0/args.steps) * v
                    trajectory.append(x.clone())
                return x.clone(), torch.stack(trajectory)

            full_kv = prefix(embs, pads, att)
            full_safe, full_trace = safe(full_kv, pads)
            keep = pads[0].bool()
            embs, pads, att = embs[:, keep].contiguous(), pads[:, keep].contiguous(), att[:, keep].contiguous()
            kv = prefix(embs, pads, att)
            trim_safe, trim_trace = safe(kv, pads)
            context = model.prepare_denoise_static_context(
                pads, args.steps, 1, noise.device, precompute_adarms=True)
            static_safe, _ = safe(kv, pads, context)
            enc = model._get_or_create_realtime_triton_prefix_encoder()
            fast_kv = enc(prefix_embs=embs, prefix_pad_masks=pads,
                          prefix_position_ids=pads.long().cumsum(1)-1, valid_prefix_len=valid)
            fast_prefix_safe, _ = safe(fast_kv, pads, context)
            prefix_errors = [{"layer": i, "k": error(fk, sk), "v": error(fv, sv)}
                             for i, ((fk, fv), (sk, sv)) in enumerate(zip(fast_kv, kv))]
            decoder = model._get_or_create_realtime_triton_decoder()
            rope_tables = {}
            model_cos, model_sin = expert_rotary(
                torch.empty(1, raw["chunk_size"], 1024, device=pipe._device, dtype=torch.bfloat16),
                context.position_ids)
            model_cos, model_sin = model_cos[0].contiguous(), model_sin[0].contiguous()

            def rope_launch(x, n, features, dim, heads, w, rope, q, k, v, use_model=False, **blocks):
                key = rope.data_ptr()
                if key not in rope_tables:
                    rope_tables[key] = (rope[:, 0::2].repeat(1, 2).contiguous(),
                                        rope[:, 1::2].repeat(1, 2).contiguous())
                cos, sin = rope_tables[key]
                if use_model:
                    cos, sin = model_cos, model_sin
                rt._matmul_gemma_rope_qkv[((n+31)//32, heads+2)](
                    x, n, features, dim, heads, w, cos, sin, q, k, v,
                    block_m=32, block_half=dim//2, block_k=blocks["block_k"])

            def rope_eager(x, n, features, dim, heads, w, rope, q, k, v, use_model=False, **blocks):
                projected = x @ w
                qr, kr, vr = projected.split((heads*dim, dim, dim), -1)
                qr = qr.reshape(n, heads, dim)
                cos = rope[:, 0::2].repeat(1, 2)
                sin = rope[:, 1::2].repeat(1, 2)
                if use_model:
                    cos, sin = model_cos, model_sin
                def rot(z):
                    return torch.cat((-z[..., dim//2:], z[..., :dim//2]), -1)
                q.copy_((qr*cos[:, None] + rot(qr)*sin[:, None]).reshape_as(q))
                k.copy_(kr*cos + rot(kr)*sin)
                v.copy_(vr)

            def optimized(prefix_kv, mode):
                state = {"x": noise[0].clone()}
                trajectory = []
                step_impl = decoder._step
                def step(*a, **kw):
                    step_impl(*a, **kw)
                    trajectory.append((state["x"] if "euler" in mode else a[0].noise).float().clone())
                def head(x, w, out, b, res, n, features, hidden, **blocks):
                    velocity = F.linear(x, model.action_out_proj.weight, model.action_out_proj.bias)
                    # Match safe: BF16 projected velocity and BF16 dt multiplication,
                    # followed by an FP32 residual update; retain original FP32 noise.
                    state["x"] = state["x"] + (-1.0/args.steps) * velocity
                    out.copy_(state["x"])
                from contextlib import ExitStack
                with ExitStack() as stack:
                    stack.enter_context(replace(decoder, "_step", step))
                    if "rope" in mode:
                        def launch(*a, **kw):
                            fn = rope_eager if "eager" in mode else rope_launch
                            return fn(*a, use_model="model" in mode, **kw)
                        stack.enter_context(replace(rt, "_matmul_rope_qkv",
                                                    LaunchAdapter(launch)))
                    if "euler" in mode:
                        stack.enter_context(replace(rt, "_matmul_small_bias_res", LaunchAdapter(head)))
                    result = decoder(prefix_kv=prefix_kv, prefix_pad_masks=pads, valid_prefix_len=valid,
                                     x_t=noise.clone(), adarms_modulations=context.adarms_modulations,
                                     num_steps=args.steps).clone()
                if "euler" in mode:
                    result = state["x"][None].clone()
                return result, torch.stack(trajectory)[:, None]

            comparisons = {"safe_trim_vs_full": error(trim_safe, full_safe),
                           "safe_static_vs_dynamic": error(static_safe, trim_safe),
                           "fast_prefix_safe_decoder_vs_safe": error(fast_prefix_safe, trim_safe)}
            arrays.update({name+"__safe_full": full_safe.cpu().numpy(),
                           name+"__safe_trim": trim_safe.cpu().numpy(),
                           name+"__safe_trace": trim_trace.cpu().numpy(), name+"__noise": noise_np})
            for prefix_name, prefix_kv in (("safe_prefix", kv), ("fast_prefix", fast_kv)):
                for mode in ("original", "rope", "rope_model", "euler", "rope_euler", "rope_model_euler",
                             "rope_eager_euler", "rope_model_eager_euler"):
                    result, trace = optimized(prefix_kv, mode)
                    label = prefix_name+"__"+mode
                    comparisons[label] = error(result, trim_safe)
                    comparisons[label]["step_errors"] = [error(a, b) for a, b in zip(trace, trim_trace)]
                    arrays[name+"__"+label] = result.cpu().numpy()
                    arrays[name+"__"+label+"__trace"] = trace.cpu().numpy()
                    print(timestamp(), name, label, json.dumps(error(result, trim_safe)), flush=True)
            all_results.append({"input": name, "image_masks": [bool(m.item()) for m in image_masks],
                                "original_prefix_len": original_length, "valid_prefix_len": valid,
                                "metadata": meta, "comparisons": comparisons, "prefix_layer_errors": prefix_errors})
            # Preserve complete cases even if a later case is interrupted.
            (args.output / "partial_cases.json").write_text(json.dumps(all_results, indent=2)+"\n")
            np.savez(args.output / "actions_and_traces.npz", **arrays)
            del full_kv, fast_kv, kv
        torch.cuda.synchronize()
        report = {"started_at": started, "finished_at": timestamp(), "clock": clock_snapshot(),
                  "seed": args.seed, "steps": args.steps, "load_audit": audit, "rotary_audit": rotary_audit,
                  "torch": torch.__version__, "checkpoint": str(args.checkpoint), "weight_transport": transport,
                  "checkpoint_lock": json.loads(Path("sources.lock.json").read_text()),
                  "sources": [file_record(p) for p in (Path(__file__), Path(rt.__file__),
                               Path("vllm-omni/vllm_omni/diffusion/models/pi05/modeling_pi05.py"),
                               args.checkpoint / "config.json", args.tokenizer / "tokenizer_config.json")],
                  "conditions": "BF16 model, FP32 initial noise, batch=1, 3 requested cameras, fixed seed; "
                                "eager common image embeddings; no CUDA Graph or timing measurement; "
                                "process-local decoder substitutions; synthetic observations, no robot task oracle",
                  "cases": all_results}
        (args.output / "numerics.json").write_text(json.dumps(report, indent=2)+"\n")
        print(timestamp(), "Complete", flush=True)


if __name__ == "__main__":
    main()
