"""Offline asset checks and host/cgroup memory headroom, without importing CUDA."""
from __future__ import annotations

import math
import os
from pathlib import Path


def offline_assets(checkpoint, tokenizer):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    for name, path in (("checkpoint", checkpoint), ("tokenizer", tokenizer)):
        if not Path(path).is_dir():
            raise FileNotFoundError(f"Existing local {name} directory required: {path}. Downloads are disabled.")
    if not (Path(tokenizer) / "tokenizer_config.json").is_file():
        raise FileNotFoundError("Local tokenizer_config.json is missing; no remote fallback is allowed.")


def memory_headroom(proc=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
    info = dict(line.split(":", 1) for line in (proc / "meminfo").read_text().splitlines())
    available = int(info["MemAvailable"].split()[0]) * 1024
    result = {"host_available_bytes": available, "cgroup_limits": []}
    # Follow only this process's cgroup ancestry, never scan the cgroup filesystem.
    for line in (proc / "self/cgroup").read_text().splitlines():
        if not line.startswith("0::"):
            continue
        relative = Path(line[3:].lstrip("/"))
        if ".." in relative.parts:
            raise RuntimeError("Cannot resolve this container's cgroup memory limit; inspect its limit before running.")
        current = cgroup_root / relative
        while True:
            maximum, usage = current / "memory.max", current / "memory.current"
            if maximum.is_file() and usage.is_file():
                limit = maximum.read_text().strip()
                if limit != "max":
                    headroom = max(0, int(limit) - int(usage.read_text()))
                    available = min(available, headroom)
                    result["cgroup_limits"].append({"path": str(current), "available_bytes": headroom})
            if current == cgroup_root:
                break
            current = current.parent
    result["available_bytes"] = available
    return result


def require_headroom(minimum_gib):
    if not math.isfinite(minimum_gib) or minimum_gib <= 0:
        raise ValueError("Minimum available memory must be finite and positive.")
    memory = memory_headroom()
    if memory["available_bytes"] < minimum_gib * 1024**3:
        raise RuntimeError(f"Only {memory['available_bytes'] / 1024**3:.1f} GiB available; "
                           f"require {minimum_gib:g} GiB before model initialization.")
    return memory


def stream_checkpoint_weights(directory):
    """Feed the loader one tensor at a time instead of retaining a full state dict."""
    from safetensors import safe_open
    directory = Path(directory)
    paths = ([directory / "model.safetensors"] if (directory / "model.safetensors").is_file()
             else sorted(directory.glob("model-*-of-*.safetensors")))
    if not paths:
        raise FileNotFoundError("No local safetensors checkpoint found.")
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


def buffered_checkpoint_weights(directory, original=False):
    """Read a verified BF16 cache with buffered I/O rather than NFS page faults.

Only the project-generated contiguous BF16 serialization is accepted here.
The benchmark separately verifies the resulting actual model fingerprint.
"""
    import json
    import struct
    import torch
    path = Path(directory) / "model.safetensors"
    if not original:
        manifest = json.loads((Path(directory) / "cache_manifest.json").read_text())
        if path.stat().st_size != manifest["cache_bytes"]:
            raise ValueError("Incomplete weight cache")
    width, dtype, storage = (4, torch.float32, "F32") if original else (2, torch.bfloat16, "BF16")
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < header_size < 16*1024**2:
            raise ValueError("Invalid cache header size")
        header = json.loads(handle.read(header_size))
        offset = 0
        for name in sorted(n for n in header if n != "__metadata__"):
            spec = header[name]
            begin, end = spec["data_offsets"]
            expected_bytes = math.prod(spec["shape"]) * width
            if spec["dtype"] != storage or begin != offset or end-begin != expected_bytes:
                raise ValueError(f"Unsupported derived cache layout: {name}")
            raw = bytearray(expected_bytes)
            if handle.readinto(raw) != expected_bytes:
                raise EOFError(name)
            yield name, torch.frombuffer(raw, dtype=dtype).reshape(spec["shape"])
            offset = end
        if handle.read(1):
            raise ValueError("Unexpected trailing cache data")
