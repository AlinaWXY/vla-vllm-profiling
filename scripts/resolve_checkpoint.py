#!/usr/bin/env python3
"""Resolve the pinned checkpoint from an existing cache. Never download."""
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "sources.lock.json").read_text())["checkpoint"]
    cache = os.environ.get("HF_HUB_CACHE", "")
    if not cache or not Path(cache).is_dir():
        raise RuntimeError("Set HF_HUB_CACHE to the existing model cache; no cache will be created.")
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"],
                             cache_dir=cache, local_files_only=True)
    directory = Path(path)
    if not (directory / "config.json").is_file() or not (
        (directory / "model.safetensors").is_file() or list(directory.glob("model-*-of-*.safetensors"))
    ):
        raise FileNotFoundError("Cached snapshot has no model config/weights; no download attempted.")
    print(path)


if __name__ == "__main__":
    main()
