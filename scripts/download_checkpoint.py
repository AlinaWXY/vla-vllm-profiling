#!/usr/bin/env python3
"""Download the pinned real checkpoint into an explicitly configured scratch cache."""
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "sources.lock.json").read_text())["checkpoint"]
    cache = os.environ.get("HF_HUB_CACHE", "")
    if not cache or not Path(cache).resolve().is_relative_to("/scratch"):
        raise RuntimeError("Source scripts/cache_env.sh first; HF_HUB_CACHE must be below /scratch.")
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"],
                             cache_dir=cache, max_workers=2,
                             allow_patterns=["*.json", "*.safetensors", "README.md"])
    print(path)


if __name__ == "__main__":
    main()
