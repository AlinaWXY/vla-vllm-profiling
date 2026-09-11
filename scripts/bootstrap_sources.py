#!/usr/bin/env python3
"""Fetch exact upstream source revisions. Never overwrite a dirty checkout."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", *map(str, args)], text=True).strip()


def main():
    sources = json.loads((ROOT / "sources.lock.json").read_text())
    for name in ("vllm", "vllm-omni", "vllm-omni-reference"):
        spec = sources[name]
        path = ROOT / name
        if path.exists():
            if not (path / ".git").exists():
                raise RuntimeError(f"Existing non-repository directory: {path}")
            if git("-C", path, "status", "--porcelain"):
                raise RuntimeError(f"Dirty checkout; preserve local changes: {path}")
            if git("-C", path, "rev-parse", "HEAD") == spec["commit"]:
                print(f"Verified {name}: {spec['commit']}")
                continue
            raise RuntimeError(f"Unexpected revision in {path}; no automatic checkout performed.")
        subprocess.run(["git", "clone", "--no-checkout", "--depth", "1", spec["url"], str(path)], check=True)
        # Fetch by immutable SHA. The PR ref is retained in the lockfile as provenance.
        git("-C", path, "fetch", "--depth", "1", "origin", spec["commit"])
        git("-C", path, "checkout", "--detach", spec["commit"])
        print(f"Fetched {name}: {spec['commit']}")


if __name__ == "__main__":
    main()
