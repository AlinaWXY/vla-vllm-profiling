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
    for name in ("vllm", "vllm-omni", "sglang"):
        spec = sources[name]
        path = ROOT / name
        if path.exists():
            if not (path / ".git").exists() and any(path.iterdir()):
                raise RuntimeError(f"Existing non-repository directory: {path}")
        if (path / ".git").exists():
            if git("-C", path, "status", "--porcelain"):
                raise RuntimeError(f"Dirty checkout; preserve local changes: {path}")
            if git("-C", path, "rev-parse", "HEAD") == spec["commit"]:
                print(f"Verified {name}: {spec['commit']}")
                continue
            raise RuntimeError(f"Unexpected revision in {path}; no automatic checkout performed.")
        if (ROOT / ".git").exists() and git("-C", ROOT, "ls-files", "--stage", "--", name).startswith("160000 "):
            git("-C", ROOT, "submodule", "update", "--init", "--depth", "1", "--", name)
            if git("-C", path, "rev-parse", "HEAD") != spec["commit"]:
                raise RuntimeError(f"Submodule and source lock disagree: {path}")
            print(f"Initialized {name}: {spec['commit']}")
            continue
        subprocess.run(["git", "clone", "--no-checkout", "--depth", "1", spec["url"], str(path)], check=True)
        # Fetch by immutable SHA. The PR ref is retained in the lockfile as provenance.
        git("-C", path, "fetch", "--depth", "1", "origin", spec["commit"])
        git("-C", path, "checkout", "--detach", spec["commit"])
        print(f"Fetched {name}: {spec['commit']}")


if __name__ == "__main__":
    main()
