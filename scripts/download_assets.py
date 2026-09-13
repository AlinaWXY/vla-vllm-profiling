#!/usr/bin/env python3
"""Download pinned public pi05 assets off Thor, with resumable transfer and hashes."""
import argparse
import getpass
import os
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm="sha256", git_blob=False):
    h = hashlib.new(algorithm)
    if git_blob:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def download(url, destination, size, expected, git_blob=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if not destination.exists():
        print(f"Downloading {destination.name}: {size:,} bytes", flush=True)
        subprocess.run(["curl", "--disable", "--fail", "--location", "--silent", "--show-error",
                        "--retry", "5", "--retry-delay", "3", "--connect-timeout", "30",
                        "--speed-limit", "1024", "--speed-time", "120", "--continue-at", "-",
                        "--output", str(partial), url], check=True)
        candidate = partial
    else:
        candidate = destination
    if candidate.stat().st_size != size:
        raise RuntimeError(f"Size mismatch: {destination.name}")
    actual = digest(candidate, "sha1" if git_blob else "sha256", git_blob)
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch: {destination.name}; preserving file for inspection")
    sha256 = digest(candidate) if git_blob else actual
    if candidate == partial:
        partial.replace(destination)
    print(f"Verified {destination.name}: sha256={sha256}", flush=True)
    return {"url": url, "path": str(destination), "bytes": size, "sha256": sha256}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path,
                   default=Path("/scratch") / getpass.getuser() / "vla-vllm-profiling/assets")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--direct-cdn", action="store_true",
                   help="Bypass the process proxy only for the public us.aws.cdn.hf.co file host")
    args = p.parse_args()
    if platform.node().split(".")[0] in {"thor0", "fact-thor"}:
        p.error("Download on another host; Thor runtime remains offline")
    if args.direct_cdn:
        for key in ("NO_PROXY", "no_proxy"):
            os.environ[key] = ",".join(filter(None, [os.environ.get(key), "us.aws.cdn.hf.co"]))
    cache = args.cache_root.resolve()
    if not cache.is_relative_to("/scratch"):
        p.error("Download cache must be below /scratch")
    report = args.report.resolve()
    if not report.is_relative_to("/fact_data"):
        p.error("Provenance report must be below /fact_data")
    spec = json.loads((ROOT / "sources.lock.json").read_text())
    model = spec["checkpoint"]
    cache.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cache).free < 18 * 1024**3:
        raise RuntimeError("Require 18 GiB free scratch space")
    checkpoint = cache / "pi05_base" / model["revision"]
    tokenizer = cache / "paligemma_tokenizer"
    entries = []
    tok = spec["tokenizer"]
    entries.append(download(tok["url"], tokenizer / "tokenizer.model", tok["bytes"], tok["sha256"]))
    # Reconstruct only the text tokenizer configuration. The SentencePiece bytes
    # are identical to Google's pinned HF tokenizer.model; no model replacement.
    tokenizer_config = {
        "tokenizer_class": "GemmaTokenizer", "add_bos_token": True, "add_eos_token": False,
        "bos_token": "<bos>", "eos_token": "<eos>", "unk_token": "<unk>",
        "pad_token": "<pad>", "padding_side": "right", "model_max_length": 2048,
    }
    config_path = tokenizer / "tokenizer_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != tokenizer_config:
            raise RuntimeError("Existing tokenizer configuration differs; refusing to overwrite it")
    else:
        config_path.write_text(json.dumps(tokenizer_config, indent=2) + "\n")
    api = f"https://huggingface.co/api/models/{model['repo_id']}/revision/{model['revision']}?blobs=true"
    with urllib.request.urlopen(api, timeout=30) as response:
        metadata = json.load(response)
    if metadata["sha"] != model["revision"]:
        raise RuntimeError("Model revision changed")
    files = {item["rfilename"]: item for item in metadata["siblings"]}
    weights = files["model.safetensors"]
    if (weights["size"] != model["safetensors_file_bytes"]
            or weights["lfs"]["sha256"] != model["safetensors_file_sha256"]):
        raise RuntimeError("Checkpoint metadata differs from the pinned checksum/size")
    for name in ("config.json", "README.md", "policy_preprocessor.json", "policy_postprocessor.json", "model.safetensors"):
        info = files[name]
        lfs = info.get("lfs")
        url = f"https://huggingface.co/{model['repo_id']}/resolve/{model['revision']}/{name}?download=true"
        entries.append(download(url, checkpoint / name, info["size"],
                                lfs["sha256"] if lfs else info["blobId"], git_blob=not lfs))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"host": platform.node(), "completed_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint), "tokenizer": str(tokenizer), "files": entries,
        "tokenizer_config": "Locally reconstructed Gemma text-tokenizer settings; runtime parity check required",
        "tokenizer_config_sha256": digest(config_path),
        "checkpoint_revision": model["revision"], "transport": "resumable HTTPS via curl", "direct_cdn": args.direct_cdn}, indent=2) + "\n")
    print(f"Assets verified; report: {report}", flush=True)


if __name__ == "__main__":
    main()
