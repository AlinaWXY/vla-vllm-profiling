"""Allocate dated experiment directories and preserve collection/figure provenance."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import socket
from zoneinfo import ZoneInfo


def timestamp():
    return datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds")


def file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def allocate(root, purpose):
    created = timestamp()
    day = datetime.fromisoformat(created).strftime("%m%d")
    parent = Path(root) / day
    parent.mkdir(parents=True, exist_ok=True)
    for index in range(10000):
        output = parent / f"{index:02d}"
        try:
            output.mkdir()
        except FileExistsError:
            continue
        record = {"experiment_id": f"{day}/{index:02d}", "created_at": created,
                  "timezone": "Asia/Hong_Kong", "purpose": purpose, "host": socket.gethostname()}
        (output / "experiment.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        return output
    raise RuntimeError("Daily experiment sequence exhausted")


def figure_record(output, inputs=()):
    """Refuse overwriting a prior figure and create its dated provenance sidecar."""
    output = Path(output)
    for suffix in (".png", ".pdf", ".plot.json"):
        if Path(str(output) + suffix).exists():
            raise FileExistsError(f"Allocate a new experiment before replotting: {output}")
    experiment = next((p / "experiment.json" for p in output.parents
                       if (p / "experiment.json").is_file()), None)
    if experiment is None:
        raise ValueError("Figure output needs a dated experiment directory; run profiling.experiments first")
    parent = json.loads(experiment.read_text())
    record = {"experiment_id": parent["experiment_id"], "plotted_at": timestamp(),
              "inputs": [file_record(p) for p in inputs], "host": socket.gethostname()}
    output.parent.mkdir(parents=True, exist_ok=True)
    Path(str(output) + ".plot.json").write_text(json.dumps(record, indent=2) + "\n")
    return f"{record['experiment_id']} | plotted {record['plotted_at']}"


def clock_snapshot():
    result = {"observed_at": timestamp(), "devfreq": {}}
    root = Path("/sys/class/devfreq")
    if root.is_dir():
        for device in sorted(root.iterdir()):
            if "gpu" in device.name:
                result["devfreq"][device.name] = {
                    key: (device / key).read_text().strip()
                    for key in ("cur_freq", "min_freq", "max_freq", "governor")
                    if (device / key).is_file()}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/processed"))
    p.add_argument("--purpose", required=True)
    args = p.parse_args()
    print(allocate(args.root, args.purpose))


if __name__ == "__main__":
    main()
