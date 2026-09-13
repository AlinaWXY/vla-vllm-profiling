#!/usr/bin/env python3
"""Serialize one authorized Thor experiment after other GPU/profiling jobs exit."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    if socket.gethostname().split(".")[0] not in ("thor0", "fact-thor"):
        raise RuntimeError("This queue is only for the user-authorized direct Thor execution")
    queue_file = args.run_dir / "queue.json"
    state = {"queued_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
             "status": "waiting_for_idle_gpu"}
    with queue_file.open("x") as handle:
        json.dump(state, handle, indent=2)
    def active():
        gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             check=True, capture_output=True, text=True).stdout.strip()
        processes = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "args="],
                                   check=True, capture_output=True, text=True).stdout
        jobs = [line for line in processes.splitlines() if " -m profiling." in line]
        return bool(gpu or jobs)
    idle_since = None
    while True:
        if active():
            idle_since = None
        else:
            idle_since = idle_since or time.monotonic()
            if time.monotonic() - idle_since >= 30:
                break
        time.sleep(15)
    state.update(status="running", started_at=datetime.now(timezone.utc).isoformat())
    queue_file.write_text(json.dumps(state, indent=2) + "\n")
    with (args.run_dir / "run.log").open("x") as log:
        result = subprocess.run(["bash", str(args.run_dir / "run.sh")], stdout=log, stderr=subprocess.STDOUT)
    state.update(status="completed" if result.returncode == 0 else "failed",
                 exit_code=result.returncode, finished_at=datetime.now(timezone.utc).isoformat())
    queue_file.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
