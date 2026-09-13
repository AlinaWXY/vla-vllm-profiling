#!/usr/bin/env python3
"""Read-only memory samples for one process; stop when it exits or its PID is reused."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    proc = Path("/proc") / str(args.pid)
    def identity():
        # The command name may contain spaces; starttime is field 22.
        return (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
    initial = identity()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utc", "pid", "rss_bytes", "process_hwm_bytes", "host_available_bytes"])
        while True:
            try:
                if identity() != initial:
                    break
                status = dict(line.split(":", 1) for line in (proc / "status").read_text().splitlines())
                mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                if "VmRSS" not in status:  # exited/zombie
                    break
                writer.writerow([datetime.now(timezone.utc).isoformat(), args.pid,
                    int(status["VmRSS"].split()[0]) * 1024, int(status["VmHWM"].split()[0]) * 1024,
                    int(mem["MemAvailable"].split()[0]) * 1024])
                f.flush()
            except (FileNotFoundError, ProcessLookupError):
                break
            time.sleep(10)


if __name__ == "__main__":
    main()
