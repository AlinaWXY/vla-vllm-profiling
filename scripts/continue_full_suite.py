"""Resume prepared Thor captures only after baseline and known-work probe succeed.

This is a low-CPU controller on the project host. It installs nothing, never
stops another job, and runs the two authorized GPU captures sequentially.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from profiling.experiments import timestamp
from profiling.roofline import load_ncu, metric_value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--baseline-pid", type=int, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--kernels", type=Path, required=True)
    p.add_argument("--ranges", type=Path, required=True)
    args = p.parse_args()
    status_file = args.kernels / "suite.json"
    status = {"started_at":timestamp(), "baseline":str(args.baseline), "probe":str(args.probe),
              "kernels":str(args.kernels), "ranges":str(args.ranges)}
    def update(state, **changes):
        status.update(state=state, updated_at=timestamp(), **changes)
        status_file.write_text(json.dumps(status,indent=2)+"\n")
        print(timestamp(), state, changes, flush=True)
    def remote(cmd, check=False):
        return subprocess.run(["ssh","thor0",cmd],cwd=ROOT,text=True,capture_output=True,check=check)
    try:
        update("waiting_for_baseline")
        while not (args.baseline/"workload/full_scopes.json").exists():
            alive = remote(f"kill -0 {args.baseline_pid}")
            if alive.returncode:
                raise RuntimeError("Baseline process exited without full_scopes.json; inspect run.log")
            time.sleep(30)
        base = json.loads((args.baseline/"workload/full_scopes.json").read_text())
        if base["scoped_direct_vs_pipeline_max_abs"] > 1e-5 or base["scoped_graph_vs_pipeline_max_abs"] > 1e-5:
            raise ValueError("Decomposed workload does not match selected framework")
        if not all(base["input_audit"]["camera_masks"]):
            raise ValueError("Missing effective camera")
        probe = load_ncu(args.probe/"range_probe/raw.csv")
        expected = json.loads((args.probe/"range_probe/workload.json").read_text())
        if len(probe) != 1 or metric_value(probe[0], "sm__ops_path_tensor_src_bf16_dst_fp32.sum", "count") != expected["expected_bf16_tensor_flops"]:
            raise ValueError("Whole-range Tensor counter failed known-work check")
        for name, directory in (("kernels",args.kernels),("ranges",args.ranges)):
            update("waiting_for_idle_gpu", next_capture=name)
            quiet = 0
            while quiet < 2:
                apps = remote("nvidia-smi --query-compute-apps=pid --format=csv,noheader",check=True).stdout.strip()
                quiet = quiet + 1 if not apps else 0
                time.sleep(15)
            update("capturing_"+name)
            with (directory/"run.log").open("w") as log:
                subprocess.run(["ssh","thor0","cd "+str(ROOT)+" && bash "+str(directory/"run.sh")],
                               stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
            if not (directory/"workload/full_scopes.json").exists():
                raise RuntimeError(name+" capture did not save its workload manifest")
        update("captures_completed")
    except BaseException as error:
        update("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
