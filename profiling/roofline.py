"""Parse NCU long or wide raw CSV without treating missing counters as zero.

Each invocation stays separate. Precision domains stay separate as well: a BF16
Tensor operation and an FP32 instruction do not share a hardware compute roof.
"""
from __future__ import annotations

import csv
import gzip
import io
import math
from pathlib import Path


def number(value):
    try:
        value = float(str(value).replace(",", "").strip())
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def load_ncu(path):
    # NCU may prepend connection/progress messages and repeat its CSV header.
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        lines = handle.read().splitlines()
    start = next((i for i, line in enumerate(lines)
                  if ("Metric Name" in line and "Metric Value" in line
                      or "Kernel Name" in line and "gpu__" in line)
                  and "ID" in next(csv.reader([line]), [])), None)
    if start is None:
        raise ValueError("No NCU raw CSV header; export with --page raw --csv.")
    kernels = {}
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    wide = "Metric Name" not in reader.fieldnames
    units = None
    for row in reader:
        if row.get("ID") == "ID":
            continue
        if not row.get("ID"):
            if wide:
                units = row
            continue
        if not wide and not row.get("Metric Name"):
            continue
        if wide and units is None:
            raise ValueError("Wide NCU CSV is missing its units row.")
        key = (row.get("Process ID", ""), row.get("Device", ""),
               row.get("Context", ""), row.get("Stream", ""), row["ID"])
        entry = kernels.setdefault(key, {
            "id": row["ID"], "process_id": key[0], "device": key[1],
            "context": key[2], "stream": key[3],
            "kernel": row.get("Kernel Name", ""),
            "grid": row.get("Grid Size", ""), "block": row.get("Block Size", ""),
            "metrics": {},
        })
        values = ((name, row[name], units.get(name, ""))
                  for name in reader.fieldnames if "__" in name) if wide else [
                      (row["Metric Name"], row["Metric Value"], row.get("Metric Unit"))]
        for name, value, unit in values:
            metric = (number(value), (unit or "").strip())
            old = entry["metrics"].get(name)
            if old is not None and old != metric:
                raise ValueError(f"Conflicting duplicate metric {name} for kernel {key}")
            entry["metrics"][name] = metric
    if not kernels:
        raise ValueError("The report contains no kernel metrics.")
    return list(kernels.values())


def metric_value(kernel, name, kind):
    if name not in kernel["metrics"]:
        return None
    value, unit = kernel["metrics"][name]
    if value is None or value < 0:
        return None
    scales = {
        "time": {"second": 1, "s": 1, "msecond": 1e-3, "ms": 1e-3,
                 "usecond": 1e-6, "us": 1e-6, "nsecond": 1e-9, "ns": 1e-9},
        "bytes": {"byte": 1, "bytes": 1, "Kbyte": 1e3, "Mbyte": 1e6,
                  "Gbyte": 1e9, "KiB": 1024, "MiB": 1024**2},
        "count": {"": 1, "inst": 1, "instruction": 1, "instructions": 1,
                  "op": 1, "ops": 1, "sector": 1, "sectors": 1},
    }
    if unit not in scales[kind]:
        raise ValueError(f"Unsupported {kind} unit {unit!r} for {name}; use --print-units base")
    return value * scales[kind][unit]


def analyze(kernels, contract):
    if not contract.get("domains"):
        raise ValueError("At least one explicitly defined precision domain is required.")
    for domain, spec in contract["domains"].items():
        terms = spec.get("terms", [])
        if not terms or len({t["metric"] for t in terms}) != len(terms):
            raise ValueError(f"Empty or duplicate FLOP terms in {domain}")
        if any(number(t["weight"]) is None or t["weight"] <= 0
               or not t["metric"].endswith(".sum") for t in terms):
            raise ValueError(f"Use positive weights and absolute .sum counters in {domain}")
    memory = contract.get("memory", {"level": "dram"})
    level = memory["level"]
    if level not in {"dram", "l2"}:
        raise ValueError("Memory level must be dram or l2.")
    traffic_metrics = memory.get("metrics")
    if traffic_metrics is not None and (not traffic_metrics
            or len(set(traffic_metrics)) != len(traffic_metrics)
            or any(not metric.endswith(".sum") for metric in traffic_metrics)):
        raise ValueError("Memory metrics must be unique absolute .sum counters.")
    if level == "l2" and traffic_metrics != ["lts__t_bytes.sum"]:
        raise ValueError("L2 roof requires the reviewed lts__t_bytes.sum metric.")
    if level == "dram" and traffic_metrics and any(not m.startswith("dram__") for m in traffic_metrics):
        raise ValueError("L2 traffic must not be labeled DRAM.")
    rows = []
    for kernel in kernels:
        duration = metric_value(kernel, "gpu__time_duration.sum", "time")
        traffic = metric_value(kernel, "dram__bytes.sum", "bytes")
        if traffic_metrics:
            values = [metric_value(kernel, name, "bytes") for name in traffic_metrics]
            traffic = sum(values) if all(value is not None for value in values) else None
        elif traffic is None:
            read = metric_value(kernel, "dram__bytes_read.sum", "bytes")
            write = metric_value(kernel, "dram__bytes_write.sum", "bytes")
            if read is not None and write is not None:
                traffic = read + write
        for domain, spec in contract["domains"].items():
            missing = []
            flops = 0.0
            for term in spec["terms"]:
                count = metric_value(kernel, term["metric"], "count")
                if count is None:
                    missing.append(term["metric"])
                else:
                    flops += count * term["weight"]
            issues = []
            if duration is None or duration <= 0:
                issues.append("missing_or_nonpositive_duration")
            if traffic is None or traffic <= 0:
                issues.append(f"missing_or_zero_{level}_traffic")
            if missing:
                issues.append("missing_flop_metrics:" + ";".join(missing))
            if not missing and flops == 0:
                issues.append("zero_counted_flops")
            complete_flops = None if missing else flops
            rows.append({
                **{k: v for k, v in kernel.items() if k != "metrics"},
                "domain": domain, "duration_ns": None if duration is None else duration * 1e9,
                "memory_level": level, "memory_bytes": traffic,
                "dram_bytes": traffic if level == "dram" else None, "flops": complete_flops,
                "ai_flops_per_byte": flops / traffic if not issues else None,
                "performance_tflops": flops / duration / 1e12 if not issues else None,
                "status": "ok" if not issues else "|".join(issues),
            })
    return rows


def attach_operators(kernels, annotated):
    def identity(k):
        return tuple(k.get(name, "") for name in ("process_id", "device", "context", "stream", "id"))
    labels = {identity(k): k for k in annotated}
    if set(labels) != {identity(k) for k in kernels}:
        raise ValueError("Operator CSV must be exported from the exact same report (kernel identities differ).")
    for kernel in kernels:
        label = labels[identity(kernel)]
        if label["metrics"] != kernel["metrics"]:
            raise ValueError("Operator CSV metrics differ; reports cannot be joined safely.")
        kernel["operator"] = label["kernel"]
