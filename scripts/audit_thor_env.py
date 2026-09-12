"""Inspect shared ARM64 packages without executing their code on the build host."""
import importlib.metadata
import json
from pathlib import Path
import platform
import struct


def main():
    project = Path(__file__).resolve().parents[1]
    site = project / ".venv-thor-cu130/lib/python3.12/site-packages"
    packages = []
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        tags = [line[5:] for line in (distribution.read_text("WHEEL") or "").splitlines()
                if line.startswith("Tag: ")]
        if not tags or any("x86" in tag for tag in tags):
            raise ValueError(f"Unexpected wheel platform: {distribution.metadata['Name']}: {tags}")
        packages.append({"name": distribution.metadata["Name"],
                         "version": distribution.version, "wheel_tags": tags})
    versions = {p["name"].lower().replace("_", "-"): p["version"] for p in packages}
    for line in (project / "requirements/thor-cu130.lock").read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==", 1)
            if versions.get(name) != version:
                raise ValueError(f"Installed version differs from lock: {line}")
    if versions.get("vllm") != "0.22.0":
        raise ValueError("Expected the pinned vLLM 0.22.0 wheel")
    native = {}
    for relative in ("torch/_C.cpython-312-aarch64-linux-gnu.so", "triton/_C/libtriton.so", "vllm/_C.abi3.so"):
        with (site / relative).open("rb") as f:
            header = f.read(64)
        if header[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", header, 18)[0] != 183:
            raise ValueError(f"Expected little-endian 64-bit AArch64 ELF: {relative}")
        native[relative] = "ELF64 AArch64"
    report = {"preparation_host": platform.node(), "preparation_cpu": platform.machine(),
              "target_cpu": "aarch64", "target_python": "3.12", "gpu_jit_target": "sm_110a",
              "package_count": len(packages), "native_extensions": native,
              "packages": sorted(packages, key=lambda p: p["name"].lower()),
              "note": "Metadata/ELF checks only; see separate Thor runtime validation."}
    output = project / "results/processed/environment_sm110a/packages.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(packages)} packages and ARM64 Torch/Triton/vLLM extensions: {output}")


if __name__ == "__main__":
    main()
