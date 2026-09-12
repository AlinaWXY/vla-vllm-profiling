"""Run a tiny externally assembled cubin on Thor with the existing CUDA driver.

Uses Python stdlib only. This checks cross-host deployment, not VLA performance.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import platform

from profiling.ncu import require_gpu_execution
from profiling.runtime import require_headroom


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cubin", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require_gpu_execution()
    require_headroom(1)
    if args.output.exists():
        raise FileExistsError(args.output)
    driver = C.CDLL("libcuda.so.1")

    def call(name, types, *values):
        function = getattr(driver, name)
        function.argtypes = types
        function.restype = C.c_int
        result = function(*values)
        if result:
            raise RuntimeError(f"{name}: CUDA error {result}")

    device = C.c_int()
    call("cuInit", [C.c_uint], 0)
    call("cuDeviceGet", [C.POINTER(C.c_int), C.c_int], C.byref(device), 0)
    name = C.create_string_buffer(128)
    call("cuDeviceGetName", [C.c_void_p, C.c_int, C.c_int], name, len(name), device)
    context, module, function = C.c_void_p(), C.c_void_p(), C.c_void_p()
    pointer = C.c_uint64()
    call("cuDevicePrimaryCtxRetain", [C.POINTER(C.c_void_p), C.c_int], C.byref(context), device)
    try:
        call("cuCtxSetCurrent", [C.c_void_p], context)
        call("cuModuleLoad", [C.POINTER(C.c_void_p), C.c_char_p], C.byref(module), str(args.cubin.resolve()).encode())
        call("cuModuleGetFunction", [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p], C.byref(function), module, b"write_indices")
        count = C.c_uint(1024)
        output = (C.c_float * count.value)()
        call("cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t], C.byref(pointer), C.sizeof(output))
        parameters = (C.c_void_p * 2)(C.addressof(pointer), C.addressof(count))
        call("cuLaunchKernel", [C.c_void_p] + [C.c_uint] * 7 + [C.c_void_p] * 3,
             function, 4, 1, 1, 256, 1, 1, 0, None, parameters, None)
        call("cuCtxSynchronize", [])
        call("cuMemcpyDtoH_v2", [C.c_void_p, C.c_uint64, C.c_size_t], output, pointer, C.sizeof(output))
        assert list(output) == [float(i + 1) for i in range(count.value)]
        report = {"status": "passed", "test": "external sm_110a cubin, 1024 float outputs",
                  "gpu": name.value.decode(), "host_arch": platform.machine(),
                  "cubin_sha256": hashlib.sha256(args.cubin.read_bytes()).hexdigest(),
                  "device_allocation_bytes": C.sizeof(output), "vla_benchmark": False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if pointer.value:
            call("cuMemFree_v2", [C.c_uint64], pointer)
        if module.value:
            call("cuModuleUnload", [C.c_void_p], module)
        call("cuDevicePrimaryCtxRelease_v2", [C.c_int], device)


if __name__ == "__main__":
    main()
