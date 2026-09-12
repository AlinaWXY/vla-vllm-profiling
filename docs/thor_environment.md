# Thor environment and script review — 2026-09-12

The SSH inspection used read-only version, metadata, permission and file queries.
It did not install packages, download weights, import Torch, initialize a model,
change system settings or start containers. The user has authorized direct GPU/CPU
execution on Thor; Slurm is still required on other hosts.

## Verified environment

| Item | Observation |
| --- | --- |
| Host | `ssh thor0` works; hostname `fact-thor` |
| Architecture / OS | aarch64, Ubuntu 24.04.4 LTS |
| GPU / driver | NVIDIA Thor, driver 595.78; separate memory capacity is reported as N/A |
| Host memory snapshot | 122 GiB total, approximately 119 GiB available; no swap |
| CUDA toolkit | 13.2, compiler V13.2.86; `/usr/local/cuda/bin/nvcc` |
| Nsight Compute | 2026.1.1.0; `/opt/nvidia/nsight-compute/2026.1.1/ncu` |
| Nsight Systems | Package 2025.6.3 installed; `/usr/local/bin/nsys` |
| Default Python | `/usr/bin/python3`, 3.12.3; NumPy 1.26.4 and Matplotlib 3.6.3 present |
| Default Python package metadata | No torch, torchvision, vllm, vllm-omni, triton, transformers, huggingface-hub, safetensors or lerobot distributions found in this interpreter |
| Shared source | Project `vllm/` and `vllm-omni/` files are readable directly from Thor through `/fact_data` |
| Shared Python checked | `/fact_data/xinyaowang/conda/envs/chia_env/bin/python3.10` is x86-64, incompatible with Thor |
| Docker | Current account lacks socket access and passwordless sudo; containers were not altered |
| Scheduler | No Slurm commands in the current PATH; direct execution allowed in `AGENTS.md` |

Missing from this interpreter does not mean absent from every environment on the
machine. Shared framework source can be reused, but does not supply an ARM Torch
binary or its runtime dependencies. No broad filesystem or home-directory scan was
used to look for another environment.

### Follow-up: what is inspectable without sudo

A second read-only pass checked known system installation directories, system and
login startup configuration files, the current user's Conda registry, selected
process names, and Docker service/storage configuration. No shell configuration
was executed, and no home directory tree was enumerated.

- The current user's Conda registry lists only the already checked x86-64
  `chia_env`. System/login configuration did not reveal another Python runtime.
- Python's user site is enabled, so the default-interpreter package check also
  covered packages exposed through that user site.
- Docker and containerd daemons are present. No Python/vLLM inference worker was
  visible in the process-name check; that does not establish whether an offline
  environment or stopped container is installed.
- Docker uses `/var/lib/docker`. Both its management socket and storage directory
  are inaccessible to this account, so container/image contents cannot be listed.
- Exact startup configuration files for other local login accounts were also
  unreadable. No permission changes or privilege escalation were attempted in
  this follow-up.

The conclusion is **partial visibility, not proof that PyTorch/vLLM is absent**.
Any readable existing environment can be checked by its exact interpreter or
package path without installation. Private container or account environments
require their owner to provide a usable path/access or an inventory.

## Script changes

| Finding | Change |
| --- | --- |
| `--profile` repeated safe/optimized and expert latency loops under NCU | Skip these loops; retain optimized warmup and one scoped expert invocation |
| NCU run captured an additional expert timing graph | Capture it only in the uninstrumented benchmark |
| Checkpoint loader assembled all tensors into a state dictionary/list | Stream safetensors one tensor at a time through the existing loader |
| Remote tokenizer IDs could trigger a download | Require a local directory, offline flags and `local_files_only=True` |
| Cache setup replaced existing paths and created unused pip/uv caches | Preserve configured workspace caches, default only unset paths, remove pip/uv setup |
| Checkpoint utility downloaded missing files | Replace with `resolve_checkpoint.py`, which resolves already cached data only |
| Model loading lacked a memory preflight | Check host and visible cgroup v2 memory headroom; minimum 32 GiB by default |
| NCU was missed because it was outside PATH | Inventory known vendor paths; use the existing absolute executable path |
| All hosts required Slurm | Direct execution on `thor0` / `fact-thor`; Slurm guard elsewhere |

No upstream submodule was modified. Streaming still creates the model parameters;
the upstream optimized pipeline still compiles/captures production graphs during
warmup. The 32 GiB check is not an allocation cap or an OOM guarantee. GPU peak
counters exclude host allocations and some external allocators; actual model/NCU
memory peaks have not been measured.

## Remaining upstream redundancy

PR 4419's `Pi05RealtimePrefixEncoder._pack_weights()` allocates separate
`ffn_gate_w` and `ffn_up_w` buffers plus a combined `ffn_gate_up_w` buffer.
The default packed Torch MLP path uses the combined buffer. At 18 layers, width
2048, MLP width 16384 and BF16, the unused separate buffers total:

```text
2 × 18 × 2048 × 16384 × 2 bytes = 2.25 GiB
```

This is a source-derived estimate, not a measured peak. Alternative prefix MLP
paths use different buffers. Removing allocations needs a targeted upstream
change and correctness testing, so it was not done in this review. NCU replay,
packed weights, compiled kernels and graph pools remain real memory consumers.
Calibration is optional and can be reused for the same hardware/power settings.

Validation: all 16 CPU unit tests passed on Thor's existing Python 3.12.3; the
direct-execution guard and host/cgroup memory query also succeeded on the actual
host. No Torch/CUDA model or kernel benchmark was executed. The full read-only
inventory is retained locally under `results/raw/thor_inventory_20260912.txt`.
