#!/usr/bin/env bash
# Run the shared ARM64 packages using Thor's existing native interpreter.
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
case "$(hostname -s):$(uname -m)" in
  thor0:aarch64|fact-thor:aarch64) ;;
  *) echo 'This runtime is for the authorized ARM64 Thor host.' >&2; exit 1 ;;
esac
runtime_dir="$project_dir/.venv-thor-cu130"
if [[ ! -f "$runtime_dir/pyvenv.cfg" || ! -x "$runtime_dir/bin/python" ]]; then
  echo 'Shared environment is not prepared. Run prepare_thor_env.sh on another host.' >&2
  exit 1
fi
source "$project_dir/scripts/cache_env.sh"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
omni_source_dir="${PI05_OMNI_SOURCE:-$project_dir/vllm-omni}"
test -f "$omni_source_dir/vllm_omni/diffusion/models/pi05/modeling_pi05.py"
export PYTHONPATH="$omni_source_dir:${PI05_PROFILE_SOURCE:-$project_dir}:$project_dir"
export CUDA_HOME=/usr/local/cuda
export PATH="$runtime_dir/bin:$CUDA_HOME/bin:$PATH"
# These settings control new extension/JIT compilation, not prebuilt wheel code.
export TORCH_CUDA_ARCH_LIST=11.0a
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MAX_JOBS=1 TORCHINDUCTOR_COMPILE_THREADS=1
exec "$runtime_dir/bin/python" "$@"
