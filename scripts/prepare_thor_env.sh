#!/usr/bin/env bash
# Prepare an ARM64 Python 3.12 environment OFF Thor, using binary wheels.
# Small pure-Python wheel packaging only; no CUDA context or native compilation.
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
case "$(hostname -s)" in
  thor0|fact-thor) echo 'Prepare this environment on another host, not Thor.' >&2; exit 1 ;;
esac
case "$project_dir/" in /fact_data/*/) ;; *) echo 'Project must reside in /fact_data.' >&2; exit 1 ;; esac
build_cache="/scratch/$(id -un)/vla-vllm-profiling"
runtime_dir="$project_dir/.venv-thor-cu130"
requirements_file="$project_dir/requirements/thor-cu130.lock"
[[ -f "$requirements_file" ]] || requirements_file="$project_dir/requirements/thor-cu130.in"
if [[ -e "$runtime_dir/READY" ]]; then
  echo "Existing completed environment: $runtime_dir; refusing to modify it." >&2
  exit 1
fi
mkdir -p "$build_cache/build-tmp" "$build_cache/uv-cache" "$build_cache/wheelhouse"
export TMPDIR="$build_cache/build-tmp" UV_CACHE_DIR="$build_cache/uv-cache"
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_INSTALLS=1 RAYON_NUM_THREADS=1
export UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
uv_bin="$build_cache/host-tools/bin/uv"
if [[ ! -x "$uv_bin" ]]; then
  python3 -m pip --isolated install --no-cache-dir --disable-pip-version-check \
    --no-compile --only-binary=:all: --no-deps \
    --target "$build_cache/host-tools" uv==0.12.13
fi
if [[ ! -f "$build_cache/wheelhouse/antlr4_python3_runtime-4.9.3-py3-none-any.whl" ]]; then
  python3 -m pip --isolated wheel --use-pep517 --no-cache-dir \
    --disable-pip-version-check --no-deps --wheel-dir "$build_cache/wheelhouse" \
    antlr4-python3-runtime==4.9.3
fi
"$uv_bin" --no-config pip install --python-version 3.12 \
  --python-platform aarch64-manylinux_2_39 \
  --target "$runtime_dir/lib/python3.12/site-packages" \
  --only-binary=:all: --find-links "$build_cache/wheelhouse" \
  --torch-backend cu130 -r "$requirements_file" "$@"
# A plan must not create a runnable/ready environment.
for arg in "$@"; do [[ "$arg" != --dry-run ]] || exit 0; done
# The interpreter is supplied by Thor's existing OS. Do not copy x86 Python.
mkdir -p "$runtime_dir/bin"
[[ -L "$runtime_dir/bin/python" && $(readlink "$runtime_dir/bin/python") == /usr/bin/python3.12 ]] || \
  ln -s /usr/bin/python3.12 "$runtime_dir/bin/python"
[[ -L "$runtime_dir/bin/python3" && $(readlink "$runtime_dir/bin/python3") == python ]] || \
  ln -s python "$runtime_dir/bin/python3"
cat > "$runtime_dir/pyvenv.cfg" <<'EOF'
home = /usr/bin
include-system-site-packages = false
version = 3.12.3
EOF
python3 -B "$project_dir/scripts/audit_thor_env.py"
echo "Packages prepared at $runtime_dir. Thor validation still required."
