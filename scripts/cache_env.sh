#!/usr/bin/env bash
# Source after selecting the EXISTING runtime. Never install or download here.
# Preserve configured caches; provide scratch defaults only for unset paths.
vla_cache_environment() {
    local scratch_root="${VLA_SCRATCH_ROOT:-/scratch/$(id -un)/vla-vllm-profiling}"
    local key canonical
    case "$scratch_root" in
        /scratch/*) ;;
        *) echo 'VLA_SCRATCH_ROOT must be below /scratch.' >&2; return 1 ;;
    esac
    local -A paths=(
        [TMPDIR]="${TMPDIR:-$scratch_root/tmp}"
        [HF_HOME]="${HF_HOME:-$scratch_root/huggingface}"
        [HF_HUB_CACHE]="${HF_HUB_CACHE:-${HF_HOME:-$scratch_root/huggingface}/hub}"
        [XDG_CACHE_HOME]="${XDG_CACHE_HOME:-$scratch_root/xdg}"
        [TRITON_CACHE_DIR]="${TRITON_CACHE_DIR:-$scratch_root/triton}"
        [TORCH_HOME]="${TORCH_HOME:-$scratch_root/torch}"
        [TORCHINDUCTOR_CACHE_DIR]="${TORCHINDUCTOR_CACHE_DIR:-$scratch_root/inductor}"
        [MPLCONFIGDIR]="${MPLCONFIGDIR:-$scratch_root/matplotlib}"
    )
    # Validate every destination before creating any directory or changing paths.
    for key in "${!paths[@]}"; do
        canonical=$(realpath -m -- "${paths[$key]}") || return 1
        case "$canonical" in
            /scratch/*|/fact_data/*) ;;
            *) echo "$key points outside /scratch or /fact_data; select a workspace cache explicitly." >&2; return 1 ;;
        esac
    done
    for key in "${!paths[@]}"; do
        mkdir -p -- "${paths[$key]}" || return 1
        export "$key=${paths[$key]}"
    done
}
vla_cache_environment || { unset -f vla_cache_environment; return 1 2>/dev/null || exit 1; }
unset -f vla_cache_environment
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export MAX_JOBS="${MAX_JOBS:-1}"
export TOKENIZERS_PARALLELISM=false
