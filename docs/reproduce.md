# Reproduce and audit the retained figures

## CPU analysis

Use an isolated Python environment with `requirements/analysis.txt`. Approved PNG/PDF files are retained byte for byte. Run from the repository root:

```bash
python -m profiling.verify_results
output_dir=$(python -m profiling.experiments --purpose 'Replot retained SGLang L20 DRAM roofline')
python -m profiling.replot --source results/processed/0914/16 --output "$output_dir"
```

Use `0914/15` for Omni. A replot gets a new experiment ID and timestamp while preserving collection times and measurements, so its image hash will differ. New local experiments are excluded from publication by `.gitignore`; the current allowlist contains only the two approved results.

The verifier checks packaged SHA256 records, decompresses NCU exports in memory, checks NVTX labels, matches per-kernel times/FLOPs/DRAM counts, recomputes every operator aggregate, validates the known-work GEMM and checks all replay passes against saved model/input/action evidence. It needs no GPU, checkpoint or Nsight installation.

## Evidence layout

- `operators.csv`, `kernels.csv`, `summary.json`: measurements used by the approved figure.
- `capture/ncu/`: raw and NVTX-renamed CSV exports (`.csv.gz`) and collection metadata.
- `capture/baseline/`, `capture/workload/`, `capture/passes/`: independent baseline and replay evidence.
- `capture/compute_ncu/`, `capture/ceilings.json`: counter validation and measured reference roofs.
- `capture/run.slurm`, `capture/harness/profiling/`: original successful job script and required profiling modules, preserved byte for byte.
- `publication.json`: original paths, published relative paths and SHA256 hashes. Historic absolute paths in original JSON records describe the collection host; the CPU verifier and replotter use bundled relative paths.
- `render_source/`, `provenance/`: original figure source for provenance. The supported portable entry point is `python -m profiling.replot`.

For four SGLang CUB scan kernels, the CLI's innermost NVTX label came from CCCL's separate domain. `nvtx_parent_audit.json` records the actual default-domain parent recovered from the NCU report; attribution was not inferred from neighboring kernels.

Large `.ncu-rep` files remain in the local archive; hashes and locations are in `publication.json`. Published CSV exports suffice for the checks above. Opening the original Nsight report requires access to the local archive.

## GPU collection provenance

| Implementation | Slurm job | CUDA runtime | PyTorch | Transformers |
| --- | --- | --- | --- | --- |
| Omni `6bdbf97` | 1506597 | 13.0 | 2.11.0+cu130 | 5.8.1 |
| SGLang `5200508` | 1506879 | 13.0 | 2.13.0+cu130 | 5.12.1 |

Each job requested one L20, eight CPUs and 64 GiB RAM in the local `dev` partition, using Nsight Compute 2025.1.1. Archived Slurm scripts retain the original cluster paths and environment setup; they are provenance snapshots, not portable installers. A new capture needs new output paths, the pinned source/environment versions and identical checkpoint/tokenizer. Submit through Slurm after adapting paths; never rerun an archived script into an existing result directory.

Pinned source is available through the `vllm`, `vllm-omni` and `sglang` submodules. `python scripts/bootstrap_sources.py` checks or fetches locked revisions without replacing a dirty checkout. Model sources/checksums are in `sources.lock.json`; `scripts/download_assets.py` retrieves them into `/scratch` with an explicit provenance report under `/fact_data`. Weights are not committed.
