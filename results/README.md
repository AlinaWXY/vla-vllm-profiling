# Results

No VLA hardware measurements yet. `processed/environment_sm110a/` contains a real
Thor deployment/NCU probe for a 1024-element toy kernel compiled off-host, runtime
import/operation checks and provisional GEMM/L2 calibration. This material is
explicitly separate from pi0.5 measurements and must not be used as a VLA result.

- `raw/`: environment inventory, uninstrumented benchmark JSON/NPZ, NCU reports,
  CSV exports, metric contracts and calibration samples (ignored by Git).
- `processed/`: reviewed per-kernel tables, summary JSON, measured figures and the
  interpretation to publish after actual Thor runs.

Retain the raw data under `/fact_data` and publish large reports as release assets
with checksums after actual Thor measurements are available.
