#!/usr/bin/env bash
# Download assets OFF Thor using existing Python and curl; install nothing.
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
case "$(hostname -s)" in
  thor0|fact-thor) echo 'Download assets on another host, not Thor.' >&2; exit 1 ;;
esac
download_python="${VLA_DOWNLOAD_PYTHON:-python3}"
"$download_python" -c 'import sys; assert sys.version_info >= (3, 9), "Select existing Python >= 3.9 using VLA_DOWNLOAD_PYTHON"'
exec "$download_python" -u "$project_dir/scripts/download_assets.py" \
  --report "$project_dir/results/processed/pi05_assets/download.json" "$@"
