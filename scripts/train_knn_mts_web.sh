#!/usr/bin/env bash
set -euo pipefail
# Run selected RQ1 conditions through the shared model runner.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
training_python="${SHARP_PYTHON:-python}"
export PYTHONUNBUFFERED=1
exec "$training_python" -u "$script_dir/train_rq1.py" \
  --framework knn_mts --suite web "$@"
