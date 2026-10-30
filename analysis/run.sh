#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${ED_PYTHON:-python3}
mkdir -p output
"$PYTHON" analysis/prepare_data.py > output/data_log.txt
"$PYTHON" -u analysis/analysis.py > output/analysis_log.txt
