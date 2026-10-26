#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
.venv/bin/python baseline.py --split dev --limit 3
