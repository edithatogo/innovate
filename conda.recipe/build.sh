#!/bin/bash
set -euo pipefail
"$PYTHON" scripts/check_migration.py --release
"$PYTHON" -m pip install . --no-deps --no-build-isolation -vv
