#!/bin/bash
set -e
uv run ruff format src/innovate/advanced_runtime.py
uv run ruff check src/innovate/advanced_runtime.py
uv run ruff format tests/unit/test_calibrate_prediction_intervals.py
uv run ruff check tests/unit/test_calibrate_prediction_intervals.py
uv run pytest tests/unit/test_calibrate_prediction_intervals.py
