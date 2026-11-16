"""V04 accuracy-controlled performance receipt validation tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def validate_performance_receipt(receipt: dict[str, Any], max_rel_error: float) -> list[str]:
    errors: list[str] = []
    if not receipt.get("workload"):
        errors.append("missing workload id")
    elapsed = receipt.get("elapsed_ns")
    if not isinstance(elapsed, int) or elapsed <= 0:
        errors.append("elapsed_ns must be a positive integer (fake/metadata-only execution rejected)")
    err = receipt.get("max_abs_error")
    if not isinstance(err, (int, float)) or err > max_rel_error:
        errors.append("accuracy gate failed")
    return errors


def test_real_bench_receipt_passes_and_fake_receipt_fails():
    protocol = json.loads(Path("validation/performance/protocol.json").read_text(encoding="utf-8"))
    proc = subprocess.run(
        ["cargo", "bench", "--locked", "-p", "innovate-core", "--bench", "workloads"],
        capture_output=True,
        text=True,
        check=True,
    )
    last_line = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")][-1]
    receipt = json.loads(last_line)
    assert validate_performance_receipt(receipt, protocol["workloads"][0]["max_rel_error"]) == []

    fake = {"workload": "bass_predict_100", "elapsed_ns": 0, "max_abs_error": 0.0}
    assert validate_performance_receipt(fake, 1e-10)
