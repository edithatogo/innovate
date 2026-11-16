"""V02 adversarial leakage and denominator tests for BenchmarkProtocol."""

from __future__ import annotations

import pytest

from validation.benchmarks.protocol import BenchmarkProtocol


def test_leakage_duplicate_holdouts_and_failed_fits_are_handled_honestly():
    proto = BenchmarkProtocol(
        vintage="2026-11-16",
        origin_index=6,
        horizon=4,
        seed=42,
        series=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 40.0, 45.0, 48.0, 50.0),
    )

    with pytest.raises(ValueError, match="leak"):
        proto.validate_split([0, 1, 6], [7, 8])

    with pytest.raises(ValueError, match="duplicate"):
        proto.validate_split([0, 1, 2], [6, 6, 7])

    ok = proto.run(lambda train, h, seed: [train[-1]] * h)
    assert ok["status"] == "ok"
    assert ok["seed"] == 42
    assert isinstance(ok["rmse"], float)

    failed = proto.run(lambda _train, _h, _seed: None)
    assert failed["status"] == "failed"
    assert failed["rmse"] is None
