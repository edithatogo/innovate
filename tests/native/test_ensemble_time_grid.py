"""Installed Python calls must retain the native ensemble time-grid contract."""

from __future__ import annotations

import math

import pytest

import innovate


@pytest.mark.parametrize(
    "time",
    [
        [0.0, 0.0, 2.0],
        [-0.0, 0.0, 2.0],
        [2.0, 1.0, 0.0],
        [0.0, 2.0, 1.0],
    ],
    ids=["duplicate", "signed-zero", "descending", "local-reversal"],
)
def test_ensemble_rejects_unordered_time_through_native_boundary(time: list[float]) -> None:
    with pytest.raises(innovate.NativeError, match="time must be finite and strictly increasing") as error:
        innovate.execute(
            "compose_regime_ensemble",
            time=time,
            predictions={"low": [2.0, 4.0, 8.0], "high": [6.0, 8.0, 12.0]},
            weights={"low": 1.0, "high": 3.0},
        )
    assert error.value.code == "invalid_request"


def test_ensemble_preserves_irregular_time_and_independent_weighted_result() -> None:
    time = [-2.0, 0.5, 7.0]
    response = innovate.execute(
        "compose_regime_ensemble",
        time=time,
        predictions={"low": [2.0, 4.0, 8.0], "high": [6.0, 8.0, 12.0]},
        weights={"low": 1.0, "high": 3.0},
        observed=[4.0, 7.0, 12.0],
    )
    assert response["metadata"]["runtime"] == "rust_native"
    result = response["result"]
    assert result["time"] == time
    assert result["mean"] == [5.0, 7.0, 11.0]
    assert result["metadata"]["weights"] == {"low": 0.25, "high": 0.75}
    assert result["diagnostics"]["mae"] == pytest.approx(2.0 / 3.0, abs=1e-14)
    assert result["diagnostics"]["rmse"] == pytest.approx(math.sqrt(2.0 / 3.0), abs=1e-14)
