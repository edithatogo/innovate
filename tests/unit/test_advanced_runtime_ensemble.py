"""Tests for advanced runtime ensemble compositions."""

from __future__ import annotations

import pytest

from innovate.advanced_runtime import compose_regime_ensemble


def test_compose_regime_ensemble_validation_and_combination() -> None:
    """Regime ensemble composition validates inputs and correctly weights predictions."""
    predictions = {"a": [1.0, 2.0], "b": [3.0, 4.0]}

    # Happy path: correct weighting
    result = compose_regime_ensemble(time=[1.0, 2.0], predictions=predictions, weights={"a": 0.25, "b": 0.75})
    assert result.mean == pytest.approx([2.5, 3.5])  # 0.25*[1,2] + 0.75*[3,4]

    # Error case: mismatched keys
    with pytest.raises(ValueError, match="weights must match prediction regime keys"):
        compose_regime_ensemble(time=[1.0, 2.0], predictions=predictions, weights={"a": 1.0, "c": 1.0})

    # Error case: invalid weights sum
    with pytest.raises(ValueError, match="weights must sum to a positive value"):
        compose_regime_ensemble(time=[1.0, 2.0], predictions=predictions, weights={"a": 0.0, "b": 0.0})
