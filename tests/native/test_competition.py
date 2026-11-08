"""Installed Polars consumers exercise competitive native models and saved state."""

import json
from pathlib import Path

import polars as pl
import pytest

import innovate


@pytest.mark.parametrize("model", ["lotka_volterra", "multi_product"])
def test_competitive_fit_and_forecast_keep_both_series_and_initial_state(model):
    fixture = json.loads(
        (Path(__file__).parents[2] / "rust/crates/innovate-core/tests/fixtures/competitive_scipy.json").read_text()
    )["cases"][model]
    frame = pl.DataFrame(
        {
            "time": [float(t) for t in fixture["time"]],
            "a": [row[0] for row in fixture["rows"]],
            "b": [row[1] for row in fixture["rows"]],
        }
    )
    fit = innovate.fit(
        model, frame, observed_columns=["a", "b"], fit_options={"initial_parameters": fixture["parameters"]}
    )
    assert fit.envelope["result"]["diagnostics"]["optimizer"]["converged"]
    future = fit.predict(pl.DataFrame({"id": ["held"], "time": [25.0]}))
    assert future["id"].to_list() == ["held"]
    assert future.select("prediction_1", "prediction_2").row(0) == pytest.approx(fixture["rows"][-1], abs=1e-4)
    rates = fit.predict(pl.DataFrame({"time": [25.0]}), quantity="rate")
    assert rates.select("rate_1", "rate_2").row(0) == pytest.approx(fixture["rates"][-1], abs=1e-5)
    discovery = innovate.capabilities().filter(pl.col("key") == model).row(0, named=True)
    assert discovery["supports_multivariate_output"] and discovery["supports_rates"]
