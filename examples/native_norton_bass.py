"""Check two-generation fitting and forecasts against analytic diffusion curves."""

import json
from math import exp, isclose, isfinite

import polars as pl

from innovate import fit, predict


def fraction_and_rate(time: float, p: float, q: float) -> tuple[float, float]:
    """Return the Bass fraction and its right derivative after a fixed launch."""
    if time < 0:
        return 0.0, 0.0
    exponential = exp(-(p + q) * time)
    denominator = 1 + q / p * exponential
    return (1 - exponential) / denominator, (p + q) ** 2 / p * exponential / denominator**2


def analytic(time: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Evaluate the two-generation transfer equations independently in Python."""
    f1, d1 = fraction_and_rate(time, 0.03, 0.2)
    f2, d2 = fraction_and_rate(time - 4, 0.02, 0.3)
    levels = (1000 * f1 * (1 - f2), (1500 + 1000 * f1) * f2)
    rates = (1000 * (d1 * (1 - f2) - f1 * d2), 1000 * d1 * f2 + (1500 + 1000 * f1) * d2)
    return levels, rates


def main() -> None:
    """Validate synthetic recovery and held-out levels and signed rates."""
    parameters = {"p1": 0.03, "q1": 0.2, "m1": 1000.0, "p2": 0.02, "q2": 0.3, "m2": 1500.0}
    options = {"variant": "successive_generations", "launch_times": [0.0, 4.0]}
    table = pl.DataFrame({"time": [i * 0.5 for i in range(41)]})
    observed = predict("norton_bass", table, parameters, model_options=options)
    for time, first, second in observed.iter_rows():
        expected, _ = analytic(time)
        assert isclose(first, expected[0], abs_tol=1e-9, rel_tol=1e-12)
        assert isclose(second, expected[1], abs_tol=1e-9, rel_tol=1e-12)

    result = fit("norton_bass", observed, model_options=options, observed_columns=["prediction_1", "prediction_2"])
    assert result.diagnostics["optimizer"]["converged"] is True
    assert isinstance(result.envelope["result"]["state"], dict)
    for name, expected in parameters.items():
        assert isclose(result.parameters[name], expected, rel_tol=0.001)

    future = pl.DataFrame({"time": [21.0, 25.0, 30.0]})
    forecast = result.predict(future)
    rates = result.predict(future, quantity="rate")
    assert forecast.columns == ["time", "prediction_1", "prediction_2"]
    assert rates.columns == ["time", "rate_1", "rate_2"]
    assert forecast["time"].to_list() == future["time"].to_list()
    assert rates["time"].to_list() == future["time"].to_list()
    for level_row, rate_row in zip(forecast.iter_rows(), rates.iter_rows(), strict=True):
        expected_levels, expected_rates = analytic(level_row[0])
        for actual, expected in zip(level_row[1:] + rate_row[1:], expected_levels + expected_rates, strict=True):
            assert isfinite(actual) and isclose(actual, expected, abs_tol=0.01, rel_tol=0)
    # Signed generation rates may be negative during substitution.
    assert rates["rate_1"][0] < 0
    print(
        json.dumps(
            {"converged": True, "forecast_rows": forecast.height, "parameters_checked": len(parameters)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
