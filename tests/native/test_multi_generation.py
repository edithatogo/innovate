"""Installed bindings preserve every generation, fitted state and scientific semantics."""

import math

import polars as pl
import pytest

import innovate

PARAMETERS = {"p1": 0.03, "q1": 0.2, "m1": 1000.0, "p2": 0.02, "q2": 0.3, "m2": 1500.0}
OPTIONS = {"n_generations": 2, "variant": "successive_generations", "launch_times": [0.0, 4.0]}


def oracle(t):
    def fraction(t, p, q):
        return 0.0 if t <= 0 else (1 - math.exp(-(p + q) * t)) / (1 + q / p * math.exp(-(p + q) * t))

    f1, f2 = fraction(t, 0.03, 0.2), fraction(t - 4, 0.02, 0.3)
    return 1000 * f1 * (1 - f2), (1500 + 1000 * f1) * f2


def test_polars_multi_generation_prediction_preserves_all_columns_and_identity():
    frame = pl.DataFrame({"time": [0.0, 3.0, 5.0, 10.0], "sample": ["a", "b", "c", "d"]})
    result = innovate.predict("norton_bass", frame.lazy(), PARAMETERS, model_options=OPTIONS)
    assert result["sample"].to_list() == frame["sample"].to_list()
    for i, t in enumerate(frame["time"]):
        assert result["prediction_1"][i] == pytest.approx(oracle(t)[0], abs=1e-9)
        assert result["prediction_2"][i] == pytest.approx(oracle(t)[1], abs=1e-9)
    with pytest.raises(ValueError, match="already exists"):
        innovate.predict("norton_bass", result, PARAMETERS, model_options=OPTIONS)


def test_multi_output_fit_roundtrip_preserves_launch_times_and_reports_convergence():
    time = [i * 0.5 for i in range(41)]
    data = pl.DataFrame({"time": time, "first": [oracle(t)[0] for t in time], "second": [oracle(t)[1] for t in time]})
    fit = innovate.fit("norton_bass", data, model_options=OPTIONS, observed_columns=["first", "second"])
    assert fit.diagnostics["optimizer"]["converged"]
    assert fit.diagnostics["identifiability"]["rank"] == 6
    future = fit.predict(pl.DataFrame({"time": [21.0, 25.0, 30.0]}))
    for i, t in enumerate(future["time"]):
        assert future["prediction_1"][i] == pytest.approx(oracle(t)[0], abs=0.01)
        assert future["prediction_2"][i] == pytest.approx(oracle(t)[1], abs=0.01)


def test_signed_rates_are_derivatives_and_never_labelled_gross_incidence():
    times = [7.0, 15.0, 20.0]
    rates = innovate.predict(
        "norton_bass", pl.DataFrame({"time": times}), PARAMETERS, model_options=OPTIONS, quantity="rate"
    )
    for i, t in enumerate(times):
        for generation in (1, 2):
            expected = (oracle(t + 1e-4)[generation - 1] - oracle(t - 1e-4)[generation - 1]) / 2e-4
            assert rates[f"rate_{generation}"][i] == pytest.approx(expected, abs=1e-6)
    assert rates["rate_1"][-1] < 0


def test_invalid_options_and_observation_columns_fail_clearly():
    data = pl.DataFrame({"time": [0.0, 1.0, 2.0, 3.0], "observed": [0.0, 3.0, 5.0, 7.0]})
    with pytest.raises(ValueError):
        innovate.fit("norton_bass", data, observed_columns=["observed", "observed"])
    with pytest.raises(ValueError):
        innovate.predict("norton_bass", data, PARAMETERS, model_options=OPTIONS, quantity="incidence")
