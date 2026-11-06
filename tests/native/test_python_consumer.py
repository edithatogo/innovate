"""Installed consumers must exercise native code without legacy scientific packages."""

import importlib.util
import math

import polars as pl
import pytest

import innovate


def test_native_polars_prediction_and_row_identity():
    frame = pl.DataFrame({"time": [0.0, 1.0, 2.0], "id": ["a", "b", "c"]})
    result = innovate.predict("logistic", frame.lazy(), {"L": 100.0, "k": 1.0, "x0": 1.0})
    assert isinstance(result, pl.DataFrame)
    assert result["id"].to_list() == ["a", "b", "c"]
    assert result["prediction"][1] == 50.0


def test_fit_roundtrip_and_native_evidence():
    time = [float(x) for x in range(20)]
    observed = [100.0 / (1.0 + math.exp(-0.6 * (t - 8))) for t in time]
    fit = innovate.fit("logistic", pl.DataFrame({"time": time, "observed": observed}))
    assert fit.envelope["metadata"]["runtime"] == "rust_native"
    assert fit.predict(pl.DataFrame({"time": time}))["prediction"].to_list() == pytest.approx(observed, abs=0.01)


@pytest.mark.parametrize("time", [[0.0, float("nan")], [1.0, 0.0], [0.0, 0.0], [0.0, None]])
def test_invalid_time_contract(time):
    with pytest.raises(ValueError):
        innovate.predict("logistic", pl.DataFrame({"time": time}), {"L": 1.0, "k": 1.0, "x0": 0.0})


def test_capabilities_are_native_and_unavailable_features_fail_closed():
    assert set(innovate.capabilities()["key"]) == {"bass", "logistic", "gompertz", "fisher_pry", "norton_bass"}
    assert innovate.capabilities()["supports_simulation"].all()
    assert innovate.capabilities()["supports_summarize"].all()
    with pytest.raises(innovate.NativeError, match="unsupported_native_operation"):
        innovate.execute("predict_model", "latent_process", inputs={"time": [0.0]})


def test_clean_runtime_has_no_legacy_scientific_dependencies():
    for name in ["pandas", "xarray", "scipy", "numpy", "statsmodels"]:
        assert importlib.util.find_spec(name) is None, name


def test_empty_model_is_an_input_error():
    with pytest.raises(ValueError):
        innovate.predict("", pl.DataFrame({"time": [0.0, 1.0]}), {})


def test_forecast_batch_can_begin_after_zero():
    frame = innovate.predict("bass", pl.DataFrame({"time": [3.0, 4.0]}), {"p": 0.03, "q": 0.35, "m": 1000.0})
    decay = math.exp(-0.38 * 3.0)
    assert frame["prediction"][0] == pytest.approx(1000.0 * (1 - decay) / (1 + 0.35 / 0.03 * decay), abs=1e-10)
