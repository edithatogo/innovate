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
    assert set(innovate.capabilities()["key"]) == {
        "bass",
        "logistic",
        "gompertz",
        "fisher_pry",
        "norton_bass",
        "lotka_volterra",
        "multi_product",
        "complementary_goods",
        "lock_in",
        "network_diffusion",
        "policy_hazard",
    }
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


def test_unrepresentable_fit_sensitivities_fail_in_the_installed_native_library():
    with pytest.raises(innovate.NativeError, match="finite numerical domain"):
        innovate.execute(
            "fit_model",
            "logistic",
            inputs={"time": [0.0, 1.0, 2.0, 3.0], "observed": [1e-12] * 4},
            fit_options={"initial_parameters": {"L": 1e140, "k": 1e15, "x0": 1.0}},
        )


def test_empty_non_mapping_options_are_not_silently_treated_as_defaults():
    data = pl.DataFrame({"time": [0.0, 1.0, 2.0, 3.0], "observed": [26.9, 50.0, 73.1, 88.1]})
    for invalid in [False, [], 0]:
        with pytest.raises(ValueError):
            innovate.predict("logistic", data, {"L": 100.0, "k": 1.0, "x0": 1.0}, model_options=invalid)
        with pytest.raises(ValueError):
            innovate.fit("logistic", data, model_options=invalid)
        with pytest.raises(ValueError):
            innovate.fit("logistic", data, fit_options=invalid)


def test_native_kairos_replay_and_engine_errors():
    payload = {
        "inputs": {"time": [0.0, 1.0, 5.0, 20.0]},
        "parameters": {"p": 0.2, "q": 0.3, "m": 100.0},
        "simulation_options": {"engine": "kairos_des", "seed": 42, "population": 100},
    }
    a = innovate.execute("simulate_model", "bass", **payload)
    assert a == innovate.execute("simulate_model", "bass", **payload)
    assert a["result"]["counts"][0] == 0
    assert a["metadata"]["engine"] == "kairos_des"
    assert len(set(a["result"]["counts"])) > 1
    payload["simulation_options"]["engine"] = "unknown"
    with pytest.raises(innovate.NativeError):
        innovate.execute("simulate_model", "bass", **payload)


def test_kairos_abm_forcing_retains_agents_and_restarts_adoption():
    r = innovate.execute(
        "simulate_model",
        "bass",
        inputs={"time": [0.0, 1.0, 3.0, 5.0]},
        parameters={"p": 0.2, "q": 0.3, "m": 100.0},
        simulation_options={
            "engine": "kairos_abm",
            "seed": 42,
            "population": 100,
            "covariate": {"time": [0.0, 3.0], "values": [-1.0, 0.0], "beta_p": 0.2, "beta_q": 0.3},
        },
    )["result"]
    assert r["counts"][:3] == [0, 0, 0]
    assert r["counts"][3] > 0
    assert len(r["agents"]) == 100
    assert sum(a["adopted"] for a in r["agents"]) == r["counts"][-1]


def test_historical_logistic_polars_covariates_and_event_fit():
    frame = pl.DataFrame({"time": [0.0, 1.0, 2.0], "price": [0.0, 1.0, 2.0]})
    p = {"L": 100.0, "k": 0.5, "x0": 2.0, "beta_L_price": 10.0, "beta_k_price": 0.1, "beta_x0_price": 0.2}
    a = innovate.predict("logistic", frame, p, model_options={"covariates": ["price"]})
    assert a["prediction"][1] == pytest.approx(110 / (1 + math.exp(0.72)))
    t = [float(i) for i in range(30)]
    y = [100 / (1 + math.exp(-0.5 * (x - 5))) if x < 12 else 200 / (1 + math.exp(-0.3 * (x - 18))) for x in t]
    fit = innovate.fit(
        "logistic",
        pl.DataFrame({"time": t, "observed": y}),
        model_options={"t_event": 12.0},
        fit_options={
            "initial_parameters": {"L": 110.0, "k": 0.4, "x0": 6.0, "L_post": 220.0, "k_post": 0.4, "x0_post": 17.0}
        },
    )
    assert fit.diagnostics["optimizer"]["converged"]
    assert fit.predict(pl.DataFrame({"time": [35.0]}))["prediction"][0] == pytest.approx(
        200 / (1 + math.exp(-5.1)), abs=0.001
    )


def test_network_resource_and_complementary_native_consumers():
    r = innovate.execute(
        "simulate_model",
        "bass",
        inputs={"time": [0.0, 5.0, 20.0]},
        parameters={"p": 1e12, "q": 0.0, "m": 3.0},
        simulation_options={
            "engine": "kairos_network",
            "population": 3,
            "seed": 42,
            "edges": [],
            "capacity": 1,
            "service_ticks": 5,
            "ticks_per_unit": 1,
        },
    )["result"]
    assert r["counts"] == [0, 0, 3]
    assert [e["tick"] for e in r["completions"]] == [6, 11, 16]
    a = innovate.predict(
        "complementary_goods",
        pl.DataFrame({"time": [0.0, 3.0]}),
        {"k1": 0.3, "k2": 0.4, "c1": 0.0, "c2": 0.0},
        model_options={"initial_values": [0.05, 0.02]},
    )
    assert a["prediction_1"][1] == pytest.approx(1 / (1 + 19 * math.exp(-0.9)), abs=1e-8)


def test_complementary_fit_matches_symmetric_analytic_limit():
    t = [float(i) for i in range(26)]
    capacity = 0.3 / 0.27
    y = [capacity / (1 + (capacity / 0.05 - 1) * math.exp(-0.3 * x)) for x in t]
    fit = innovate.fit(
        "complementary_goods",
        pl.DataFrame({"time": t, "a": y, "b": y}),
        observed_columns=["a", "b"],
        fit_options={"initial_parameters": {"k1": 0.28, "k2": 0.28, "c1": 0.025, "c2": 0.025}},
    )
    assert fit.diagnostics["optimizer"]["converged"]
    assert fit.parameters["c1"] == pytest.approx(0.03, abs=1e-4)


def test_lock_in_installed_polars_forecast_matches_analytic_limit():
    p = {"alpha1": 0.3, "alpha2": 0.2, "beta1": 0.0, "beta2": 0.0, "gamma1": 0.0, "gamma2": 0.0, "m": 1000.0}
    a = innovate.predict(
        "lock_in", pl.DataFrame({"time": [10.0, 20.0]}), p, model_options={"initial_values": [10.0, 0.0]}
    )
    assert a["prediction_1"][0] == pytest.approx(1000 / (1 + 99 * math.exp(-3)), abs=1e-5)
    assert a["prediction_2"].to_list() == [0.0, 0.0]


def test_historical_network_polars_fitting_and_forecast():
    t = [float(i) for i in range(26)]

    def bass(t, p, q, m):
        d = math.exp(-(p + q) * t)
        return m * (1 - d) / (1 + q / p * d)

    fit = innovate.fit(
        "network_diffusion",
        pl.DataFrame({"time": t, "a": [bass(x, 0.03, 0.3, 100) for x in t], "b": [bass(x, 0.02, 0.2, 200) for x in t]}),
        observed_columns=["a", "b"],
        model_options={"adjacency": [[0.0, 1.0], [1.0, 0.0]], "spillover_strength": 0.0},
    )
    a = fit.predict(pl.DataFrame({"time": [30.0]}))
    assert a["prediction_1"][0] == pytest.approx(bass(30, 0.03, 0.3, 100), abs=0.01)
    assert a["prediction_2"][0] == pytest.approx(bass(30, 0.02, 0.2, 200), abs=0.01)


def test_policy_event_boundary_and_native_baseline_fit():
    a = innovate.predict(
        "policy_hazard",
        pl.DataFrame({"time": [0.0, 2.0, 3.0]}),
        {"decay": 0.5, "base_L": 100.0, "base_k": 1.0, "base_x0": 2.0},
        model_options={"base_model": "logistic", "event_times": [2.0], "event_effects": [0.5]},
    )
    assert a["prediction"][1] == 75.0
    t = [float(i) for i in range(20)]
    y = [100 / (1 + math.exp(-0.5 * (x - 5))) for x in t]
    fit = innovate.fit(
        "policy_hazard",
        pl.DataFrame({"time": t, "observed": y}),
        model_options={"base_model": "logistic", "decay": 0.3},
    )
    assert fit.parameters["decay"] == 0.3
    assert fit.predict(pl.DataFrame({"time": [20.0]}))["prediction"][0] == pytest.approx(
        100 / (1 + math.exp(-7.5)), abs=0.001
    )
