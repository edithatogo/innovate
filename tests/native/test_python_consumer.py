"""Installed consumers must exercise native code without legacy scientific packages."""

import importlib.util
import json
import math
from pathlib import Path

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
        "latent_process",
        "regime_switching",
        "hierarchical",
        "mixture",
        "composite",
    }
    assert innovate.capabilities()["supports_simulation"].all()
    assert innovate.capabilities()["supports_summarize"].all()
    with pytest.raises(innovate.NativeError, match="unsupported_native_operation"):
        innovate.execute("predict_model", "not_a_model", inputs={"time": [0.0]})
    with pytest.raises(innovate.NativeError, match="unsupported_native_operation"):
        innovate.execute(
            "predict_model",
            "bass",
            inputs={"time": [0.0], "covariates": {"price": [1.0]}},
            parameters={"p": 0.03, "q": 0.35, "m": 100.0},
        )


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


def test_heterogeneous_network_service_and_hazards():
    r = innovate.execute(
        "simulate_model",
        "bass",
        inputs={"time": [0.0, 100.0]},
        parameters={"p": 1e12, "q": 0.0, "m": 3.0},
        simulation_options={
            "engine": "kairos_network",
            "population": 3,
            "seed": 42,
            "edges": [],
            "capacity": 1,
            "service_ticks": 5,
            "ticks_per_unit": 1,
            "innovation_multipliers": [1.0, 0.0, 1.0],
            "agent_service_ticks": [4, 9, 7],
        },
    )["result"]
    assert r["counts"] == [0, 2]
    expected = 1
    for request, completion in zip(r["requests"], r["completions"], strict=True):
        assert request["agent"] == completion["agent"]
        expected += [4, 9, 7][request["agent"]]
        assert completion["tick"] == expected


def test_latent_polars_fit_preserves_residual_state():
    time = [float(i) for i in range(21)]
    observed = [100 / (1 + math.exp(-0.5 * (t - 8))) + 2 + 0.1 * math.sin(t) for t in time]
    fit = innovate.fit(
        "latent_process",
        pl.DataFrame({"time": time, "observed": observed}),
        model_options={"base_model": "logistic", "smoothing": 0.4},
    )
    state = fit.envelope["result"]["state"]
    assert len(state["constructor_kwargs"]["latent_values"]) == len(time)
    p = fit.parameters
    expected = p["base_L"] / (1 + math.exp(-p["base_k"] * (30 - p["base_x0"])))
    expected += state["constructor_kwargs"]["latent_values"][-1]
    assert fit.predict(pl.DataFrame({"time": [30.0]}))["prediction"][0] == pytest.approx(expected, abs=1e-10)


def test_fixed_time_regime_polars_fit_and_saved_forecast():
    time = [float(i) for i in range(30)]
    observed = [100 / (1 + math.exp(-0.4 * (t - 6))) if t < 15 else 200 / (1 + math.exp(-0.3 * (t - 18))) for t in time]
    fit = innovate.fit(
        "regime_switching",
        pl.DataFrame({"time": time, "observed": observed}),
        model_options={"base_model": "logistic", "prediction_mode": "fixed_time", "split_index": 15},
    )
    assert fit.envelope["result"]["state"]["constructor_kwargs"]["split_time"] == 15.0
    assert fit.parameters["regime_1_k"] == pytest.approx(0.3, abs=0.001)
    combined = fit.predict(pl.DataFrame({"time": [20.0, 30.0, 40.0]}))["prediction"].to_list()
    assert fit.predict(pl.DataFrame({"time": [40.0]}))["prediction"][0] == combined[-1]
    short = innovate.fit(
        "regime_switching",
        pl.DataFrame({"time": [0.0, 1.0], "observed": [0.11920292202211755, 0.2689414213699951]}),
        model_options={"base_model": "fisher_pry"},
    )
    assert short.envelope["result"]["details"]["regime_count"] == 1
    assert short.predict(pl.DataFrame({"time": [2.0]}))["prediction"][0] == pytest.approx(0.5, abs=1e-8)


def test_centered_hierarchical_polars_group_fit_and_forecast():
    time = [float(i) for i in range(25)]
    a = [100 / (1 + math.exp(-0.4 * (t - 7))) for t in time]
    b = [200 / (1 + math.exp(-0.3 * (t - 9))) for t in time]
    fit = innovate.fit(
        "hierarchical",
        pl.DataFrame({"time": time, "a": a, "b": b}),
        observed_columns=["a", "b"],
        model_options={
            "base_model": "logistic",
            "groups": ["a", "b"],
            "output": "groups",
        },
    )
    assert fit.parameters["global_L"] == pytest.approx(150.0, abs=0.01)
    forecast = fit.predict(pl.DataFrame({"time": [30.0]}))
    assert forecast["prediction_1"][0] == pytest.approx(100 / (1 + math.exp(-9.2)), abs=0.01)
    assert forecast["prediction_2"][0] == pytest.approx(200 / (1 + math.exp(-6.3)), abs=0.01)


def test_weighted_polars_fit_excludes_zero_weight_outlier():
    time = [float(i) for i in range(25)]
    y = [100 / (1 + math.exp(-0.4 * (t - 7))) for t in time]
    y[20] = 300
    weights = [1.0] * 25
    weights[20] = 0.0
    fit = innovate.fit(
        "logistic",
        pl.DataFrame({"time": time, "observed": y}),
        fit_options={"weights": weights, "initial_parameters": {"L": 110.0, "k": 0.35, "x0": 8.0}},
    )
    assert fit.parameters["L"] == pytest.approx(100.0, abs=1e-5)
    assert fit.diagnostics["weighting"]["positive_observations"] == 24


def test_gaussian_mixture_polars_fit_and_forecast_state():
    time = [i * 0.5 for i in range(50)]
    y = [
        100 / (1 + math.exp(-0.4 * (t - 6))) if i % 2 == 0 else 200 / (1 + math.exp(-0.3 * (t - 10)))
        for i, t in enumerate(time)
    ]
    parameters = {
        "model_0_L": 100.0,
        "model_0_k": 0.4,
        "model_0_x0": 6.0,
        "model_1_L": 200.0,
        "model_1_k": 0.3,
        "model_1_x0": 10.0,
        "weight_0": 0.4,
        "weight_1": 0.6,
        "noise_scale": 2.0,
    }
    fit = innovate.fit(
        "mixture",
        pl.DataFrame({"time": time, "observed": y}),
        model_options={"components": ["logistic", "logistic"], "max_iter": 30, "noise_floor": 0.001},
        fit_options={"initial_parameters": parameters},
    )
    assert fit.envelope["result"]["optimizer"]["converged"]
    assert fit.parameters["model_1_L"] == pytest.approx(200.0, abs=0.05)
    assert fit.predict(pl.DataFrame({"time": [40.0]}))["prediction"][0] > 140.0


def test_composite_polars_levels_rates_and_fitted_origin():
    frame = pl.DataFrame({"time": [0.0, 3.0, 10.0, 20.0]})
    parameters = {
        "p_1": 0.03,
        "q_1": 0.35,
        "m_1": 100.0,
        "L_2": 200.0,
        "k_2": 0.4,
        "x0_2": 99.0,
        "alpha_1_2": 0.0,
        "alpha_2_1": 0.0,
    }
    options = {"components": ["bass", "logistic"], "initial_values": [0.0, 10.0]}
    levels = innovate.predict("composite", frame, parameters, model_options=options)
    logistic = 200 / (1 + 19 * math.exp(-4))
    assert levels["prediction_2"][2] == pytest.approx(logistic, abs=1e-6)
    rates = innovate.predict("composite", frame, parameters, model_options=options, quantity="rate")
    assert rates["rate_2"][2] == pytest.approx(0.4 * logistic * (1 - logistic / 200), abs=1e-6)
    time = [float(i) for i in range(25)]
    y = [100 * (1 - math.exp(-0.38 * t)) / (1 + 0.35 / 0.03 * math.exp(-0.38 * t)) for t in time]
    fit = innovate.fit(
        "composite",
        pl.DataFrame({"time": time, "observed": y}),
        model_options={"components": ["bass"]},
        fit_options={"initial_parameters": {"p_1": 0.025, "q_1": 0.32, "m_1": 110.0}},
    )
    assert fit.diagnostics["optimizer"]["converged"]
    assert fit.predict(pl.DataFrame({"time": [30.0]}))["prediction"][0] > 99.5


def test_bass_mechanistic_covariate_event_fit_and_history_forecast():
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2] / "rust/crates/innovate-core/tests/fixtures/mechanistic_scipy.json"
        ).read_text()
    )
    n = fixture["training_rows"]
    parameters = fixture["parameters"].copy()
    parameters["beta_p_dose"] = 0.004
    fit = innovate.fit(
        "bass",
        pl.DataFrame({"time": fixture["time"][:n], "observed": fixture["observed"][:n], "dose": fixture["dose"][:n]}),
        model_options={"covariates": ["dose"], "t_event": 10.0},
        fit_options={"initial_parameters": parameters},
    )
    assert fit.diagnostics["optimizer"]["converged"]
    replay = fit.predict(pl.DataFrame({"time": fixture["time"][:n], "dose": fixture["dose"][:n]}))
    assert replay["prediction"].to_list() == pytest.approx(fit.envelope["result"]["predictions"]["values"], abs=1e-7)
    future = pl.DataFrame({"time": fixture["time"][n:], "dose": fixture["dose"][n:]})
    assert fit.predict(future)["prediction"][-1] == pytest.approx(fixture["observed"][-1], abs=0.001)
    assert fit.predict(future, quantity="rate")["rate"][-1] == pytest.approx(fixture["rates"][-1], abs=0.001)


def test_directed_weighted_kairos_resource():
    options = {
        "engine": "kairos_network",
        "population": 3,
        "seed": 42,
        "directed": True,
        "edges": [[0, 1]],
        "edge_weights": [1e308],
        "initial_agents": [0],
        "agent_priorities": [-2, 0, 3],
        "capacity": 1,
        "service_ticks": 5,
        "ticks_per_unit": 1,
        "schedule": [{"tick": 6, "edges": [[1, 2]], "service_ticks": 2, "capacity": 2, "p": 1e12, "q": 0.0}],
    }
    result = innovate.execute(
        "simulate_model",
        "bass",
        inputs={"time": [0.0, 5.0, 20.0]},
        parameters={"p": 0.0, "q": 1e12, "m": 3.0},
        simulation_options=options,
    )["result"]
    assert result["counts"] == [1, 1, 3]
    assert [entry["tick"] for entry in result["completions"]] == [6, 9]
    assert result["metadata"]["agent_priorities"] == [-2, 0, 3]
    assert result["metadata"]["current_capacity"] == 2
    assert result["metadata"]["current_p"] == 1e12
    assert result["metadata"]["current_q"] == 0.0


def test_kairos_wait_expiry():
    result = innovate.execute(
        "simulate_model",
        "bass",
        inputs={"time": [0.0, 7.0, 11.0]},
        parameters={"p": 1e12, "q": 0.0, "m": 3.0},
        simulation_options={
            "engine": "kairos_network",
            "population": 3,
            "seed": 42,
            "edges": [],
            "capacity": 1,
            "service_ticks": 5,
            "ticks_per_unit": 1,
            "max_wait_ticks": 4,
        },
    )["result"]
    assert result["counts"] == [0, 1, 2]
    assert [entry["tick"] for entry in result["abandonments"]] == [7]


def test_kairos_queue_retry():
    result = innovate.execute(
        "simulate_model",
        "bass",
        inputs={"time": [0.0, 6.0, 12.0]},
        parameters={"p": 1e12, "q": 0.0, "m": 2.0},
        simulation_options={
            "engine": "kairos_network",
            "population": 2,
            "seed": 42,
            "edges": [],
            "capacity": 1,
            "service_ticks": 5,
            "ticks_per_unit": 1,
            "max_queue_length": 0,
            "retry": {"delay_ticks": 4, "max_retries": 1},
        },
    )["result"]
    assert result["counts"] == [0, 1, 2]
    assert [entry["tick"] for entry in result["balks"]] == [2]
    assert [entry["tick"] for entry in result["retry_eligibility"]] == [6]


def test_norton_bass_covariate_history():
    parameters = {
        "p1": 0.1,
        "q1": 0.0,
        "m1": 100.0,
        "p2": 0.1,
        "q2": 0.0,
        "m2": 100.0,
        "beta_p1_price": 0.05,
        "beta_q1_price": 0.0,
        "beta_m1_price": 0.0,
        "beta_p2_price": 0.05,
        "beta_q2_price": 0.0,
        "beta_m2_price": 0.0,
    }
    result = innovate.execute(
        "predict_model",
        "norton_bass",
        inputs={"time": [0.0, 1.0, 2.0], "covariates": {"price": [0.0, 1.0, 2.0]}},
        parameters=parameters,
        constructor_kwargs={"n_generations": 2, "covariates": ["price"]},
    )["result"]
    forecast = innovate.execute(
        "predict_model",
        "norton_bass",
        inputs={"time": [3.0]},
        state=result["state"],
    )["result"]
    assert result["rows"][2] == pytest.approx([100 * 0.3 * math.exp(-0.3), 100 * (1 - math.exp(-0.3))], abs=1e-5)
    assert forecast["rows"][0][0] == pytest.approx(50 * math.exp(-0.5), abs=1e-5)


@pytest.mark.parametrize(
    "model,fixture_name,start_override",
    [
        ("lotka_volterra", "lotka_forcing_scipy.json", {"alpha1": 0.29, "beta_beta1_price": -0.025}),
        ("complementary_goods", "complementary_forcing_scipy.json", {"k1": 0.29, "beta_c2_price": -0.007}),
        ("lock_in", "lock_forcing_scipy.json", {"alpha1": 0.29}),
    ],
)
def test_ecosystem_forcing_polars_fit_and_forecast(model, fixture_name, start_override):
    fixture = json.loads(
        (Path(__file__).parents[2] / "rust/crates/innovate-core/tests/fixtures" / fixture_name).read_text()
    )
    data = pl.DataFrame(
        {
            "time": fixture["time"],
            "price": fixture["covariates"]["price"],
            "first": [row[0] for row in fixture["rows"]],
            "second": [row[1] for row in fixture["rows"]],
        }
    )
    start = dict(fixture["parameters"], **start_override)
    fitted = innovate.fit(
        model,
        data,
        observed_columns=["first", "second"],
        model_options={"initial_values": fixture["initial"], "covariates": ["price"]},
        fit_options={"initial_parameters": start, "max_iterations": 150},
    )
    assert fitted.diagnostics["optimizer"]["converged"]
    forecast = innovate.execute(
        "predict_model",
        model,
        inputs={"time": fixture["forecast_time"]},
        state=fitted.envelope["result"]["state"],
    )["result"]
    assert forecast["rows"][0] == pytest.approx(fixture["forecast_rows"][0], abs=1e-5)


def test_multi_product_forcing_layouts_and_polars_fit():
    fixture = json.loads(
        (Path(__file__).parents[2] / "rust/crates/innovate-core/tests/fixtures/multi_forcing_scipy.json").read_text()
    )
    for layout in ["corrected", "historical"]:
        result = innovate.execute(
            "predict_model",
            "multi_product",
            inputs={"time": fixture["time"], "covariates": fixture["covariates"]},
            parameters=fixture["parameters"],
            constructor_kwargs={
                "n_products": 2,
                "covariates": ["price"],
                "coefficient_layout": layout,
            },
        )["result"]
        assert result["rows"][-1] == pytest.approx(fixture["cases"][layout]["rows"][-1], abs=1e-5)
        forecast = innovate.execute(
            "predict_model",
            "multi_product",
            inputs={"time": fixture["forecast_time"]},
            state=result["state"],
        )["result"]
        assert forecast["rows"][0] == pytest.approx(fixture["cases"][layout]["forecast_rows"][0], abs=1e-5)
    rows = fixture["cases"]["corrected"]["rows"]
    data = pl.DataFrame(
        {
            "time": fixture["time"],
            "price": fixture["covariates"]["price"],
            "first": [row[0] for row in rows],
            "second": [row[1] for row in rows],
        }
    )
    start = dict(fixture["parameters"], p1=0.0305)
    fit = innovate.fit(
        "multi_product",
        data,
        observed_columns=["first", "second"],
        model_options={"n_products": 2, "covariates": ["price"]},
        fit_options={"initial_parameters": start, "max_iterations": 200},
    )
    assert fit.diagnostics["optimizer"]["converged"]
    future = fit.predict(
        pl.DataFrame(
            {
                "time": fixture["forecast_time"],
                "price": [fixture["covariates"]["price"][-1]],
            }
        )
    )
    assert [future["prediction_1"][0], future["prediction_2"][0]] == pytest.approx(
        fixture["cases"]["corrected"]["forecast_rows"][0], abs=1e-3
    )


def test_lock_in_capacity_projection_recovers_retained_state():
    parameters = dict(alpha1=0.0, alpha2=0.0, beta1=0.0, beta2=0.0, gamma1=0.0, gamma2=0.0, m=100.0)
    parameters.update({f"beta_{name}_price": 0.0 for name in parameters})
    parameters["beta_m_price"] = -10.0
    result = innovate.execute(
        "predict_model",
        "lock_in",
        inputs={"time": [0.0, 1.0, 2.0, 3.0], "covariates": {"price": [0.0, 1.0, 2.0, 0.0]}},
        parameters=parameters,
        constructor_kwargs={"initial_values": [95.0, 0.0], "covariates": ["price"]},
    )["result"]
    assert [row[0] for row in result["rows"]] == [95.0, 90.0, 80.0, 95.0]
    assert result["rates"] == [[0.0, 0.0]] * 4
    assert result["metadata"]["projection"] == "historical_clip_to_current_capacity"


@pytest.mark.parametrize("model", ["lotka_volterra", "complementary_goods", "lock_in", "multi_product"])
def test_ecosystem_fixed_events_and_saved_forecast(model):
    fixture = json.loads(
        (Path(__file__).parents[2] / "rust/crates/innovate-core/tests/fixtures/ecosystem_events_scipy.json").read_text()
    )
    case = fixture["cases"][model]
    result = innovate.execute(
        "predict_model",
        model,
        inputs={"time": fixture["time"], "covariates": fixture["covariates"]},
        parameters=case["parameters"],
        constructor_kwargs=case["constructor_kwargs"],
    )["result"]
    assert result["rows"][-1] == pytest.approx(case["rows"][-1], abs=1e-5)
    event_index = fixture["time"].index(2.125)
    assert result["rates"][event_index] == pytest.approx(case["rates"][event_index], abs=1e-5)
    future = innovate.execute(
        "predict_model",
        model,
        inputs={"time": fixture["forecast_time"]},
        state=result["state"],
    )["result"]
    assert future["rows"][0] == pytest.approx(case["forecast_rows"][0], abs=1e-5)


def test_ecosystem_fixed_event_polars_fit():
    fixture = json.loads(
        (Path(__file__).parents[2] / "rust/crates/innovate-core/tests/fixtures/ecosystem_events_scipy.json").read_text()
    )
    case = fixture["cases"]["lotka_volterra"]
    data = pl.DataFrame(
        {
            "time": fixture["time"],
            "price": fixture["covariates"]["price"],
            "first": [row[0] for row in case["rows"]],
            "second": [row[1] for row in case["rows"]],
        }
    )
    fit = innovate.fit(
        "lotka_volterra",
        data,
        observed_columns=["first", "second"],
        model_options=case["constructor_kwargs"],
        fit_options={"initial_parameters": dict(case["parameters"], alpha1=0.29), "max_iterations": 200},
    )
    assert fit.diagnostics["optimizer"]["converged"]
    assert not fit.diagnostics["parameter_events"]["parameters_estimated"]
    future = fit.predict(
        pl.DataFrame(
            {
                "time": fixture["forecast_time"],
                "price": [fixture["covariates"]["price"][-1]],
            }
        )
    )
    assert [future["prediction_1"][0], future["prediction_2"][0]] == pytest.approx(case["forecast_rows"][0], abs=1e-4)


@pytest.mark.parametrize("fixture_name", ["norton_events_scipy.json", "successive_forcing_scipy.json"])
@pytest.mark.parametrize("explicit_start", [True, False])
def test_norton_event_variants_polars_fit_and_forecast(fixture_name, explicit_start):
    fixture = json.loads(
        (Path(__file__).parents[2] / "rust/crates/innovate-core/tests/fixtures" / fixture_name).read_text()
    )
    data = pl.DataFrame(
        {
            "time": fixture["time"],
            "price": fixture["covariates"]["price"],
            "first": [row[0] for row in fixture["rows"]],
            "second": [row[1] for row in fixture["rows"]],
        }
    )
    fit_options: dict[str, object] = {"max_iterations": 200}
    if explicit_start:
        fit_options["initial_parameters"] = dict(fixture["parameters"], p1=0.0305)
    fit = innovate.fit(
        "norton_bass",
        data,
        observed_columns=["first", "second"],
        model_options=fixture["constructor_kwargs"],
        fit_options=fit_options,
    )
    assert fit.diagnostics["optimizer"]["converged"]
    assert not fit.diagnostics["parameter_events"]["parameters_estimated"]
    forecast = fit.predict(
        pl.DataFrame(
            {
                "time": fixture["forecast_time"],
                "price": [fixture["covariates"]["price"][-1]],
            }
        )
    )
    assert [forecast["prediction_1"][0], forecast["prediction_2"][0]] == pytest.approx(
        fixture["forecast_rows"][0], abs=1e-3
    )


def test_norton_baseline_threshold_summary():
    result = innovate.execute(
        "summarize_model",
        "norton_bass",
        inputs={"time": [4.0]},
        parameters={"p1": 0.1, "q1": 0.9, "m1": 100.0},
        constructor_kwargs={"variant": "successive_generations", "launch_times": [4.0]},
    )["result"]
    row = result["threshold_diagnostics"]["generations"][0]
    assert row["reference_time"] == 4.0
    assert row["half_life"] == pytest.approx(math.log(9))
    assert row["saturation"] == pytest.approx(2 * math.log(9))


def test_norton_forced_rate_peak():
    result = innovate.execute(
        "summarize_model",
        "norton_bass",
        inputs={"time": [0.0, 1.0, 2.0, 3.0]},
        parameters={"p1": 0.1, "q1": 0.0, "m1": 100.0},
        constructor_kwargs={
            "variant": "successive_generations",
            "parameter_events": [{"time": 2.0, "parameters": {"p1": 0.5}}],
        },
    )["result"]
    trajectory = result["threshold_diagnostics"]["trajectory"]
    assert trajectory["window"] == [0.0, 3.0]
    assert trajectory["generations"][0]["peak_time"] == 2.0
    assert trajectory["generations"][0]["peak_rate"] == pytest.approx(50 * math.exp(-0.2))
    milestones = result["threshold_diagnostics"]["intrinsic_adoption_milestones"]
    row = milestones["generations"][0]
    assert row["median"] == pytest.approx(2 + (math.log(2) - 0.2) / 0.5)
    assert row["saturation"] is None


def test_norton_historical_level_milestones():
    result = innovate.execute(
        "summarize_model",
        "norton_bass",
        inputs={"time": [1.0, 2.0, 3.0]},
        parameters={"p1": 0.1, "q1": 0.0, "m1": 100.0},
        constructor_kwargs={
            "initial_time": 1.0,
            "initial_values": [12.0],
            "parameter_events": [{"time": 2.0, "parameters": {"m1": 1000.0}}],
        },
    )["result"]
    row = result["threshold_diagnostics"]["net_level_milestones"]["generations"][0]
    assert row == {
        "generation": 1,
        "reference_market": 100.0,
        "takeoff": 1.0,
        "median": 3.0,
        "saturation": 3.0,
    }


def test_kairos_scheduled_agent_hazards():
    result = innovate.execute(
        "simulate_model",
        "bass",
        inputs={"time": [0, 2, 5, 8]},
        parameters={"p": 1e12, "q": 0, "m": 2},
        simulation_options={
            "engine": "kairos_network",
            "population": 2,
            "seed": 42,
            "edges": [],
            "capacity": 1,
            "service_ticks": 4,
            "ticks_per_unit": 1,
            "innovation_multipliers": [1, 0],
            "schedule": [{"tick": 2, "innovation_multipliers": [0, 1]}],
        },
    )["result"]
    assert result["requests"] == [{"tick": 1, "agent": 0}, {"tick": 3, "agent": 1}]
    assert result["completions"][0] == {"tick": 5, "agent": 0}


def test_kairos_continuous_hazard_wakeup():
    result = innovate.execute(
        "simulate_model",
        "bass",
        inputs={"time": [0, 2, 3, 4]},
        parameters={"p": 1e12, "q": 0, "m": 1},
        simulation_options={
            "engine": "kairos_network",
            "population": 1,
            "seed": 42,
            "edges": [],
            "capacity": 1,
            "service_ticks": 1,
            "ticks_per_unit": 1,
            "hazard_forcing": {
                "time": [0, 2, 3],
                "innovation": [0, 0, 1],
                "imitation": [1, 1, 1],
            },
        },
    )["result"]
    assert result["requests"] == [{"tick": 3, "agent": 0}]
    assert result["counts"] == [0, 0, 0, 1]
