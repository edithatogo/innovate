"""Installed consumers must exercise native code without legacy scientific packages."""

import hashlib
import importlib.util
import json
import math
import os
import random
import struct
from fractions import Fraction
from itertools import product
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
    rate = innovate.predict("logistic", frame, p, model_options={"covariates": ["price"]}, quantity="rate")
    f = 1 / (1 + math.exp(0.72))
    assert rate["rate"][1] == pytest.approx(110 * 0.6 * f * (1 - f))
    event_parameters = {"L": 100, "k": 0.5, "x0": 2, "L_post": 200, "k_post": 1, "x0_post": 2}
    boundary = innovate.predict(
        "logistic",
        pl.DataFrame({"time": [2.0]}),
        event_parameters,
        model_options={"t_event": 2.0},
        quantity="rate",
    )
    assert boundary["rate"][0] == 50
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
    assert r["resource_metrics"]["peak_queue_length"] == 2
    assert r["resource_metrics"]["peak_in_service"] == 1
    assert r["resource_metrics"]["idle_slot_ticks"] == 5
    assert r["resource_metrics"]["over_capacity_slot_ticks"] == 0
    closed = innovate.execute(
        "simulate_model",
        "bass",
        inputs={"time": [0, 5, 20]},
        parameters={"p": 1e12, "q": 0, "m": 3},
        simulation_options={
            "engine": "kairos_network",
            "population": 3,
            "seed": 42,
            "edges": [],
            "capacity": 1,
            "service_ticks": 5,
            "ticks_per_unit": 1,
            "schedule": [{"tick": 3, "capacity": 0}],
        },
    )["result"]
    assert closed["resource_metrics"]["idle_slot_ticks"] == 1
    assert closed["resource_metrics"]["over_capacity_slot_ticks"] == 3
    assert [event["tick"] for event in closed["completions"]] == [6]
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


@pytest.mark.parametrize("explicit_start, profile_event", [(True, False), (False, False), (False, True)])
def test_bass_mechanistic_covariate_event_fit_and_history_forecast(explicit_start, profile_event):
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2] / "rust/crates/innovate-core/tests/fixtures/mechanistic_scipy.json"
        ).read_text()
    )
    n = fixture["training_rows"]
    parameters = fixture["parameters"].copy()
    parameters["beta_p_dose"] = 0.004
    event_options = {"event_candidates": [9.0, 10.0, 11.0]} if profile_event else {"t_event": 10.0}
    fit = innovate.fit(
        "bass",
        pl.DataFrame({"time": fixture["time"][:n], "observed": fixture["observed"][:n], "dose": fixture["dose"][:n]}),
        model_options={"covariates": ["dose"], **event_options},
        fit_options={"initial_parameters": parameters} if explicit_start else None,
    )
    assert (fit.event_profile is not None) == profile_event
    assert fit.envelope["result"]["state"]["constructor_kwargs"]["t_event"] == 10.0
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
        ("lotka_volterra", "lotka_forcing_scipy.json", None),
        ("complementary_goods", "complementary_forcing_scipy.json", None),
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
    fit_options: dict[str, object] = {"max_iterations": 150}
    if start_override is not None:
        fit_options["initial_parameters"] = dict(fixture["parameters"], **start_override)
    fitted = innovate.fit(
        model,
        data,
        observed_columns=["first", "second"],
        model_options={"initial_values": fixture["initial"], "covariates": ["price"]},
        fit_options=fit_options,
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
    at_event = 100 - 88 * math.exp(-0.1)
    assert row["refined_crossings"]["takeoff"] == 1
    for name, target in [("median", 50), ("saturation", 90)]:
        expected = 2 - math.log((1000 - target) / (1000 - at_event)) / 0.1
        assert row["refined_crossings"][name] == pytest.approx(expected, abs=1e-7, rel=0)
    assert {k: v for k, v in row.items() if k != "refined_crossings"} == {
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


def test_logistic_polars_total_rate():
    frame = pl.DataFrame({"time": [2.0], "price": [1.0], "price_slope": [2.0]})
    parameters = {
        "L": 200,
        "k": 1,
        "x0": 1.5,
        "beta_L_price": 10,
        "beta_k_price": 0.2,
        "beta_x0_price": 0.5,
    }
    options = {"covariates": ["price"]}
    result = innovate.predict(
        "logistic",
        frame,
        parameters,
        model_options=options,
        quantity="rate",
        covariate_rate_columns={"price": "price_slope"},
    )
    assert result["rate"].to_list() == [10.0]
    with pytest.raises(ValueError, match="must match declared controls"):
        innovate.predict(
            "logistic",
            frame,
            parameters,
            model_options=options,
            quantity="rate",
            covariate_rate_columns={"unexpected": "price_slope"},
        )


def test_polars_batch_keeps_errors_state_and_ragged_frames():
    times = list(range(20))
    frames = [
        pl.DataFrame({"time": times, "observed": [capacity / (1 + math.exp(-(t - 5))) for t in times]})
        for capacity in (100, 200)
    ]
    fits = innovate.fit_frames("logistic", [frames[0].lazy(), pl.DataFrame({"time": [1]}), frames[1]])
    assert isinstance(fits[1], ValueError)
    grids = [
        pl.DataFrame({"time": [5], "id": ["first"]}),
        pl.DataFrame({"time": [5]}),
        pl.DataFrame({"time": [5, 6], "id": ["second", "third"]}),
    ]
    first, failed, last = innovate.predict_frames(zip(fits, grids, strict=True))
    assert failed is fits[1]
    assert isinstance(first, pl.DataFrame)
    assert isinstance(last, pl.DataFrame)
    assert first["prediction"][0] == pytest.approx(50)
    assert last["prediction"][0] == pytest.approx(100)
    assert last["id"].to_list() == ["second", "third"]

    invalid_query = grids[0].lazy().select("absent")
    failed, recovered = innovate.predict_frames([(fits[0], invalid_query), (fits[2], grids[2].lazy())])
    assert isinstance(failed, pl.exceptions.ColumnNotFoundError)
    assert isinstance(recovered, pl.DataFrame)
    assert recovered.equals(last)


def test_fitted_logistic_adopter_categories_use_native_rates():
    time = list(range(21))
    observed = [100 / (1 + math.exp(-(t - 10))) for t in time]
    fitted = innovate.fit("logistic", pl.DataFrame({"time": time, "observed": observed}))
    result = fitted.categorize_adopters(pl.DataFrame({"time": time, "id": time}))
    assert result["rate"][10] == pytest.approx(25)
    assert result["category"][0] == "Innovators"
    assert result["category"][-1] == "Laggards"
    assert result["id"].to_list() == time


@pytest.mark.parametrize(
    ("model", "parameters"),
    [
        ("bass", {"p": 0.03, "q": 0.3, "m": 100.0}),
        ("gompertz", {"a": 100.0, "b": 4.0, "c": 0.3}),
        ("fisher_pry", {"alpha": 0.5, "t0": 10.0}),
    ],
)
def test_fitted_plain_curves_classify_from_native_rates(model, parameters):
    time = list(range(31))
    frame = pl.DataFrame({"time": time})
    observed = innovate.predict(model, frame, parameters).rename({"prediction": "observed"})
    fitted = innovate.fit(model, observed, fit_options={"initial_parameters": parameters})
    result = fitted.categorize_adopters(frame)
    assert result.height == len(time)
    assert result["rate"].min() >= 0
    assert result["rate"].max() > 0
    assert result["category"].n_unique() >= 3


def test_polars_preprocessing_preserves_order_and_full_window():
    series = pl.Series("adoption", [1, 3, 5, 7])
    assert innovate.cumulative_sum(series).to_list() == [1, 4, 9, 16]
    assert innovate.rolling_average(series, 3).to_list() == [None, None, 3.0, 5.0]
    for value in (float.fromhex("0x1.fffffffffffffp+1023"), math.ulp(0.0)):
        extreme = pl.Series("rate", [value, value, 0.0])
        result = innovate.rolling_average(extreme, 2)
        assert result.name == extreme.name
        assert result.dtype == extreme.dtype
        assert result.to_list() == [None, value, value / 2]
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    with_gap = pl.Series("rate", [largest, largest, None, largest, largest])
    assert innovate.rolling_average(with_gap, 2).to_list() == [None, largest, None, None, largest]
    with_nan = pl.Series("rate", [largest, largest, math.nan, largest, largest])
    result = innovate.rolling_average(with_nan, 2).to_list()
    assert result[0] is None and result[1] == result[4] == largest
    assert math.isnan(result[2]) and math.isnan(result[3])
    with pytest.raises(ValueError, match="window must be positive"):
        innovate.rolling_average(series, 0)


def test_weekly_polars_aggregation_uses_first_timestamp_and_omits_empty_weeks():
    from datetime import date

    days = [date(2026, 1, d) for d in (13, 6, 12, 27)]
    result = innovate.aggregate_weekly(pl.DataFrame({"time": days, "y": [3, 1, 2, 4]}).lazy(), ["y"])
    assert result["time"].to_list() == [date(2026, 1, d) for d in (6, 13, 27)]
    assert result["y"].to_list() == [3, 3, 4]


def test_calendar_polars_aggregation_labels_and_fills_empty_periods():
    from datetime import date

    daily = pl.DataFrame({"time": [date(2026, 1, 3), date(2026, 1, 1)], "y": [3, 1]})
    assert innovate.aggregate_time_series(daily, "D", ["y"])["y"].to_list() == [1, 0, 3]
    monthly = pl.DataFrame({"time": [date(2024, 1, 2), date(2024, 3, 31)], "y": [2, 3]})
    result = innovate.aggregate_time_series(monthly, "M", ["y"])
    assert result["time"].to_list() == [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31)]
    assert result["y"].to_list() == [2, 0, 3]


def test_native_seasonal_decomposition_recovers_analytic_components():
    trend = [10 + i * 0.5 for i in range(32)]
    seasonal = [-3, 2, 1, 0] * 8
    values = pl.Series([t + s for t, s in zip(trend, seasonal, strict=True)])
    result = innovate.seasonal_decomposition(values, period=4, trend_span=9)
    assert result["trend"].to_list() == pytest.approx(trend, abs=1e-10)
    assert result["seasonal"].to_list() == pytest.approx(seasonal, abs=1e-10)
    assert result["residual"].abs().max() == pytest.approx(0, abs=1e-10)


def test_robust_native_seasonal_decomposition_isolates_outlier():
    values = [10 + i * 0.5 + (-3, 2, 1, 0)[i % 4] for i in range(80)]
    values[40] += 100
    result = innovate.seasonal_decomposition(pl.Series(values), period=4, trend_span=13, robust=True)
    assert result["residual"][40] == pytest.approx(100, abs=0.1)


@pytest.mark.parametrize("suffix", [".arrow", ".ipc_stream"])
def test_ingest_local_ipc_stream_preserves_dataset_and_provenance(tmp_path, suffix):
    path = tmp_path / f"adoption{suffix}"
    frame = pl.DataFrame({"time": [0.0, 1.0], "adoption": [1.0, 2.0]})
    provenance = {
        "source": "synthetic",
        "license": "CC-BY-4.0",
        "extraction_time": "2026-11-07T00:00:00Z",
    }
    frame.write_ipc_stream(path)
    artifact = innovate.ingest_local(path, "adoption", provenance=provenance)
    dataset = artifact["dataset"]
    assert artifact["validation"]["ok"]
    assert dataset["adoption"] == [1.0, 2.0]
    assert dataset["provenance"]["extra"]["format"] == suffix[1:]
    assert innovate.execute("ingest_dataset", dataset=dataset)["result"] == artifact
    if suffix == ".arrow":
        frame.write_ipc(path)
        assert innovate.ingest_local(path, "adoption", provenance=provenance) == artifact


def test_logistic_event_profile_polars_fit_and_saved_split_prediction():
    time = list(range(24))
    observed = [100 / (1 + math.exp(-0.5 * (t - 5))) if t < 12 else 200 / (1 + math.exp(-0.4 * (t - 17))) for t in time]
    frame = pl.DataFrame({"time": time, "observed": observed})
    fit = innovate.fit("logistic", frame.lazy(), model_options={"event_candidates": [10, 12, 14]})
    profile = fit.event_profile
    assert profile is not None
    assert min(profile["candidates"], key=lambda row: row["objective"])["t_event"] == 12
    query = pl.DataFrame({"time": [11, 12, 13], "id": ["pre", "split", "post"]})
    forecast = fit.predict(query)
    assert forecast["id"].to_list() == ["pre", "split", "post"]
    assert forecast["prediction"].to_list() == pytest.approx(observed[11:14], abs=1e-7)


def test_norton_joint_time_profile_polars_fit_and_forecast():
    path = Path(__file__).parents[2] / "rust/crates/innovate-core/tests/fixtures/successive_forcing_scipy.json"
    fixture = json.loads(path.read_text())
    first, second = zip(*fixture["rows"], strict=True)
    price = fixture["covariates"]["price"]
    frame = pl.DataFrame({"time": fixture["time"], "price": price, "first": first, "second": second})
    fit = innovate.fit(
        "norton_bass",
        frame.lazy(),
        observed_columns=["first", "second"],
        model_options={
            **fixture["constructor_kwargs"],
            "launch_candidates": [3, 3.125, 3.25],
            "launch_index": 1,
            "event_candidates": [2, 2.125, 2.25],
            "event_index": 0,
        },
        fit_options={"initial_parameters": fixture["parameters"], "max_iterations": 200},
    )
    assert fit.event_profile is None
    assert fit.launch_profile is None
    profile = fit.joint_time_profile
    assert profile is not None
    assert len(profile["candidates"]) == 9
    best = profile["selected"]
    assert best == profile["candidates"][profile["selected_index"]]
    assert (best["launch_time"], best["t_event"]) == (3.125, 2.125)
    forecast = fit.predict(pl.DataFrame({"time": fixture["forecast_time"], "price": price[-1]}))
    prediction_columns = [name for name in forecast.columns if name.startswith("prediction")]
    assert forecast.select(prediction_columns).row(0) == pytest.approx(fixture["forecast_rows"][0], abs=1e-3)


def test_bootstrap_forecast_frame_preserves_series_and_null_bounds():
    result = {
        "forecast": {
            "time": [15, 16],
            "columns": ["series_1", "series_2"],
            "baseline_rows": [[13.0, 206.0], [12.0, 210.0]],
            "pointwise_intervals": [[[12.0, 14.0], None], [None, [208.0, 212.0]]],
        }
    }
    frame = innovate.bootstrap_forecast_frame(result)
    assert frame.rows() == [
        (15.0, "series_1", 13.0, 12.0, 14.0),
        (15.0, "series_2", 206.0, None, None),
        (16.0, "series_1", 12.0, None, None),
        (16.0, "series_2", 210.0, 208.0, 212.0),
    ]
    assert frame.schema["lower"] == pl.Float64


@pytest.mark.parametrize(
    ("kind", "csv", "expected"),
    [
        (
            "network_edges",
            "source,target,weight\n001,002,1\n1,002,2\n",
            {"source": ["001", "1"], "target": ["002", "002"], "weight": [1.0, 2.0]},
        ),
        (
            "competition",
            "time,value,unit_id,product_id\n0,1,001,002\n0,2,1,002\n",
            {"unit_id": ["001", "1"], "product_id": ["002", "002"]},
        ),
        ("policy_timing", "event_times,event_effects,event_labels\n0,1,001\n0,2,1\n", {"event_labels": ["001", "1"]}),
    ],
)
def test_csv_ingestion_preserves_distinct_textual_ids(tmp_path, kind, csv, expected):
    path = tmp_path / f"{kind}.csv"
    path.write_text(csv)
    provenance = {
        "source": "synthetic",
        "license": "CC-BY-4.0",
        "extraction_time": "2026-11-07T00:00:00Z",
    }
    artifact = innovate.ingest_local(path, kind, provenance=provenance)
    assert {key: artifact["dataset"][key] for key in expected} == expected
    assert innovate.execute("ingest_dataset", dataset=artifact["dataset"])["result"] == artifact

    path.write_text(csv.replace(",001,", ",,").replace(",001\n", ",\n").replace("\n001,", "\n,"))
    with pytest.raises(innovate.NativeError) as exc:
        innovate.ingest_local(path, kind, provenance=provenance)
    assert "identifier_labels" in exc.value.validation["failed"]


def test_kairos_resource_comparison_arrow_preserves_signed_totals():
    fixture = Path(__file__).parents[2] / "rust/crates/innovate-core/tests/fixtures/kairos_resource_comparisons.json"
    response = innovate.execute("execute_scenario_grid", **json.loads(fixture.read_text()))
    frame = innovate.scenario_comparisons(response)
    assert frame["comparison_index"].to_list() == [0, 1]
    assert frame["successful"].to_list() == [True, False]
    success, failure = map(json.loads, frame["comparison_json"])
    metrics = success["result"]["resource_differences"]
    assert metrics["configured_slot_ticks"] == -3
    assert metrics["peak_in_service"] == 1
    assert metrics["horizon_ticks"] == 11
    assert metrics["ticks_per_unit"] == 1
    assert failure["error"]["code"] == "invalid_comparison"
    assert failure["result"] is None

    payload = json.loads(fixture.read_text())
    for scenario in payload["scenarios"]:
        scenario["simulation"]["inputs"]["time"] = [0, 7]
    response = innovate.execute("execute_scenario_grid", **payload)
    row = json.loads(innovate.scenario_comparisons(response)["comparison_json"][0])
    totals = row["result"]["resource_differences"]
    assert totals["waiting_at_horizon"] == 1
    assert totals["in_service_at_horizon"] == 2


@pytest.mark.parametrize("suffix", ["csv", "parquet", "ipc"])
def test_local_ingestion_defaults_optional_policy_labels(tmp_path, suffix):
    frame = pl.DataFrame({"event_times": [0, 2], "event_effects": [0.0, -0.5]})
    path = tmp_path / f"policy.{suffix}"
    getattr(frame, f"write_{suffix}")(path)
    provenance = {
        "source": "synthetic",
        "license": "CC-BY-4.0",
        "extraction_time": "2026-11-07T00:00:00Z",
    }
    artifact = innovate.ingest_local(path, "policy_timing", provenance=provenance)
    assert artifact["dataset"]["event_labels"] == ["event_0", "event_1"]
    assert artifact["dataset"]["event_effects"] == [0.0, -0.5]
    assert innovate.execute("ingest_dataset", dataset=artifact["dataset"])["result"] == artifact


def test_local_ingestion_explicit_format_preserves_upstream_provenance(tmp_path):
    path = tmp_path / "observations.data"
    pl.DataFrame({"time": [0, 1], "adoption": [1, 2]}).write_parquet(path)
    provenance = {
        "source": "synthetic",
        "license": "CC-BY-4.0",
        "extraction_time": "2026-11-07T00:00:00Z",
        "transform_steps": ["normalize"],
        "extra": {"cohort": "α"},
    }
    artifact = innovate.ingest_local(path, "adoption", provenance=provenance, format=".PARQUET")
    actual = artifact["dataset"]["provenance"]
    assert actual["transform_steps"] == ["normalize", "load:observations.data"]
    assert actual["extra"] == {"cohort": "α", "path": str(path), "format": "parquet"}
    assert provenance["transform_steps"] == ["normalize"]
    assert provenance["extra"] == {"cohort": "α"}


def test_dataset_checksum_rejects_positional_provenance():
    positional = ["survey", "CC-BY-4.0", "2026-11-07T00:00:00Z", [], "1.0", "", "", {}]
    with pytest.raises(innovate.NativeError, match="dataset provenance must be an object") as exc:
        innovate.execute("dataset_checksum", dataset={"provenance": positional})
    assert exc.value.code == "invalid_request"


def test_legacy_dataset_artifact_reingestion_uses_recorded_encoding():
    dataset = {
        "kind": "adoption",
        "time": [0, 1],
        "adoption": [1, 2],
        "provenance": {
            "source": "survey α",
            "license": "CC-BY-4.0",
            "extraction_time": "2026-11-07T00:00:00Z",
        },
    }
    artifact = innovate.execute("ingest_dataset", dataset=dataset, encoding="python_sorted_json_v1")["result"]
    assert artifact["checksum_encoding"] == "python_sorted_json_v1"
    assert (
        innovate.execute("ingest_dataset", dataset=artifact["dataset"], encoding=artifact["checksum_encoding"])[
            "result"
        ]
        == artifact
    )
    with pytest.raises(innovate.NativeError, match="checksum does not match"):
        innovate.execute("ingest_dataset", dataset=artifact["dataset"])
    assert innovate.execute("ingest_dataset", artifact=artifact)["result"] == artifact
    artifact["dataset"]["adoption"][1] = 3
    with pytest.raises(innovate.NativeError, match="checksum does not match"):
        innovate.execute("ingest_dataset", artifact=artifact)


def test_legacy_checksum_matches_python_across_numeric_and_unicode_domains():
    rng = random.Random(420)
    values = [struct.unpack(">d", rng.getrandbits(64).to_bytes(8, "big"))[0] for _ in range(2048)]
    dataset = {
        "values": [value for value in values if math.isfinite(value)],
        "keys": {"😀": 2**64 - 1, "\ue000": -(2**63), "é": [2**53 + 1, -0.0]},
    }
    encoded = json.dumps(dataset, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    actual = innovate.execute("dataset_checksum", dataset=dataset, encoding="python_sorted_json_v1")["result"]
    assert actual["checksum"] == hashlib.sha256(encoded).hexdigest()


def test_polars_ingestion_distinguishes_directed_and_undirected_edges():
    frame = pl.DataFrame({"source": ["a", "b"], "target": ["b", "a"]})
    provenance = {
        "source": "synthetic",
        "license": "CC-BY-4.0",
        "extraction_time": "2026-11-07T00:00:00Z",
    }
    artifact = innovate.ingest_polars(frame, "network_edges", provenance=provenance, directed=True)
    assert artifact["dataset"]["directed"] is True
    with pytest.raises(innovate.NativeError, match="dataset diagnostics failed") as exc:
        innovate.ingest_polars(frame, "network_edges", provenance=provenance, directed=False)
    assert exc.value.validation["failed"] == ["duplicate_keys:source_target"]


@pytest.mark.parametrize("local_csv", [False, True])
def test_edgeless_network_dataset_round_trips(tmp_path, local_csv):
    provenance = {
        "source": "synthetic",
        "license": "CC-BY-4.0",
        "extraction_time": "2026-11-07T00:00:00Z",
    }
    frame = pl.DataFrame(schema={"source": pl.String, "target": pl.String})
    if local_csv:
        path = tmp_path / "edges.csv"
        frame.write_csv(path)
        artifact = innovate.ingest_local(path, "network_edges", provenance=provenance)
        assert artifact["dataset"]["provenance"]["extra"]["format"] == "csv"
    else:
        artifact = innovate.ingest_polars(frame, "network_edges", provenance=provenance)
    assert artifact["dataset"]["source"] == artifact["dataset"]["target"] == artifact["dataset"]["weight"] == []
    assert innovate.execute("ingest_dataset", artifact=artifact)["result"] == artifact


@pytest.mark.parametrize("local_csv", [False, True])
def test_network_ingestion_retains_kairos_incompatible_edges(tmp_path, local_csv):
    provenance = {
        "source": "synthetic",
        "license": "CC-BY-4.0",
        "extraction_time": "2026-11-07T00:00:00Z",
    }
    frame = pl.DataFrame({"source": ["a", "a"], "target": ["a", "b"], "weight": [1.0, 0.0]})
    if local_csv:
        path = tmp_path / "edges.csv"
        frame.write_csv(path)
        artifact = innovate.ingest_local(path, "network_edges", provenance=provenance)
    else:
        artifact = innovate.ingest_polars(frame, "network_edges", provenance=provenance)
    assert artifact["dataset"]["weight"] == [1.0, 0.0]
    assert artifact["validation"]["warnings"] == ["self_loops:kairos_network", "zero_weights:kairos_network"]
    assert innovate.execute("ingest_dataset", artifact=artifact)["result"] == artifact


@pytest.mark.parametrize(("estimand", "estimate", "weight"), [("ate", 6, 3 / 7), ("att", 7.25, 1 / 4)])
def test_native_stratified_treatment_contrast(estimand, estimate, weight):
    payload = {
        "method": "stratified_treatment_contrast",
        "estimand": estimand,
        "outcomes": [0, 0, 2, 10, 19, 19, 19],
        "treatment": [0, 0, 1, 0, 1, 1, 1],
        "strata": ["low"] * 3 + ["high"] * 4,
    }
    result = innovate.execute("summarize_causal", **payload)["result"]
    assert result["estimate"] == pytest.approx(estimate)
    assert result["by_stratum"]["low"]["weight"] == pytest.approx(weight)
    assert result["conditional_standard_error"] is None
    assert result["by_stratum"]["low"]["treated_fraction"] == pytest.approx(1 / 3)
    if estimand == "att":
        with_control_only = {**payload, "strata": ["control_only", *payload["strata"][1:]]}
        overlap = innovate.execute("summarize_causal", **with_control_only)["result"]
        assert overlap["excluded_control_observations"] == 1
        assert overlap["estimate"] == pytest.approx(estimate)
    replicated = innovate.execute(
        "summarize_causal",
        method="stratified_treatment_contrast",
        estimand=estimand,
        outcomes=[0, 2, 4, 6, 10, 10, 12, 12],
        treatment=[0, 0, 1, 1] * 2,
        strata=["low"] * 4 + ["high"] * 4,
    )["result"]
    assert replicated["conditional_standard_error"] == pytest.approx(2**-0.5)
    payload["strata"][-1] = "unsupported"
    with pytest.raises(innovate.NativeError, match="each stratum requires treated and control observations"):
        innovate.execute("summarize_causal", **payload)


def test_discover_treatment_contrasts():
    result = innovate.execute("summarize_causal", method="describe_treatment_contrasts")["result"]
    descriptive, adjusted, weighted, matching = result["contrasts"][:4]
    assert descriptive["adjustment"] == "none"
    assert adjusted["required_fields"] == ["outcomes", "treatment", "strata"]
    assert adjusted["estimands"] == ["ate", "att"]
    assert matching["estimands"] == ["att"]
    assert matching["required_fields"] == ["outcomes", "treatment", "scores", "caliper"]
    assert weighted["required_fields"] == ["outcomes", "treatment", "propensities"]
    payload = dict(
        method=weighted["method"], outcomes=[0, 10, 2, 12], treatment=[0, 0, 1, 1], propensities=[0.2, 0.8, 0.2, 0.8]
    )
    result = innovate.execute("summarize_causal", **payload)["result"]
    assert result["estimate"] == pytest.approx(-4)
    assert result["fixed_propensity_standard_error"] == pytest.approx(20.48**0.5)
    singleton = innovate.execute(
        "summarize_causal",
        method=weighted["method"],
        outcomes=[0, 2],
        treatment=[0, 1],
        propensities=[0.5, 0.5],
    )["result"]
    assert singleton["fixed_propensity_standard_error"] is None
    for arm in ("treated", "control"):
        diagnostics = result["weight_diagnostics"][arm]
        assert diagnostics["observations"] == 2
        assert diagnostics["maximum_normalized_weight"] == pytest.approx(0.8)
        assert diagnostics["effective_sample_size"] == pytest.approx(25 / 17)
    assert weighted["estimands"] == ["ate", "att"]
    att = innovate.execute("summarize_causal", **payload, estimand="att")["result"]
    assert att["estimate"] == pytest.approx(-41 / 17)
    assert att["weight_diagnostics"]["treated"]["maximum_normalized_weight"] == 0.5
    payload["propensities"][0] = 0
    with pytest.raises(innovate.NativeError, match="strictly between zero and one"):
        innovate.execute("summarize_causal", **payload)


def test_fitted_stratum_propensity_workflow():
    treatment = [0, 1, 0, 1, 0, 1, 1]
    fit = innovate.execute(
        "summarize_causal",
        method="estimate_stratum_propensities",
        treatment=treatment,
        strata=["low", "high", "low", "low", "high", "high", "high"],
    )["result"]
    assert fit["propensities"] == pytest.approx([1 / 3, 0.75, 1 / 3, 1 / 3, 0.75, 0.75, 0.75])
    weighted = innovate.execute(
        "summarize_causal",
        method="propensity_weighted_contrast",
        outcomes=[0, 19, 0, 2, 10, 19, 19],
        treatment=treatment,
        propensities=fit["propensities"],
    )["result"]
    assert weighted["estimate"] == pytest.approx(6)


def test_nearest_score_matching():
    methods = innovate.execute("summarize_causal", method="describe_treatment_contrasts")["result"]["contrasts"]
    matching = next(method for method in methods if method["adjustment"] == "nearest_score_with_replacement")
    payload = dict(
        method=matching["method"],
        outcomes=[0, 10, 12, 3, 99],
        treatment=[0, 0, 1, 1, 0],
        scores=[0, 2, 2, 1, 0],
        caliper=1,
    )
    result = innovate.execute("summarize_causal", **payload)["result"]
    assert result["estimate"] == pytest.approx(2.5)
    assert [pair["control_index"] for pair in result["pairs"]] == [1, 0]
    assert result["control_diagnostics"] == {"unique_controls": 2, "effective_sample_size": 2}
    assert result["conditional_standard_error"] is None
    reused = innovate.execute(
        "summarize_causal", **{**payload, "scores": [0, 2, 2, 2, 0]}, outcome_variances=[1, 4, 1, 1, 1]
    )["result"]
    assert reused["conditional_standard_error"] == pytest.approx(4.5**0.5)
    assert reused["control_diagnostics"] == {"unique_controls": 1, "effective_sample_size": 1}
    correlated = innovate.execute(
        "summarize_causal",
        **{**payload, "scores": [0, 2, 2, 2, 0]},
        outcome_variances=[1, 4, 1, 1, 1],
        outcome_covariance_factors=[[7] * 5, [0, 0, 2, 2, 0]],
    )["result"]
    assert correlated["conditional_standard_error"] == pytest.approx(math.sqrt(8.5))
    assert correlated["uncertainty_model"] == {"model": "diagonal_plus_factors", "factor_count": 2}
    payload["caliper"] = 0.5
    partial = innovate.execute("summarize_causal", **payload, allow_partial=True)["result"]
    assert partial["estimand"] == "matched_subset_att"
    assert partial["estimate"] == pytest.approx(2)
    assert partial["unmatched_treated_indices"] == [3]
    with pytest.raises(innovate.NativeError, match="requires a control within the caliper"):
        innovate.execute("summarize_causal", **payload)


def test_matching_with_cross_fitted_control_predictions():
    observations = {"outcomes": [0, 4, 2, 6], "treatment": [0, 1, 0, 1]}
    fitted = innovate.execute(
        "summarize_causal",
        method="cross_fitted_stratum_ate",
        strata=["a"] * 4,
        folds=[1, 2, 2, 1],
        **observations,
    )["result"]
    matched = innovate.execute(
        "summarize_causal",
        method="matched_treatment_contrast",
        scores=[0, 0, 1, 0],
        caliper=0,
        control_predictions=fitted["control_predictions"],
        **observations,
    )["result"]
    assert matched["estimate"] == 5
    assert matched["adjusted_estimate"] == 6
    assert matched["conditional_standard_error"] is None


def test_matching_propagates_fitted_control_uncertainty():
    result = innovate.execute(
        "summarize_causal",
        method="matched_treatment_contrast",
        outcomes=[0, 4, 2, 6],
        treatment=[0, 1, 0, 1],
        scores=[0, 0, 1, 0],
        caliper=0,
        outcome_variances=[1] * 4,
        outcome_covariance_factors=[[7] * 4, [0, 2, 0, 2]],
        fit_controls={"strata": ["a"] * 4, "folds": [1, 2, 2, 1]},
    )["result"]
    assert result["estimate"] == 5
    assert result["adjusted_estimate"] == 6
    assert result["adjusted_outcome_weights"] == [[0, -1.5], [1, 0.5], [2, 0.5], [3, 0.5]]
    assert result["conditional_standard_error"] == pytest.approx(math.sqrt(5.5))
    assert result["adjusted_standard_error"] == pytest.approx(math.sqrt(7))


def test_matching_standardized_balance():
    adjacent = math.nextafter(1.5, math.inf)
    balance = innovate.execute(
        "summarize_causal",
        method="matched_treatment_contrast",
        outcomes=[0, 0, 1, 1, 1],
        treatment=[0, 0, 1, 1, 1],
        scores=[0, 1, 0, 0, 1],
        caliper=0,
        balance_covariates={
            "age": [0, 30, 30, 30, 30],
            "constant": [1] * 5,
            "narrow": [1.5, adjacent, 1.5, adjacent, 1.5],
        },
    )["result"]["covariate_balance"]
    age = balance["age"]
    assert age["before_matching"]["standardized_mean_difference"] == pytest.approx(15 / math.sqrt(150))
    assert age["standardized_mean_difference"] == pytest.approx(20 / math.sqrt(150))
    assert balance["constant"]["standardized_mean_difference"] is None
    assert balance["narrow"]["standardized_mean_difference"] == pytest.approx(0)
    assert balance["narrow"]["before_matching"]["mean_difference"] / (adjacent - 1.5) == pytest.approx(-1 / 6)
    assert balance["narrow"]["before_matching"]["standardized_mean_difference"] == pytest.approx(
        (-1 / 6) / math.sqrt(7 / 18)
    )


def test_ridge_logistic_propensity_fit():
    fit = innovate.execute(
        "summarize_causal",
        method="fit_logistic_propensities",
        treatment=[0, 1],
        covariates=[[-2], [2]],
        ridge=0.25 / math.log(3),
    )["result"]
    assert fit["propensities"] == pytest.approx([0.25, 0.75], abs=1e-7)
    assert fit["covariate_scales"] == [2]
    prediction = innovate.execute(
        "summarize_causal", method="predict_logistic_propensities", fit=fit, covariates=[[0], [2], [-2]]
    )["result"]
    assert prediction["propensities"] == pytest.approx([0.5, 0.75, 0.25], abs=1e-7)
    score = innovate.execute(
        "summarize_causal", method="score_propensities", treatment=[0, 1, 0], propensities=prediction["propensities"]
    )["result"]
    assert score["brier_score"] == pytest.approx(0.125)
    assert score["log_loss"] == pytest.approx(-(math.log(0.5) + 2 * math.log(0.75)) / 3)
    cross_fit = innovate.execute(
        "summarize_causal",
        method="cross_fit_logistic_propensities",
        treatment=[0, 1, 1, 0],
        covariates=[[-2], [1], [2], [-1]],
        folds=[9, 4, 9, 4],
        ridge=0.25 / math.log(3),
    )["result"]
    low = 1 / (1 + math.sqrt(3))
    assert cross_fit["propensities"] == pytest.approx([0.1, 1 - low, 0.9, low], abs=1e-7)


def test_augmented_treatment_contrast():
    result = innovate.execute(
        "summarize_causal",
        method="augmented_treatment_contrast",
        outcomes=[0, 4],
        treatment=[0, 1],
        propensities=[0.5, 0.5],
        treated_predictions=[2, 2],
        control_predictions=[0, 0],
    )["result"]
    assert result["estimate"] == pytest.approx(4)
    assert result["influence_standard_error"] == pytest.approx(2)
    att = innovate.execute(
        "summarize_causal",
        method="augmented_treatment_contrast",
        estimand="att",
        outcomes=[1, 4],
        treatment=[0, 1],
        propensities=[0.5, 0.5],
        control_predictions=[0, 1],
    )["result"]
    assert att["estimand"] == "att"
    assert att["estimate"] == pytest.approx(2)
    assert att["influence_standard_error"] == pytest.approx(2)


def test_cross_fitted_stratum_ate():
    payload = dict(
        method="cross_fitted_stratum_ate",
        outcomes=[0, 4, 2, 6],
        treatment=[0, 1, 0, 1],
        strata=["a"] * 4,
        folds=[1, 2, 2, 1],
    )
    result = innovate.execute("summarize_causal", **payload)["result"]
    assert result["control_predictions"] == [2, 0, 0, 2]
    assert result["treated_predictions"] == [4, 6, 6, 4]
    clustered = innovate.execute("summarize_causal", **payload, clusters=["a", "b", "b", "a"])["result"]["contrast"]
    assert clustered["cluster_count"] == 2
    assert clustered["influence_standard_error"] == pytest.approx(2)
    att_payload = {**payload, "method": "cross_fitted_stratum_att"}
    att = innovate.execute("summarize_causal", **att_payload, clusters=["a", "b", "b", "a"])["result"]["contrast"]
    assert att["estimand"] == "att"
    assert att["estimate"] == pytest.approx(4)
    assert att["cluster_count"] == 2
    assert att["influence_standard_error"] == pytest.approx(2)
    for options in (payload, att_payload):
        result_with_guard = innovate.execute(
            "summarize_causal",
            **options,
            clusters=["a", "b", "b", "a"],
            minimum_training_arm_count=1,
            minimum_training_arm_clusters=1,
        )["result"]
        assert result_with_guard["overlap"]["min_training_arm_clusters"] == 1
        assert result_with_guard["contrast"]["cluster_degrees_of_freedom"] == 1
        with pytest.raises(innovate.NativeError, match="falls below minimum_training_arm_clusters"):
            innovate.execute(
                "summarize_causal",
                **options,
                clusters=["a", "b", "b", "a"],
                minimum_training_arm_clusters=2,
            )
    with pytest.raises(innovate.NativeError, match="single cross-fitting fold"):
        innovate.execute("summarize_causal", **payload, clusters=["a", "a", "b", "b"])
    assert result["contrast"]["estimate"] == pytest.approx(4)


def test_cross_fitted_stratum_ate_adjusts_for_unequal_treatment_prevalence():
    outcomes = [0, 0, 2, 10, 19, 19, 19] * 2
    treatment = [0, 0, 1, 0, 1, 1, 1] * 2
    strata = (["low"] * 3 + ["high"] * 4) * 2
    result = innovate.execute(
        "summarize_causal",
        method="cross_fitted_stratum_ate",
        outcomes=outcomes,
        treatment=treatment,
        strata=strata,
        folds=[1] * 7 + [2] * 7,
    )["result"]
    assert result["propensities"] == pytest.approx(([1 / 3] * 3 + [3 / 4] * 4) * 2)
    assert result["contrast"]["estimate"] == pytest.approx(6)
    unadjusted = innovate.execute(
        "summarize_causal",
        method="treatment_contrast",
        outcomes=outcomes,
        treatment=treatment,
    )["result"]
    assert unadjusted["overall"]["estimate"] == pytest.approx(137 / 12)
    att = innovate.execute(
        "summarize_causal",
        method="cross_fitted_stratum_att",
        outcomes=outcomes,
        treatment=treatment,
        strata=strata,
        folds=[1] * 7 + [2] * 7,
    )["result"]
    assert att["contrast"]["estimand"] == "att"
    assert att["contrast"]["estimate"] == pytest.approx(7.25)


def test_clustered_augmented_treatment_contrast():
    payload = dict(
        method="augmented_treatment_contrast",
        outcomes=[0, 4] * 2,
        treatment=[0, 1] * 2,
        propensities=[0.5] * 4,
        treated_predictions=[2] * 4,
        control_predictions=[0] * 4,
    )
    independent = innovate.execute("summarize_causal", **payload)["result"]
    clustered = innovate.execute("summarize_causal", **payload, clusters=["a", "b"] * 2)["result"]
    assert clustered["estimate"] == independent["estimate"] == pytest.approx(4)
    assert clustered["cluster_count"] == 2
    assert clustered["influence_standard_error"] == pytest.approx(2)
    with pytest.raises(innovate.NativeError, match="requires at least two clusters"):
        innovate.execute("summarize_causal", **payload, clusters=["a"] * 4)


def test_balanced_panel_event_study():
    payload = dict(
        method="panel_event_study",
        periods=[-2, -1, 0],
        baseline_index=1,
        treated=[[1, 2, 7], [3, 4, 11]],
        control=[[0, 1, 3], [4, 5, 7]],
    )
    pre, reference, post = innovate.execute("summarize_causal", **payload)["result"]["contrasts"]
    assert pre["estimate"] == pytest.approx(0)
    assert reference["reference"] is True
    assert post["estimate"] == pytest.approx(4)
    assert post["standard_error"] == pytest.approx(1)


def test_staggered_panel_att():
    payload = dict(
        method="staggered_panel_att", periods=[0, 1, 2], outcomes=[[0, 5, 7], [10, 11, 16]], adoption_indices=[1, 2]
    )
    result = innovate.execute("summarize_causal", **payload)["result"]
    first, second, third = result["contrasts"]
    onset, later = result["event_time"]
    assert onset["unsupported_cohorts"] == 1
    assert later["supported_cohort_att"] is None
    assert first["estimate"] == pytest.approx(4)
    assert first["n_control"] == 1
    assert all(row["estimate"] is None and row["n_control"] == 0 for row in (second, third))
    assert first["standard_error"] is None
    replicated = innovate.execute(
        "summarize_causal",
        **{
            **payload,
            "outcomes": [[0, 5, 7], [0, 7, 9], [10, 12, 16], [10, 12, 16]],
            "adoption_indices": [1, 1, 2, None],
        },
    )["result"]
    assert replicated["contrasts"][0]["estimate"] == pytest.approx(4)
    assert replicated["contrasts"][0]["standard_error"] == pytest.approx(1)
    assert replicated["contrasts"][1]["n_control"] == 1
    assert replicated["event_time"][0]["supported_cohort_att"] == pytest.approx(8 / 3)
    assert replicated["event_time"][0]["standard_error"] is None
    shared = innovate.execute(
        "summarize_causal",
        **{
            **payload,
            "outcomes": [[0, 5, 5]] * 2 + [[0, 1, 6]] * 2 + [[0, 0, 0], [0, 2, 4]],
            "adoption_indices": [1, 1, 2, 2, None, None],
        },
        include_event_time_covariance=True,
        simultaneous_bands={"draws": 100, "seed": 7},
    )["result"]
    error = shared["event_time"][0]["standard_error"]
    assert error == pytest.approx((1 / math.sqrt(48) + 1 / math.sqrt(8)) * math.sqrt(2))
    covariance = shared["event_time_covariance"]
    assert covariance[0] == pytest.approx([error**2, 2 * error])
    assert covariance[1] == pytest.approx([2 * error, 4])
    bands = shared["simultaneous_bands"]
    assert bands["critical_value"] == pytest.approx(math.sqrt(2))
    assert bands["intervals"][1]["interval"] == pytest.approx(
        {"lower": 3 - 2 * math.sqrt(2), "upper": 3 + 2 * math.sqrt(2)}
    )
    sparse = innovate.execute("summarize_causal", **payload, include_event_time_covariance=True)["result"]
    assert sparse["event_time_covariance"] == [[None, None], [None, None]]
    balanced = innovate.execute(
        "summarize_causal",
        **{
            **payload,
            "outcomes": [[0, 5, 7], [0, 7, 9], [10, 12, 16], [10, 12, 16]],
            "adoption_indices": [1, 1, 2, None],
        },
        balanced_horizon=1,
    )["result"]
    assert balanced["balanced_event_time"] == {"horizon": 1, "included_cohorts": 1, "excluded_cohorts": 1}
    assert [row["supported_cohort_att"] for row in balanced["event_time"]] == pytest.approx([4, 2])
    assert balanced["contrasts"] == replicated["contrasts"]
    placebo = innovate.execute(
        "summarize_causal",
        **{**payload, "outcomes": [[0, 5, 7], [10, 11, 16], [0, 2, 4]], "adoption_indices": [1, 2, None]},
        include_pre_trends=True,
    )["result"]["pre_trends"]
    assert len(placebo) == 1
    assert placebo[0]["estimate"] == pytest.approx(1)
    assert placebo[0]["n_control"] == 1
    assert placebo[0]["baseline_period"] == 1
    comparator = {**payload, "outcomes": [[0, 5, 7], [10, 11, 16], [0, 2, 4]], "adoption_indices": [1, 2, None]}
    for group, count, estimate in [("not_yet_treated", 2, 3.5), ("never_treated", 1, 3)]:
        onset = innovate.execute("summarize_causal", **comparator, control_group=group)["result"]["contrasts"][0]
        assert onset["n_control"] == count
        assert onset["estimate"] == pytest.approx(estimate)


def test_staggered_long_panel_att():
    frame = pl.DataFrame(
        {
            "unit_ids": ["z", "a", "z", "a", "a", "z"],
            "time": [20, 10, 10, 30, 20, 30],
            "outcomes": [11, 0, 10, 7, 5, 16],
        }
    )
    payload = frame.to_dict(as_series=False)
    result = innovate.execute(
        "summarize_causal", method="staggered_long_panel_att", **payload, adoption_times={"a": 20, "z": 30}
    )["result"]
    assert result["unit_order"] == ["a", "z"]
    assert result["contrasts"][0]["estimate"] == pytest.approx(4)
    assert innovate.staggered_att_polars(frame, {"a": 20, "z": 30}) == result
    assert innovate.staggered_att_polars(frame.lazy(), {"a": 20, "z": 30}) == result
    payload["time"][0] = 10
    with pytest.raises(innovate.NativeError, match="duplicate unit and period"):
        innovate.execute(
            "summarize_causal", method="staggered_long_panel_att", **payload, adoption_times={"a": 20, "z": 30}
        )


def test_scenario_grid_ranking():
    scenarios = [
        {
            "id": key,
            "model_key": "logistic",
            "simulation": {
                "inputs": {"time": [1, 2, 4]},
                "parameters": {"L": capacity, "k": 1, "x0": 1},
            },
        }
        for key, capacity in [("base", 100), ("alt", 200), ("tie", 200)]
    ]
    result = innovate.execute("execute_scenario_grid", scenarios=scenarios, rank_scenarios=["tie", "base", "alt"])[
        "result"
    ]
    assert [(row["id"], row["place"]) for row in result["ranking"]["rows"]] == [
        ("tie", 1),
        ("alt", 1),
        ("base", 3),
    ]
    area = innovate.execute(
        "execute_scenario_grid", scenarios=scenarios, rank_scenarios=["tie", "base", "alt"], rank_metric="adoption_area"
    )["result"]
    first, second, third = area["executions"][1]["result"]["values"]
    assert area["ranking"]["metric"] == "adoption_area"
    assert area["ranking"]["rows"][0]["adoption_area"] == pytest.approx((first + second) / 2 + second + third)
    threshold = innovate.execute(
        "execute_scenario_grid",
        scenarios=scenarios,
        rank_scenarios=["tie", "base", "alt"],
        rank_metric="time_to_threshold",
        rank_threshold=150,
    )["result"]["ranking"]["rows"]
    expected_crossing = 2 + 2 * (150 - second) / (third - second)
    assert [(row["id"], row["place"]) for row in threshold] == [("tie", 1), ("alt", 1), ("base", None)]
    assert threshold[0]["time_to_threshold"] == pytest.approx(expected_crossing)
    assert threshold[2]["time_to_threshold"] is None
    scenarios.append({**scenarios[0], "id": "base2"})
    comparison = {
        "pairs": [{"baseline": "base", "alternative": "alt"}, {"baseline": "tie", "alternative": "base2"}],
        "interval_mass": 0.5,
        "evaluation_metric": "adoption_area",
    }
    paired = innovate.execute("execute_scenario_grid", scenarios=scenarios, comparisons=[comparison])["result"][
        "comparisons"
    ][0]["result"]
    assert paired["evaluation_horizon"] == [1, 4]
    expected = area["ranking"]["rows"][0]["adoption_area"] / 2
    assert paired["differences"] == pytest.approx([expected, -expected])
    assert paired["mean_difference"] == pytest.approx(0)
    comparison.pop("evaluation_metric")
    comparison["include_path_covariance"] = True
    endpoint = innovate.execute("execute_scenario_grid", scenarios=scenarios, comparisons=[comparison])["result"][
        "comparisons"
    ][0]["result"]
    rows = endpoint["path_summary"]["rows"]
    assert [row["time"] for row in rows] == [1, 2, 4]
    for field in ("mean_difference", "mean_standard_error", "empirical_interval"):
        assert rows[-1][field] == endpoint[field]
    errors = [row["mean_standard_error"] for row in rows]
    for row, error in zip(endpoint["path_covariance"], errors, strict=True):
        assert row == pytest.approx([error * other for other in errors])

    for scenario in scenarios:
        scenario["simulation"]["inputs"]["time"] = list(range(128))
    boundary = innovate.execute("execute_scenario_grid", scenarios=scenarios, comparisons=[comparison])["result"][
        "comparisons"
    ][0]["result"]
    assert len(boundary["path_covariance"]) == 128
    assert all(len(row) == 128 for row in boundary["path_covariance"])

    for scenario in scenarios:
        scenario["simulation"]["inputs"]["time"] = list(range(129))
    rejected = innovate.execute("execute_scenario_grid", scenarios=scenarios, comparisons=[comparison])["result"][
        "comparisons"
    ][0]
    assert rejected["result"] is None
    assert rejected["error"] == {
        "code": "invalid_comparison",
        "message": "ensemble path covariance supports at most 128 sample times",
    }

    comparison.pop("include_path_covariance")
    comparison["include_path_summary"] = True
    summary = innovate.execute("execute_scenario_grid", scenarios=scenarios, comparisons=[comparison])["result"][
        "comparisons"
    ][0]["result"]
    assert len(summary["path_summary"]["rows"]) == 129
    assert "path_covariance" not in summary

    comparison.pop("include_path_summary")
    comparison.update(interval_mass=0.9, exact_sign_flip=True, bootstrap={"draws": 1000, "seed": 42})
    inference = innovate.execute("execute_scenario_grid", scenarios=scenarios, comparisons=[comparison])["result"][
        "comparisons"
    ][0]["result"]
    assert inference["sign_flip"]["assignments"] == 4
    assert inference["sign_flip"]["extreme_assignments"] == 4
    assert inference["sign_flip"]["p_value"] == 1
    assert inference["bootstrap"]["mean_interval"] == pytest.approx([-100, 100])
    assert inference["bootstrap"]["seed"] == 42


def test_adopter_classification_preserves_irregular_time_grids():
    origin = 1.5
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    for time in (
        [origin, math.nextafter(origin, math.inf), origin + 4 * math.ulp(origin)],
        [-largest, -largest / 2, largest],
    ):
        result = innovate.execute("categorize_adopters", time=time, adoption_rate=[1, 2, 1])["result"]
        assert result["time"] == time
        assert result["category"] == ["Early Adopters", "Early Majority", "Laggards"]

    for origin in (0, 1024):
        time = list(range(origin, origin + 4))
        result = innovate.execute("categorize_adopters", time=time, adoption_rate=[0, 1, 0, 1])["result"]
        assert result["category"] == ["Innovators", "Early Adopters", "Early Majority", "Late Majority"]

    smallest = math.ulp(0.0)
    for time, rates, categories in (
        ([0, 1e-317, 1], [largest, 0, smallest], ["Early Majority", "Late Majority", "Laggards"]),
        (
            [-largest, 0, smallest, 2 * smallest, largest],
            [0, 1, 0, 1, 0],
            ["Innovators", "Early Adopters", "Early Majority", "Late Majority", "Laggards"],
        ),
    ):
        result = innovate.execute("categorize_adopters", time=time, adoption_rate=rates)["result"]
        assert result["time"] == time
        assert result["adoption_rate"] == rates
        assert result["category"] == categories


def test_adopter_classification_matches_exact_weighted_moments():
    rng = random.Random(401)
    cases = [(sorted(rng.sample(range(-300, 301), 16)), [rng.randrange(31) for _ in range(16)]) for _ in range(512)]
    cases.extend((list(range(5)), weights) for weights in product(range(5), repeat=5) if any(weights))
    for time, rates in cases:
        total = sum(rates)
        mean = Fraction(sum(t * w for t, w in zip(time, rates, strict=True)), total)
        variance = sum((t - mean) ** 2 * w for t, w in zip(time, rates, strict=True)) / total

        expected = []
        for t in time:
            delta = t - mean
            squared = delta**2
            if delta <= 0 and squared >= 4 * variance:
                expected.append("Innovators")
            elif delta <= 0 and squared >= variance:
                expected.append("Early Adopters")
            elif delta <= 0:
                expected.append("Early Majority")
            elif squared <= variance:
                expected.append("Late Majority")
            else:
                expected.append("Laggards")
        result = innovate.execute("categorize_adopters", time=time, adoption_rate=rates)["result"]
        assert result["category"] == expected, (time, rates)


def test_kairos_tick_checkpoint_resume():
    fixture = Path(__file__).parents[2] / "rust/crates/innovate-core/tests/fixtures/kairos_network_checkpoint.json"
    for engine in ("kairos_des", "kairos_abm", "kairos_network"):
        payload = json.loads(fixture.read_text())
        if engine != "kairos_network":
            payload["simulation_options"] = {
                "engine": engine,
                "seed": 42,
                "population": 6,
                "ticks_per_unit": 1,
            }
        expected = innovate.execute("simulate_model", "bass", **payload)["result"]
        options = payload["simulation_options"]
        for tick in (0, 4, 6):
            options["pause_at_tick"] = tick
            paused = innovate.execute("simulate_model", "bass", **payload)["result"]
            assert paused["status"] == "paused"
            del options["pause_at_tick"]
            options["resume_checkpoint"] = paused["checkpoint"]
            assert innovate.execute("simulate_model", "bass", **payload)["result"] == expected
            del options["resume_checkpoint"]


def test_paused_scenario_polars_export_preserves_checkpoint():
    fixture = Path(__file__).parents[2] / "rust/crates/innovate-core/tests/fixtures/kairos_network_checkpoint.json"
    simulation = json.loads(fixture.read_text())
    paused = {
        **simulation,
        "simulation_options": {**simulation["simulation_options"], "pause_at_tick": 6},
    }
    response = innovate.execute(
        "execute_scenario_grid",
        scenarios=[
            {"id": "complete", "model_key": "bass", "simulation": simulation},
            {"id": "paused", "model_key": "bass", "simulation": paused},
        ],
    )
    frame = innovate.scenario_executions(response)
    assert frame["scenario_id"].to_list() == ["complete", "paused"]
    assert frame["successful"].to_list() == [True, False]
    assert [json.loads(row) for row in frame["execution_json"]] == response["result"]["executions"]


def test_matching_with_fitted_scores():
    result = innovate.execute(
        "summarize_causal",
        method="matched_treatment_contrast",
        outcomes=[0, 4, 2, 6],
        treatment=[0, 1, 0, 1],
        caliper=0,
        fit_scores={"covariates": [[0]] * 4, "folds": [1, 1, 2, 2], "ridge": 1},
    )["result"]
    assert result["estimate"] == 5
    assert result["score_fit"]["propensities"] == [0.5] * 4
    assert result["score_uncertainty_scope"] == "conditional_on_fitted_scores_and_pairs"
    assert result["match_diagnostics"]["minimum_matched_caliper_slack"] == 0
    assert result["match_diagnostics"]["minimum_runner_up_distance_gap"] == 0


def test_matching_score_draw_mixture():
    sensitivity = innovate.execute(
        "summarize_causal",
        method="matched_treatment_contrast",
        outcomes=[0, 4, 2],
        treatment=[0, 1, 0],
        scores=[0, 0, 1],
        caliper=0,
        score_draws=[[0, 0, 1], [1, 0, 0]],
        outcome_variances=[0, 1, 6],
        control_predictions=[0, 0, 2],
    )["result"]["score_sensitivity"]
    assert [draw["estimate"] for draw in sensitivity["draws"]] == [4, 2]
    assert sensitivity["same_treated_subset"]
    assert sensitivity["mixture_standard_deviation"] == pytest.approx(math.sqrt(5))
    assert sensitivity["adjusted_mixture_standard_deviation"] == pytest.approx(2)


def test_matching_score_refits():
    result = innovate.execute(
        "summarize_causal",
        method="matched_treatment_contrast",
        outcomes=[0, 4, 2, 6, 8, 12, 10, 14],
        treatment=[0, 1] * 4,
        caliper=1,
        outcome_variances=[1] * 8,
        fit_scores={"covariates": [[0]] * 8, "folds": [1] * 4 + [2] * 4, "ridge": 1},
        score_refit_omit_indices=[0, 4],
    )["result"]["score_sensitivity"]
    assert result["omitted_training_indices"] == [0, 4]
    assert [pair["control_index"] for pair in result["draws"][0]["pairs"]] == [0, 0, 4, 4]
    assert result["mixture_standard_deviation"] is None


def test_matching_retains_insufficient_score_draw():
    sensitivity = innovate.execute(
        "summarize_causal",
        method="matched_treatment_contrast",
        outcomes=[0, 4],
        treatment=[0, 1],
        scores=[0, 0],
        caliper=0,
        score_draws=[[0, 0], [0, 1]],
        outcome_variances=[1, 1],
    )["result"]["score_sensitivity"]
    assert [draw["status"] for draw in sensitivity["draws"]] == ["matched", "insufficient_matches"]
    assert sensitivity["draws"][0]["estimate"] == 4
    assert sensitivity["mixture_standard_deviation"] is None


def test_matching_score_bootstrap():
    payload = dict(
        method="matched_treatment_contrast",
        outcomes=[0, 4, 2, 6, 8, 12, 10, 14],
        treatment=[0, 1] * 4,
        caliper=1,
        outcome_variances=[1] * 8,
        fit_scores={"covariates": [[0], [3], [2], [5], [1], [4], [3], [6]], "folds": [1] * 4 + [2] * 4, "ridge": 1},
        score_bootstrap={"draws": 8, "seed": 17},
    )
    result = innovate.execute("summarize_causal", **payload)["result"]["score_sensitivity"]
    assert result == innovate.execute("summarize_causal", **payload)["result"]["score_sensitivity"]
    assert result["seed"] == 17 and len(result["draws"]) == 8
    assert result["mixture_standard_deviation"] > 0


def test_extension_dataset_provider():
    selection = dict(name="provider-fixture", expected_version="1", extension_point="dataset_provider")
    innovate.execute(
        "register_extension_manifest",
        manifest=dict(
            name=selection["name"],
            version="1",
            entrypoint="fixture:provide",
            stability="stable",
            extension_points=["dataset_provider"],
        ),
    )
    try:
        dataset = dict(
            kind="adoption",
            time=[0, 1],
            adoption=[1, 2],
            provenance=dict(source="fixture", license="CC-BY-4.0", extraction_time="2026-11-07T00:00:00Z"),
        )
        artifact = innovate.invoke_registered_extension(lambda _, data: data, selection, dataset)
        assert artifact["validation"]["ok"]
        assert artifact["dataset"]["provenance"]["checksum"] == (
            "fada5d71f8981fa746be73b26de45ad470fa4e74f6a7d7a48234f4f50de0a76a"
        )
        with pytest.raises(innovate.NativeError, match="requires dataset provenance"):
            innovate.invoke_registered_extension(lambda *_: {"kind": "adoption"}, selection, None)
    finally:
        innovate.execute("unregister_extension_manifest", name=selection["name"], expected_version="1")


def test_empirical_prediction_intervals():
    response = innovate.execute(
        "calibrate_prediction_intervals",
        time=[1, 2, 3],
        observed=[11, 18, 35],
        predicted=[10, 20, 30],
        confidence=0.5,
        holdout=[0, 1, 0],
    )
    result = response["result"]
    assert response["metadata"]["runtime"] == "rust_native"
    assert result["lower"] == [8, 18, 28]
    assert result["upper"] == [12, 22, 32]
    assert result["diagnostics"]["holdout_coverage"] == 1.0


def test_split_conformal_prediction_intervals():
    result = innovate.execute(
        "calibrate_prediction_intervals",
        time=[1, 2, 3, 4],
        observed=[11, 12, 13, 100],
        predicted=[10, 10, 10, 10],
        confidence=0.5,
        holdout=[0, 0, 0, 1],
        method="split_conformal",
    )["result"]
    assert result["metadata"]["interval_half_width"] == 2.0
    assert result["diagnostics"]["holdout_coverage"] == 0.0


def test_split_conformal_calibrates_an_independently_fitted_forecast():
    time = list(range(24))
    curve = [100.0 / (1.0 + math.exp(-0.6 * (t - 8))) for t in time]
    fit = innovate.fit("logistic", pl.DataFrame({"time": time[:20], "observed": curve[:20]}))
    forecast_time = time[20:]
    predicted = fit.predict(pl.DataFrame({"time": forecast_time}))["prediction"].to_list()
    observed = [value + offset for value, offset in zip(curve[20:], [1, -2, 3, 10], strict=True)]
    # Three calibration scores at 50% confidence select the second order statistic.
    reference_width = sorted(abs(observed[i] - predicted[i]) for i in range(3))[1]
    result = innovate.execute(
        "calibrate_prediction_intervals",
        time=forecast_time,
        observed=observed,
        predicted=predicted,
        confidence=0.5,
        holdout=[0, 0, 0, 1],
        method="split_conformal",
    )["result"]
    assert result["metadata"]["interval_half_width"] == reference_width
    assert result["lower"] == pytest.approx([value - reference_width for value in predicted])
    assert result["upper"] == pytest.approx([value + reference_width for value in predicted])
    assert result["diagnostics"]["holdout_coverage"] == 0.0


def test_stability_tier_guidance():
    result = innovate.execute("describe_stability_tier", stability=" Beta ")["result"]
    assert result["stability"] == "provisional"
    assert "active iteration" in result["guidance"]
    with pytest.raises(innovate.NativeError, match="unknown extension stability tier"):
        innovate.execute("describe_stability_tier", stability="unknown")


def test_policy_scenario_compares_supplied_paths():
    result = innovate.execute(
        "summarize_causal",
        method="policy_scenario",
        time=[1, 2, 3],
        baseline=[10, 20, 30],
        intervention=[10, 25, 45],
    )["result"]
    assert result["difference"] == [0, 5, 15]
    assert result["metadata"]["incremental_effect"] == 20
    assert result["metadata"]["relative_lift_final"] == 0.5


def test_policy_specification_rejects_positional_contracts():
    with pytest.raises(innovate.NativeError, match="policy specification contracts must be objects"):
        innovate.execute(
            "summarize_causal",
            method="policy_specification",
            intervention={"name": "Pilot", "timing": "post", "comparator": "control"},
            causal_model=["Evaluation", "policy", "adoption", ["age"], None, {}],
        )


def test_compose_regime_ensemble():
    response = innovate.execute(
        "compose_regime_ensemble",
        time=[1, 2],
        predictions={"a": [1, 2], "b": [3, 4]},
        weights={"a": 1, "b": 3},
        observed=[2.5, 3.5],
    )
    assert response["result"]["mean"] == [2.5, 3.5]
    assert response["result"]["diagnostics"]["rmse"] == 0


def test_diagnostics_artifact_tables_preserve_polars_shape():
    tables = innovate.execute(
        "diagnostics_artifact_tables",
        artifacts={
            "ordered": {"columns": ["z", "a"], "rows": [{"a": 1, "z": 2}, {"a": 3}, False]},
            "single": {"columns": ["value"], "rows": [{"value": 4}, {}]},
        },
    )["result"]["tables"]
    ordered = pl.DataFrame(tables["ordered"]["rows"], schema=tables["ordered"]["columns"], orient="row")
    single = pl.DataFrame(tables["single"]["rows"], schema=tables["single"]["columns"], orient="row")
    assert ordered.columns == ["z", "a"]
    assert ordered.rows() == [(2, 1), (None, 3), (None, None)]
    assert single.rows() == [(4,), (None,)]


def test_streaming_update_and_capability_discovery():
    capabilities = innovate.execute("list_advanced_capabilities")["result"]["capabilities"]
    streaming = next(item for item in capabilities if item["key"] == "streaming_update")
    assert streaming["supports_incremental"] is True
    result = innovate.execute(
        streaming["operation"],
        previous_time=[1, 2],
        previous_observed=[2, 4],
        new_time=[3],
        new_observed=[6],
    )["result"]
    assert result["mean"] == [2, 4, 6]
    assert result["diagnostics"]["growth_rate"] == 0.5


def test_residual_svg_export():
    result = innovate.execute(
        "render_residual_diagnostics_svg",
        time=[1, 2, 3, 4],
        residuals=[-1, 1, -1, 1],
        config={"lags": 2, "acf_only": True},
    )["result"]
    assert result["series"]["acf"] == [1, -0.75, 0.5]
    assert "<svg" in result["svg"]
    assert "Partial autocorrelation" not in result["svg"]


def test_oecd_csv_ingestion_preserves_period_codes(monkeypatch):
    def fetch(operation, **payload):
        assert operation == "fetch_oecd_dataset"
        assert payload["start_date"] == "1960"
        assert payload["end_date"] == "2020"
        return {"result": {"csv": "REF_AREA,TIME_PERIOD,OBS_VALUE\nAUS,2020,1.5\nAUS,2021,\n"}}

    monkeypatch.setattr(innovate, "execute", fetch)
    frame = innovate.fetch_oecd_dataset("OECD.SDD.STES,DSD_STES@DF_CLI", ".M.LI...AA...H")
    assert frame["TIME_PERIOD"].to_list() == ["2020", "2021"]
    assert frame["OBS_VALUE"].to_list() == [1.5, None]


@pytest.mark.skipif(os.getenv("INNOVATE_TEST_LIVE_OECD") != "1", reason="requires the live OECD SDMX endpoint")
def test_live_oecd_fetch_returns_polars_observations():
    observations = innovate.fetch_oecd_dataset(
        "OECD.SDD.STES,DSD_STES@DF_CLI", ".M.LI...AA...H", start_date="2023-02", end_date="2023-03"
    )
    assert observations.height > 0
    assert set(observations["TIME_PERIOD"]) <= {"2023-02", "2023-03"}
    assert observations.schema["OBS_VALUE"] == pl.Float64


def test_competition_equilibrium_summaries():
    operation = "summarize_competition_equilibrium"
    shares = innovate.execute(operation, model="multi_product", market_potentials=[100, 300])["result"]
    assert shares["shares"] == {"product_0": 0.25, "product_1": 0.75}
    assert shares["interaction_adjusted"] is False
    boundary = innovate.execute(operation, model="lotka_volterra")["result"]
    assert (boundary["N1_eq"], boundary["N2_eq"]) == (1000, 0)


def test_historical_competition_cross_elasticity():
    time = [0, 1, 2]
    result = innovate.execute(
        "competition_cross_elasticity",
        time=time,
        p=[0.1, 0.2],
        q=[[0, 0.4], [0, 0]],
        market_potentials=[50, 100],
    )["result"]
    first, second = result["elasticity"]
    expected = [40 * (1 - math.exp(-0.2 * t)) / (50 + 1e-10) for t in time]
    assert first == pytest.approx(expected, abs=1e-7, rel=0)
    assert second == [0, 0, 0]


def test_historical_interaction_matrix_prediction():
    time = [3, 4, 5]
    result = innovate.execute(
        "predict_competition_interaction_matrix",
        time=time,
        p=[0.1, 0.2],
        q=[[0, -0.05], [0, 0]],
        market_potentials=[50, 100],
    )["result"]
    assert result["axis_order"] == ["time", "product"]
    assert result["time"] == time
    for t, levels, rates in zip(time, result["rows"], result["rates"], strict=True):
        elapsed = t - time[0]
        survival = math.exp(-0.2 * elapsed)
        remaining = 50 * math.exp(-0.05 * elapsed - 0.25 * (1 - survival))
        assert levels == pytest.approx([50 - remaining, 100 * (1 - survival)], abs=1e-6, rel=0)
        assert rates == pytest.approx([(0.05 + 0.05 * survival) * remaining, 20 * survival], abs=1e-6, rel=0)


def test_historical_interaction_matrix_pooled_score():
    payload = {"time": [0, 1], "p": [0, 0], "q": [[0, 0], [0, 0]], "market_potentials": [50, 100]}
    for observed, expected in [([[1, 2], [3, 4]], -5), ([[0.1, 0.1], [0.1, 0.1]], 0)]:
        result = innovate.execute("predict_competition_interaction_matrix", observed=observed, **payload)["result"]
        assert result["r_squared"] == expected


def test_historical_interaction_matrix_fit():
    time = [i / 2 for i in range(24)]
    observed = [[50 * (1 - math.exp(-0.1 * t)), 100 * (1 - math.exp(-0.2 * t))] for t in time]
    result = innovate.execute(
        "fit_competition_interaction_matrix",
        time=time,
        p=[0.11, 0.19],
        q=[[0.001, -0.001], [0.001, 0.001]],
        market_potentials=[49, 101],
        observed=observed,
    )["result"]
    assert result["predictions"]["r_squared"] > 0.999999
    assert result["identifiability"]["rank"] == 8
    assert result["parameters"]["market_potentials"] == pytest.approx([50, 100], abs=0.1)


def test_historical_interaction_matrix_automatic_fit():
    time = [i / 2 for i in range(32)]
    survival = [math.exp(-0.2 * t) for t in time]
    observed = [
        [50 * (1 - math.exp(-0.05 * t - 0.25 * (1 - s))), 100 * (1 - s)] for t, s in zip(time, survival, strict=True)
    ]
    result = innovate.execute("fit_competition_interaction_matrix", time=time, observed=observed)["result"]
    assert result["predictions"]["r_squared"] > 0.999999
    assert result["parameters"]["q"][0][1] == pytest.approx(-0.05, abs=0.005, rel=0)


def test_norton_historical_transient_crossing():
    result = innovate.execute(
        "summarize_model",
        "norton_bass",
        inputs={"time": [0, 25]},
        parameters={
            "p1": 0.03,
            "q1": 0.2,
            "m1": 1000,
            "p2": 0.02,
            "q2": 0.3,
            "m2": 1500,
            "p3": 0.04,
            "q3": 0.25,
            "m3": 600,
        },
        constructor_kwargs={"n_generations": 3, "initial_values": [1e-6, 0, 0]},
    )["result"]
    row = result["threshold_diagnostics"]["net_level_milestones"]["generations"][0]
    assert row["takeoff"] is None
    assert 2.2 < row["refined_crossings"]["takeoff"] < 5
    assert row["refined_crossings"]["median"] is None


def test_norton_refined_rate_peak():
    result = innovate.execute(
        "summarize_model",
        "norton_bass",
        inputs={"time": [0, 20]},
        parameters={"p1": 0.02, "q1": 0.3, "m1": 100},
    )["result"]
    peak = result["threshold_diagnostics"]["refined_rate_peaks"]["generations"][0]
    assert peak["peak_time"] == pytest.approx(math.log(15) / 0.32, abs=1e-5, rel=0)
    assert peak["peak_rate"] == pytest.approx(100 * 0.32**2 / 1.2, abs=1e-8, rel=0)
