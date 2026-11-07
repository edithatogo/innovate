"""Independent fixed-event ecosystem references on Python 3.15."""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

fixtures = Path("rust/crates/innovate-core/tests/fixtures")
sources = {
    "lotka_volterra": "lotka_forcing_scipy.json",
    "complementary_goods": "complementary_forcing_scipy.json",
    "lock_in": "lock_forcing_scipy.json",
    "multi_product": "multi_forcing_scipy.json",
}
events = {
    "lotka_volterra": [
        dict(time=2.125, parameters=dict(alpha1=0.4, beta_beta2_price=0.01)),
        dict(time=5.375, parameters=dict(beta1=0.08)),
    ],
    "complementary_goods": [
        dict(time=2.125, parameters=dict(c1=0.06, beta_k1_price=-0.01)),
        dict(time=5.375, parameters=dict(k2=0.2, c2=0.015)),
    ],
    "lock_in": [
        dict(time=2.125, parameters=dict(beta1=0.06, m=110.0)),
        dict(time=5.375, parameters=dict(m=90.0, beta_m_price=3.0)),
    ],
    "multi_product": [
        dict(time=2.125, parameters=dict(p1=0.025, alpha_1_2=0.2)),
        dict(time=5.375, parameters=dict(beta_m2_price=-2.0, q2=0.35)),
    ],
}
time = np.unique(np.concatenate([np.arange(0, 12.25, 0.25), [2.125, 5.375]]))
control = np.interp(time, np.arange(0, 12.25, 0.25), np.sin(np.arange(0, 12.25, 0.25) / 1.7))


def active_parameters(parameters, changes, t):
    active = parameters.copy()
    for event in changes:
        if event["time"] <= t:
            active.update(event["parameters"])
    return active


def effective_parameters(parameters, t):
    x = np.interp(t, time, control)
    return {
        name: value + parameters[f"beta_{name}_price"] * x
        for name, value in parameters.items()
        if not name.startswith("beta_")
    }


def rhs(model, parameters, t, y):
    p = effective_parameters(parameters, t)
    a, b = y
    if model == "lotka_volterra":
        return [p["alpha1"] * a * (1 - a) - p["beta1"] * a * b, p["alpha2"] * b * (1 - b) - p["beta2"] * a * b]
    if model == "complementary_goods":
        return [p["k1"] * a * (1 - a) + p["c1"] * a * b, p["k2"] * b * (1 - b) + p["c2"] * a * b]
    if model == "lock_in":
        market = p["m"]
        a, b = np.clip(y, 0, market)
        return [
            p["alpha1"] * a * (1 - (a + b) / market) + p["beta1"] * a * a / market - p["gamma1"] * a * b / market,
            p["alpha2"] * b * (1 - (a + b) / market) + p["beta2"] * b * b / market - p["gamma2"] * b * a / market,
        ]
    return [
        (p["p1"] + p["q1"] * a / p["m1"]) * (p["m1"] - a - p["alpha_1_2"] * b),
        (p["p2"] + p["q2"] * b / p["m2"]) * (p["m2"] - b - p["alpha_2_1"] * a),
    ]


cases = {}
for model, source in sources.items():
    original = json.loads((fixtures / source).read_text())
    parameters = original["parameters"]
    changes = events[model]
    raw = [original["initial"]]
    for left, right in zip(time[:-1], time[1:], strict=True):
        active = active_parameters(parameters, changes, left)
        solution = solve_ivp(
            lambda t, y, model=model, active=active: rhs(model, active, t, y),
            (left, right),
            raw[-1],
            method="DOP853",
            rtol=1e-12,
            atol=1e-13,
        )
        assert solution.success
        raw.append(solution.y[:, -1].tolist())
    rows = []
    rates = []
    for t, y in zip(time, raw, strict=True):
        active = active_parameters(parameters, changes, t)
        row = np.clip(y, 0, effective_parameters(active, t)["m"]).tolist() if model == "lock_in" else y
        rows.append(row)
        rates.append(rhs(model, active, t, row))
    active = active_parameters(parameters, changes, time[-1])
    future = solve_ivp(
        lambda t, y, model=model, active=active: rhs(model, active, t, y),
        (time[-1], 15.0),
        raw[-1],
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    )
    assert future.success
    forecast = (
        np.clip(future.y[:, -1], 0, effective_parameters(active, 15.0)["m"]).tolist()
        if model == "lock_in"
        else future.y[:, -1].tolist()
    )
    kwargs = dict(initial_values=original["initial"], covariates=["price"], parameter_events=changes)
    if model == "multi_product":
        kwargs["n_products"] = 2
    cases[model] = dict(
        parameters=parameters,
        constructor_kwargs=kwargs,
        rows=rows,
        rates=rates,
        forecast_rows=[forecast],
    )
output = dict(
    oracle=dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        method="DOP853, split and freeze parameters at every control/event boundary, rtol=1e-12 atol=1e-13",
    ),
    time=time.tolist(),
    covariates=dict(price=control.tolist()),
    forecast_time=[15.0],
    cases=cases,
)
(fixtures / "ecosystem_events_scipy.json").write_text(json.dumps(output, indent=2) + "\n")
