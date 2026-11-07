"""Independent fraction-space successive-generation reference on Python 3.15."""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

launches = [0.0, 3.125]
time = np.unique(np.concatenate([np.arange(0, 12.25, 0.25), launches, [2.125, 5.375]]))
control = np.interp(time, np.arange(0, 12.25, 0.25), np.sin(np.arange(0, 12.25, 0.25) / 1.7))
parameters = dict(
    p1=0.03,
    q1=0.3,
    m1=100.0,
    p2=0.02,
    q2=0.4,
    m2=140.0,
    beta_p1_price=0.003,
    beta_q1_price=0.01,
    beta_m1_price=5.0,
    beta_p2_price=0.002,
    beta_q2_price=-0.015,
    beta_m2_price=-3.0,
)
events = [
    dict(time=2.125, parameters=dict(p1=0.025, beta_q1_price=0.02)),
    dict(time=5.375, parameters=dict(m1=110.0, m2=130.0, beta_q2_price=-0.005, q1=0.35)),
]


def active_at(t):
    active = parameters.copy()
    for event in events:
        if event["time"] <= t:
            active.update(event["parameters"])
    return active


def coefficients(active, t):
    x = np.interp(t, time, control)
    return [[active[f"{name}{i}"] + active[f"beta_{name}{i}_price"] * x for name in ["p", "q", "m"]] for i in [1, 2]]


def control_slope(t):
    index = np.searchsorted(time, t, side="right")
    if index == 0 or index == len(time):
        return 0.0
    return (control[index] - control[index - 1]) / (time[index] - time[index - 1])


def fraction_rhs(t, state, active, launched):
    return [
        (p + q * fraction) * (1 - fraction) if enabled else 0.0
        for (p, q, _), fraction, enabled in zip(coefficients(active, t), state, launched, strict=True)
    ]


fractions = [[0.0, 0.0]]
for left, right in zip(time[:-1], time[1:], strict=True):
    active = active_at(left)
    launched = [left >= launch for launch in launches]
    solution = solve_ivp(
        lambda t, state, active=active, launched=launched: fraction_rhs(t, state, active, launched),
        (left, right),
        fractions[-1],
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    )
    assert solution.success
    fractions.append(solution.y[:, -1].tolist())


def outputs(t, state):
    active = active_at(t)
    rates = fraction_rhs(t, state, active, [t >= launch for launch in launches])
    markets = [row[2] for row in coefficients(active, t)]
    slopes = [active[f"beta_m{i}_price"] * control_slope(t) for i in [1, 2]]
    first = state[0] * markets[0]
    first_rate = rates[0] * markets[0] + state[0] * slopes[0]
    return (
        [first * (1 - state[1]), state[1] * (markets[1] + first)],
        [
            first_rate * (1 - state[1]) - first * rates[1],
            rates[1] * (markets[1] + first) + state[1] * (slopes[1] + first_rate),
        ],
    )


rows, rates = zip(*(outputs(t, state) for t, state in zip(time, fractions, strict=True)), strict=True)
active = active_at(time[-1])
future = solve_ivp(
    lambda t, state: fraction_rhs(t, state, active, [True, True]),
    (time[-1], 15.0),
    fractions[-1],
    method="DOP853",
    rtol=1e-12,
    atol=1e-13,
)
assert future.success
fixture = dict(
    oracle=dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        method="Direct Bass fractions with DOP853, frozen launch/event regimes, control knots, rtol=1e-12 atol=1e-13",
    ),
    time=time.tolist(),
    covariates=dict(price=control.tolist()),
    parameters=parameters,
    constructor_kwargs=dict(
        n_generations=2,
        variant="successive_generations",
        launch_times=launches,
        covariates=["price"],
        parameter_events=events,
    ),
    fractions=fractions,
    rows=rows,
    rates=rates,
    forecast_time=[15.0],
    forecast_rows=[outputs(15.0, future.y[:, -1])[0]],
)
Path("rust/crates/innovate-core/tests/fixtures/successive_forcing_scipy.json").write_text(
    json.dumps(fixture, indent=2) + "\n"
)
