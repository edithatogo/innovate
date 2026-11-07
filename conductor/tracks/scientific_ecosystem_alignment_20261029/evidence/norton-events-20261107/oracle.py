"""Independent historical Norton-Bass event reference on Python 3.15."""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

time = np.unique(np.concatenate([np.arange(0, 12.25, 0.25), [2.125, 5.375]]))
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
    dict(time=2.125, parameters=dict(p1=0.025, m2=130.0)),
    dict(time=5.375, parameters=dict(beta_q2_price=-0.005, q1=0.35)),
]


def active_at(t):
    active = parameters.copy()
    for event in events:
        if event["time"] <= t:
            active.update(event["parameters"])
    return active


def rhs(t, y, active):
    x = np.interp(t, time, control)
    result = []
    for index in [1, 2]:
        innovation, imitation, market = [
            active[f"{name}{index}"] + active[f"beta_{name}{index}_price"] * x for name in ["p", "q", "m"]
        ]
        result.append(
            (innovation + imitation * y[index - 1] / market) * (market - sum(y[index - 1 :])) if market > 0 else 0.0
        )
    return result


rows = [[0.0, 0.0]]
for left, right in zip(time[:-1], time[1:], strict=True):
    active = active_at(left)
    solution = solve_ivp(
        lambda t, y, active=active: rhs(t, y, active),
        (left, right),
        rows[-1],
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    )
    assert solution.success
    rows.append(solution.y[:, -1].tolist())
active = active_at(time[-1])
future = solve_ivp(
    lambda t, y: rhs(t, y, active),
    (time[-1], 15.0),
    rows[-1],
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
        method="DOP853, split and freeze parameters at every control/event boundary, rtol=1e-12 atol=1e-13",
    ),
    time=time.tolist(),
    covariates=dict(price=control.tolist()),
    parameters=parameters,
    constructor_kwargs=dict(n_generations=2, covariates=["price"], parameter_events=events),
    rows=rows,
    rates=[rhs(t, y, active_at(t)) for t, y in zip(time, rows, strict=True)],
    forecast_time=[15.0],
    forecast_rows=[future.y[:, -1].tolist()],
)
Path("rust/crates/innovate-core/tests/fixtures/norton_events_scipy.json").write_text(
    json.dumps(fixture, indent=2) + "\n"
)
