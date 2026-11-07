"""Generate the independent multi-product forcing reference with Python 3.15."""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

time = np.arange(0, 12.25, 0.25)
control = np.sin(time / 1.7)
parameters = dict(
    p1=0.03,
    p2=0.025,
    q1=0.25,
    q2=0.3,
    m1=100.0,
    m2=140.0,
    alpha_1_2=0.1,
    alpha_2_1=0.15,
    beta_p1_price=0.003,
    beta_p2_price=0.002,
    beta_q1_price=0.01,
    beta_q2_price=-0.015,
    beta_m1_price=5.0,
    beta_m2_price=-3.0,
    beta_alpha_1_2_price=0.02,
    beta_alpha_2_1_price=-0.01,
)
names = ["p1", "p2", "q1", "q2", "m1", "m2", "alpha_1_2", "alpha_2_1"]
base = np.array([parameters[name] for name in names])
coefficients = np.array([parameters[f"beta_{name}_price"] for name in names])
cases = {}
for layout in ["corrected", "historical"]:

    def rhs(t, y, historical=layout == "historical"):
        x = np.interp(t, time, control)
        beta = coefficients if not historical else coefficients[[0, 3, 1, 4, 2, 5, 6, 7]]
        p1, p2, q1, q2, m1, m2, a12, a21 = base + beta * x
        return [
            (p1 + q1 * y[0] / m1) * (m1 - y[0] - a12 * y[1]) if m1 > 0 else 0.0,
            (p2 + q2 * y[1] / m2) * (m2 - y[1] - a21 * y[0]) if m2 > 0 else 0.0,
        ]

    rows = [[0.0, 0.0]]
    for left, right in zip(time[:-1], time[1:], strict=True):
        solution = solve_ivp(rhs, (left, right), rows[-1], method="DOP853", rtol=1e-12, atol=1e-13)
        assert solution.success
        rows.append(solution.y[:, -1].tolist())
    forecast = solve_ivp(rhs, (time[-1], 15.0), rows[-1], method="DOP853", rtol=1e-12, atol=1e-13)
    assert forecast.success
    cases[layout] = dict(
        rows=rows,
        rates=[rhs(t, y) for t, y in zip(time, rows, strict=True)],
        forecast_rows=[forecast.y[:, -1].tolist()],
    )
fixture = dict(
    oracle=dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        method="DOP853, split at every control knot, rtol=1e-12 atol=1e-13",
    ),
    time=time.tolist(),
    covariates=dict(price=control.tolist()),
    initial=[0.0, 0.0],
    parameters=parameters,
    cases=cases,
    forecast_time=[15.0],
)
Path("rust/crates/innovate-core/tests/fixtures/multi_forcing_scipy.json").write_text(
    json.dumps(fixture, indent=2) + "\n"
)
