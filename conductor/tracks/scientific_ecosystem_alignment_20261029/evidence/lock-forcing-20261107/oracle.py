"""Independent lock-in forcing reference on Python 3.15."""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

time = np.arange(0, 12.25, 0.25)
control = np.sin(time / 1.7)
names = ["alpha1", "alpha2", "beta1", "beta2", "gamma1", "gamma2", "m"]
parameters = dict(
    alpha1=0.3,
    alpha2=0.25,
    beta1=0.04,
    beta2=0.03,
    gamma1=0.05,
    gamma2=0.04,
    m=100.0,
    beta_alpha1_price=0.04,
    beta_alpha2_price=-0.02,
    beta_beta1_price=0.01,
    beta_beta2_price=-0.005,
    beta_gamma1_price=0.012,
    beta_gamma2_price=-0.008,
    beta_m_price=5.0,
)


def rhs(t, y):
    x = np.interp(t, time, control)
    alpha1, alpha2, beta1, beta2, gamma1, gamma2, market = [
        parameters[name] + parameters[f"beta_{name}_price"] * x for name in names
    ]
    a, b = np.clip(y, 0, market)
    return [
        alpha1 * a * (1 - (a + b) / market) + beta1 * a * a / market - gamma1 * a * b / market,
        alpha2 * b * (1 - (a + b) / market) + beta2 * b * b / market - gamma2 * b * a / market,
    ]


raw = [[10.0, 7.0]]
for left, right in zip(time[:-1], time[1:], strict=True):
    solution = solve_ivp(rhs, (left, right), raw[-1], method="DOP853", rtol=1e-12, atol=1e-13)
    assert solution.success
    raw.append(solution.y[:, -1].tolist())
rows = [
    np.clip(y, 0, parameters["m"] + parameters["beta_m_price"] * x).tolist() for y, x in zip(raw, control, strict=True)
]
future = solve_ivp(rhs, (time[-1], 15.0), raw[-1], method="DOP853", rtol=1e-12, atol=1e-13)
assert future.success
fixture = dict(
    oracle=dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        method="DOP853, split at every control knot, rtol=1e-12 atol=1e-13",
    ),
    time=time.tolist(),
    covariates=dict(price=control.tolist()),
    initial=raw[0],
    parameters=parameters,
    rows=rows,
    rates=[rhs(t, y) for t, y in zip(time, rows, strict=True)],
    forecast_time=[15.0],
    forecast_rows=[np.clip(future.y[:, -1], 0, parameters["m"] + parameters["beta_m_price"] * control[-1]).tolist()],
)
Path("rust/crates/innovate-core/tests/fixtures/lock_forcing_scipy.json").write_text(
    json.dumps(fixture, indent=2) + "\n"
)
