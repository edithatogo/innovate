"""Independent complementary-goods forcing reference on Python 3.15."""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

time = np.arange(0, 12.25, 0.25)
control = np.sin(time / 1.7)
parameters = dict(
    k1=0.3,
    k2=0.25,
    c1=0.04,
    c2=0.025,
    beta_k1_price=0.04,
    beta_k2_price=-0.02,
    beta_c1_price=0.015,
    beta_c2_price=-0.008,
)


def rhs(t, y):
    x = np.interp(t, time, control)
    k1 = parameters["k1"] + parameters["beta_k1_price"] * x
    k2 = parameters["k2"] + parameters["beta_k2_price"] * x
    c1 = parameters["c1"] + parameters["beta_c1_price"] * x
    c2 = parameters["c2"] + parameters["beta_c2_price"] * x
    return [
        k1 * y[0] * (1 - y[0]) + c1 * y[0] * y[1],
        k2 * y[1] * (1 - y[1]) + c2 * y[0] * y[1],
    ]


rows = [[0.1, 0.2]]
for left, right in zip(time[:-1], time[1:], strict=True):
    solution = solve_ivp(rhs, (left, right), rows[-1], method="DOP853", rtol=1e-12, atol=1e-13)
    assert solution.success
    rows.append(solution.y[:, -1].tolist())
future = solve_ivp(rhs, (time[-1], 15), rows[-1], method="DOP853", rtol=1e-12, atol=1e-13)
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
    initial=[0.1, 0.2],
    parameters=parameters,
    rows=rows,
    rates=[rhs(t, y) for t, y in zip(time, rows, strict=True)],
    forecast_time=[15.0],
    forecast_rows=[future.y[:, -1].tolist()],
)
Path("rust/crates/innovate-core/tests/fixtures/complementary_forcing_scipy.json").write_text(
    json.dumps(fixture, indent=2) + "\n"
)
