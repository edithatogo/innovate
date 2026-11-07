import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

time = np.arange(0, 12.25, 0.25)
control = np.sin(time / 1.7)
parameters = dict(
    alpha1=0.3,
    beta1=0.12,
    alpha2=0.22,
    beta2=0.15,
    beta_alpha1_price=0.04,
    beta_beta1_price=-0.03,
    beta_alpha2_price=-0.02,
    beta_beta2_price=0.025,
)


def rhs(t, y):
    x = np.interp(t, time, control)
    a = parameters["alpha1"] + parameters["beta_alpha1_price"] * x
    b = parameters["beta1"] + parameters["beta_beta1_price"] * x
    c = parameters["alpha2"] + parameters["beta_alpha2_price"] * x
    d = parameters["beta2"] + parameters["beta_beta2_price"] * x
    return [a * y[0] * (1 - y[0]) - b * y[0] * y[1], c * y[1] * (1 - y[1]) - d * y[0] * y[1]]


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
Path("rust/crates/innovate-core/tests/fixtures/lotka_forcing_scipy.json").write_text(
    json.dumps(fixture, indent=2) + "\n"
)
