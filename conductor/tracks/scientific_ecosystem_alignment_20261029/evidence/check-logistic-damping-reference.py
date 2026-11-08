"""Compare the sparse Logistic damping-limit fit with an analytic Jacobian."""

from math import exp
from random import Random

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from innovate import execute

rng = Random(272)
time = np.array([-3, -2, -1, 0, 100, 101, 102, 103], dtype=float)
for _ in range(40):
    observed = np.array([100 / (1 + exp(-t)) + rng.gauss(0, 1) for t in time])
scale = max(observed)


def residual(parameters):
    capacity, rate, midpoint = parameters
    return (capacity * expit(rate * (time - midpoint)) - observed) / scale


def jacobian(parameters):
    capacity, rate, midpoint = parameters
    fraction = expit(rate * (time - midpoint))
    slope = capacity * fraction * (1 - fraction)
    return np.column_stack((fraction, slope * (time - midpoint), -rate * slope)) / scale


fit = execute(
    "fit_model", "logistic", inputs={"time": time.tolist(), "observed": observed.tolist()},
    fit_options={"initial_parameters": {"L": 100, "k": 1, "x0": 0}},
)["result"]
native = np.array([fit["parameters"][key] for key in ("L", "k", "x0")])
reference = least_squares(residual, [100, 1, 0], jac=jacobian, gtol=1e-14, ftol=1e-14, xtol=1e-14)
gap = abs(np.dot(residual(native), residual(native)) - 2 * reference.cost)
difference = max(abs(native - reference.x))
print(f"Native termination {fit['diagnostics']['optimizer']['termination']}, "
      f"scaled RSS gap {gap:.3g}, maximum coefficient difference {difference:.3g}")
assert reference.success and gap < 1e-12 and difference < 1e-5
assert fit["diagnostics"]["optimizer"]["termination"] == "damping_limit"
