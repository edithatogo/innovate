---
title: Native Rust and Polars migration
description: Current development API, scientific conventions and remaining obligations
---

Innovate 0.7 is a breaking development line. Rust owns computation and scientific
validation. Python, Julia and R are the supported bindings; Polars owns tables.
The installed Python package requires Polars and its native extension. There is
no Python numerical fallback, pandas or xarray runtime.

## Current scope

Seven native model families expose deterministic fitting, prediction, simulation,
summary and diagnostics: Bass, Logistic, Gompertz, Fisher–Pry, Norton–Bass,
Lotka–Volterra and multi-product competition. Their support remains bounded.
Covariates, events, uncertainty, other model families and native Kairos simulation
remain required migration work. Discovery describes the executable subset;
release stays blocked by the full model and feature ledger.

## Polars fitting and prediction

```python
import polars as pl
from innovate import predict, fit

parameters = {"p1": 0.03, "q1": 0.2, "m1": 1000.0,
              "p2": 0.02, "q2": 0.3, "m2": 1500.0}
options = {"variant": "successive_generations", "launch_times": [0.0, 4.0]}
table = pl.DataFrame({"time": [i * 0.5 for i in range(41)]})
observed = predict("norton_bass", table, parameters, model_options=options)
result = fit("norton_bass", observed, model_options=options,
             observed_columns=["prediction_1", "prediction_2"])
forecast = result.predict(pl.DataFrame({"time": [21.0, 25.0, 30.0]}))
rates = result.predict(pl.DataFrame({"time": [21.0, 25.0, 30.0]}), quantity="rate")
```

Inputs require finite, strictly increasing time. Nulls, duplicate times, unknown
options and invalid parameter domains fail in Rust. Scalar results append
`prediction`; matrix results append `prediction_1` through `prediction_N`.
Rates use `rate` or `rate_N`. Existing output column names are errors, preventing
accidental overwrite. Observation column order defines model series order.
IPC currently copies; LazyFrames collect at the execution boundary.

## Norton–Bass conventions

`legacy_cannibalization` preserves the repository's historical ODE:
`dy_i/dt = (p_i + q_i*y_i/m_i) * (m_i - sum(y_j, j >= i))`.
Specify `initial_time` and `initial_values` to reproduce a historical initial
condition. Defaults are time zero and zero levels; the old `[1e-6, 0, ...]`
initial state is available explicitly. Forecast batches integrate from that
saved origin, rather than restarting at their first requested time. Historical
negative levels are retained with a warning; they are not silently clipped.

`successive_generations` uses ordered, fixed `launch_times`. Its two-generation
case follows the Norton–Bass substitution formulation: the second generation
captures a fraction of the first generation's diffusion and adds its own market.
This implementation recursively extends that transfer to later generations.
See [Norton and Bass (1987)](https://pubsonline.informs.org/doi/10.1287/mnsc.33.9.1069)
and equations 1–2 in [Franses and Hernández-Mireles (2006)](https://repub.eur.nl/pub/7867/ERS-2006-032-MKT.pdf).

The current native limit is 32 generations. Parameters are `p1`, `q1`, `m1`,
then `p2`, `q2`, `m2`, and so on. Innovation and imitation rates may be zero for
prediction; potentials must be positive. Launch times are supplied, not estimated.
At launch, rates use the right derivative. A generation level can decline:
its signed derivative is neither gross incidence nor cumulative unique adoption.
Threshold/half-life helpers and covariate extensions remain unported.

## Competitive models

Lotka–Volterra preserves the two-technology equations:
`dy1/dt = alpha1*y1*(1-y1) - beta1*y1*y2`, with the symmetric second equation.
Prediction needs explicit `initial_values` in `[0, 1]`. The two shares have
separate capacities and need not sum to one.

Multi-product competition preserves
`dy_i/dt = (p_i + q_i*y_i/m_i) * (m_i - y_i - sum(alpha_i_j*y_j, j != i))`.
Parameters are `p1..pN`, `q1..qN`, `m1..mN`, and each off-diagonal
`alpha_i_j`. Dimensions are inferred or supplied as `n_products` (1–32).
Prediction defaults to zero levels at time zero; historical nonzero initial
states can be supplied explicitly. A potential must be positive in this port;
zero-capacity product handling remains a required extension. Rates and
interactions are nonnegative; negative predicted levels produce a warning.

Competitive fitting uses the first observation as its initial condition unless
an explicit initial state is supplied. The fitted state retains the origin.
Conditional fit statistics exclude the initial observation used to set that
state, and do not propagate its measurement uncertainty. Covariates, events,
equilibrium helpers and complete original feature parity remain open.

## Fit quality and numerical limits

All seven fitters report actual numerical termination, iterations and scaled
residual loss. `fit_options` accepts `initial_parameters` and `max_iterations`
(1–2000 per local refinement start). Scalar models otherwise use their native
model-specific initializer; the limit does not constrain that initialization.
The deterministic multi-start damped least-squares optimizer uses positive log
parameters, so exact zero parameter estimates are not part of this fitter.
A failed convergence flag is returned with warnings, not hidden.

Local identifiability uses a column-scaled, pivoted QR of finite-difference
sensitivities, with relative rank tolerance `1e-7`. The reported QR diagonal
ratio is not an SVD condition number. Full local rank does not establish global
uniqueness, calibrated uncertainty or scientific validity. These are point
estimates; Bayesian inference and bootstrap intervals remain required work.

ODE models use adaptive step-doubling RK4 with Richardson correction, absolute
and relative tolerances `1e-10`, and an explicit step budget. Stiff/extreme cases
can return numerical-domain or budget errors. Independent SciPy DOP853 fixtures,
analytic limits, forecast checks and parameter recovery exercise the bounded
implementation; they do not certify all parameter regimes.

Schema 2.0 makes Gompertz explicit: `a * exp(-b * exp(-c*t))`. The previous
implementation ignored `b`. Uncomputed significance tests remain null with an
unavailable status; no p-values are fabricated.

## R and Julia

Both call the Rust library through the C ABI and need no Python interpreter.
Build with `cargo build --locked -p innovate-ffi`; set `INNOVATE_NATIVE_LIBRARY`
to the resulting shared library. Julia uses JSON 1.13.4 and R uses jsonlite 2.0.0.
They accept the same `constructor_kwargs`, `fit_options`, observation matrices
and fitted `state` as the native JSON contract. R retains one-element vectors
as arrays and preserves optional null fields. Native distribution bundles and
tabular IPC wrappers remain open qualification.

## Ecosystem alignment

Voiage owns economic/VOI calculations and consumes Polars/Arrow scenario tables.
Kairos owns scheduling, DES/ABM state and deterministic replay. Innovate retains
diffusion fitting and diagnostics. The pre-migration source/test inventory and
all 38 model/feature obligations remain retained in Conductor. A passing native
subset does not replace the full scientific, platform or release gates.
