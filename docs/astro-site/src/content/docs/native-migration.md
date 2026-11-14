---
title: Native Rust and Polars migration
description: Current development API, scientific conventions and remaining obligations
---

Innovate 0.7 is a breaking development line. Rust owns computation and scientific
validation. Python, Julia and R are the supported bindings; Polars owns tables.
The installed Python package requires Polars and its native extension. There is
no Python numerical fallback, pandas or xarray runtime.

## Current scope

All sixteen model families expose bounded native fitting, prediction, simulation,
summary and diagnostics paths: Bass, Logistic, Gompertz, Fisher–Pry, Norton–Bass,
Lotka–Volterra, multi-product competition, complementary goods, lock-in,
network diffusion, policy diffusion, latent process, regime switching,
hierarchical, mixture and composite. Operation metadata distinguishes point
simulation, stochastic Kairos engines, supported rate outputs and named variants.
Pointwise Logistic and mechanistic Bass/Gompertz covariates/events, plus other
specific extensions, are described below;
full covariates/events across models, inference, uncertainty, richer Kairos and
all remaining scientific/distribution obligations stay required. Discovery is
the executable subset, and all 38 ledger obligations still block release.

## Polars fitting and prediction

Run `.venv-native/bin/python -I examples/native_norton_bass.py` from the repository root.
The contract checks this displayed source against that executable file.

```python
"""Check two-generation fitting and forecasts against analytic diffusion curves."""

import json
from math import exp, isclose, isfinite

import polars as pl

from innovate import fit, predict


def fraction_and_rate(time: float, p: float, q: float) -> tuple[float, float]:
    """Return the Bass fraction and its right derivative after a fixed launch."""
    if time < 0:
        return 0.0, 0.0
    exponential = exp(-(p + q) * time)
    denominator = 1 + q / p * exponential
    return (1 - exponential) / denominator, (p + q) ** 2 / p * exponential / denominator**2


def analytic(time: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Evaluate the two-generation transfer equations independently in Python."""
    f1, d1 = fraction_and_rate(time, 0.03, 0.2)
    f2, d2 = fraction_and_rate(time - 4, 0.02, 0.3)
    levels = (1000 * f1 * (1 - f2), (1500 + 1000 * f1) * f2)
    rates = (1000 * (d1 * (1 - f2) - f1 * d2), 1000 * d1 * f2 + (1500 + 1000 * f1) * d2)
    return levels, rates


def main() -> None:
    """Validate synthetic recovery and held-out levels and signed rates."""
    parameters = {"p1": 0.03, "q1": 0.2, "m1": 1000.0, "p2": 0.02, "q2": 0.3, "m2": 1500.0}
    options = {"variant": "successive_generations", "launch_times": [0.0, 4.0]}
    table = pl.DataFrame({"time": [i * 0.5 for i in range(41)]})
    observed = predict("norton_bass", table, parameters, model_options=options)
    for time, first, second in observed.iter_rows():
        expected, _ = analytic(time)
        assert isclose(first, expected[0], abs_tol=1e-9, rel_tol=1e-12)
        assert isclose(second, expected[1], abs_tol=1e-9, rel_tol=1e-12)

    result = fit("norton_bass", observed, model_options=options, observed_columns=["prediction_1", "prediction_2"])
    assert result.diagnostics["optimizer"]["converged"] is True
    assert isinstance(result.envelope["result"]["state"], dict)
    for name, expected in parameters.items():
        assert isclose(result.parameters[name], expected, rel_tol=0.001)

    future = pl.DataFrame({"time": [21.0, 25.0, 30.0]})
    forecast = result.predict(future)
    rates = result.predict(future, quantity="rate")
    assert forecast.columns == ["time", "prediction_1", "prediction_2"]
    assert rates.columns == ["time", "rate_1", "rate_2"]
    assert forecast["time"].to_list() == future["time"].to_list()
    assert rates["time"].to_list() == future["time"].to_list()
    for level_row, rate_row in zip(forecast.iter_rows(), rates.iter_rows(), strict=True):
        expected_levels, expected_rates = analytic(level_row[0])
        for actual, expected in zip(level_row[1:] + rate_row[1:], expected_levels + expected_rates, strict=True):
            assert isfinite(actual) and isclose(actual, expected, abs_tol=0.01, rel_tol=0)
    # Signed generation rates may be negative during substitution.
    assert rates["rate_1"][0] < 0
    print(
        json.dumps(
            {"converged": True, "forecast_rows": forecast.height, "parameters_checked": len(parameters)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
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

## Calendar aggregation

Use `aggregate_dates` with aligned `dates` and `values` arrays and a
`frequency` of `D`, `W`, `M`, `ME`, `MS`, `QS`, `QE`, `YS`, or `YE`. Daily, monthly, quarterly and annual
aggregation fills missing periods between the first and last observed
buckets with zero. Use `MS` for month-start labels or `M` and `ME`
for month-end labels. Use `QS` and `QE` for calendar-quarter start
and end labels, and `YS` and `YE` for calendar-year start and end
labels. Weekly aggregation uses seven-day windows
anchored to the earliest observation and returns occupied buckets.

For quarter and year frequencies, set `fiscal_start_month` to an
integer from 1 through 12 to shift the calendar. It defaults to
January. For example, `YE` with `fiscal_start_month: 7` labels
each fiscal year on June 30.

Use ISO calendar dates, naive timestamps, or offset timestamps
consistently within a request. Offset timestamps are converted to UTC
before bucketing. For example, `2024-03-01T00:00:00+01:00` falls in
February when aggregated at month start.

## Adopter categories

Use `categorize_adopters` with strictly increasing finite timestamps
and aligned nonnegative adoption rates with positive total weight.
Each rate weights its sample. The spacing between samples does not
multiply the weight, so changing the sampling grid can change the
categories. Zero-weight samples receive labels without contributing
to the weighted moments.

The five historical categories use inclusive upper boundaries at the
weighted mean minus two standard deviations, the mean minus one
standard deviation, the mean, and the mean plus one standard deviation.
Samples beyond the last boundary are Laggards.

In Python, `innovate.categorize_adopters(frame)` adds a category column
to a Polars frame and preserves its existing rows and columns. Julia
and R can call `kernel_call` with operation `categorize_adopters` and
payload fields `time` and `adoption_rate`.

## Fit quality and numerical limits

All seven fitters report actual numerical termination, iterations and scaled
residual loss. `fit_options` accepts `initial_parameters` and `max_iterations`
(1–2000 per local refinement start). Scalar models otherwise use their native
model-specific initializer; the limit does not constrain that initialization.
The deterministic multi-start damped least-squares optimizer uses positive log
parameters, so exact zero parameter estimates are not part of this fitter.
A failed convergence flag is returned with warnings, not hidden.

The `optimizer.start_reports` array follows attempted-start order.
`selected_start_index` is zero-based. Each attempt records its initial
transformed coordinates and termination reason. Finite starts also
report their final transformed coordinates and initial and final scaled
residual sums of squares. A start rejected by the initial predictor
records the prediction error.

`converged_starts` counts starts that met a numerical stopping criterion.
`alternative_start_scaled_rss_gap` is the best other finite start's final
loss minus the selected start's final loss. It is null when no other
finite result exists. Compare the final coordinates when the gap is
small. Two starts can have the same objective at different coordinates,
as in a squared-parameter model evaluated at −1 and 1. The coordinates
are in the optimizer's parameterization.

Local identifiability uses a column-scaled, pivoted QR of finite-difference
sensitivities, with relative rank tolerance `1e-7`. The reported QR diagonal
ratio is not an SVD condition number. Full local rank does not establish global
uniqueness, calibrated uncertainty or scientific validity. These are point
estimates. Separate operations provide bounded Bayesian inference and bootstrap intervals.

The diagnostic reports `sensitivity_column_norm_cutoff` and the
zero-based `excluded_sensitivity_column_indices` in sensitivity
column order. Columns at or below the absolute norm cutoff are
excluded before the relative QR rank test. A raw column norm above
the representable range is reported as null. The normalized column
can still contribute to rank.

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

## Native Kairos Bass simulation

`simulate_model` accepts an explicit `simulation_options` object for Bass:

Run `.venv-native/bin/python -I examples/native_kairos.py` from the repository root.
The contract checks this displayed source against that executable file.

```python
"""Check a seeded Bass DES path, replay and its count-to-level interpretation.

The population has 100 agents, while market potential is 250 arbitrary units.
One seeded path is a synthetic realization, not an expected adoption curve.
Finite-population expectations with imitation need not equal deterministic Bass.
"""

import json
import math
from itertools import pairwise

import polars as pl

from innovate import NativeError, execute


def main() -> None:
    """Validate replay, sampling invariance, physical bounds and engine errors."""
    times = [0.0, 1.0, 5.0, 20.0]
    parameters = {"p": 0.2, "q": 0.3, "m": 250.0}
    options = {"engine": "kairos_des", "seed": 42, "population": 100, "ticks_per_unit": 1_000_000}
    request = {"inputs": {"time": times}, "parameters": parameters, "simulation_options": options}
    response = execute("simulate_model", "bass", **request)
    replay_equal = response == execute("simulate_model", "bass", **request)
    assert replay_equal
    assert response["metadata"]["engine"] == "kairos_des"

    result = response["result"]
    counts = result["counts"]
    assert len(counts) == len(times)
    assert all(type(count) is int and 0 <= count <= options["population"] for count in counts)
    assert counts[0] == 0 and counts[-1] > 0
    assert all(left <= right for left, right in pairwise(counts))

    # Independent unit conversion: each agent represents m / population units.
    expected_levels = [parameters["m"] * count / options["population"] for count in counts]
    assert all(
        math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        for actual, expected in zip(result["values"], expected_levels, strict=True)
    )
    trajectory = pl.DataFrame({"time": times, "adopters": counts, "level": result["values"]})
    assert trajectory["time"].to_list() == times

    # Merely asking for fewer observations must not restart the event process.
    sparse = execute("simulate_model", "bass", **{**request, "inputs": {"time": times[2:]}})["result"]
    sparse_grid_equal = sparse["counts"] == counts[2:] and sparse["values"] == result["values"][2:]
    assert sparse_grid_equal

    # With no innovators and no initial adopters, imitation has no source.
    no_innovators = execute("simulate_model", "bass", **{**request, "parameters": {**parameters, "p": 0.0}})["result"]
    assert no_innovators["counts"] == [0] * len(times)

    try:
        execute("simulate_model", "bass", **{**request, "simulation_options": {**options, "engine": "unknown"}})
    except NativeError as error:
        unknown_engine_rejected = error.code == "invalid_request"
    else:
        unknown_engine_rejected = False
    assert unknown_engine_rejected, "An unsupported engine must fail instead of selecting a deterministic curve."

    print(
        json.dumps(
            {
                "counts": trajectory["adopters"].to_list(),
                "levels": trajectory["level"].to_list(),
                "replay_equal": replay_equal,
                "sparse_grid_equal": sparse_grid_equal,
                "unknown_engine_rejected": unknown_engine_rejected,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
```

Kairos revision `e76b97be2ea3712645933d903822ff6381c8883e` owns scheduler
dispatch and the seeded random stream. Innovate supplies the scientific hazard
`(p + q*n/N)*(N-n)` for a finite population starting with zero adopters.
Returned levels are `m*n/N`; counts are integers. This stochastic process has a
deterministic Bass limit as population grows; its finite-population expectation
is not assumed to equal the deterministic curve when imitation is nonzero.

Population is 1–100,000. Seed is an unsigned integer; R numeric seeds should stay
within its exact integer range. `ticks_per_unit` defaults to 1,000,000 and accepts
1–1,000,000,000. Waiting times round upward to at least one tick, creating a
resolution-dependent delay. Time must be finite, nonnegative and within the
reported tick domain. No simulation option is silently ignored. Existing calls
without `simulation_options` retain deterministic curve semantics.

The same payload works through R's `kernel_simulate_model` and Julia's
`kernel_simulate_model`. Metadata records the immutable dependency revision,
seed, resolution and RNG checkpoint. The checkpoint is provenance; resume,
ABM agent state, networks, resources, interventions and general scenario replay
remain required migration work. This port supplies aggregate adoption events.

## Kairos ABM, covariates and timed events

Use `simulation_options.engine="kairos_abm"` for Kairos-owned entity identities,
component state and event dispatch. Agents are exchangeable in this homogeneous
Bass port; a separate Kairos identity stream selects susceptible agents without
changing the DES timing stream. `result.agents` contains each entity's index,
generation and adoption state at the last requested time. Networks, heterogeneous
agent hazards, resource queues and resumable checkpoints remain required work.

Both native simulation engines accept this bounded forcing contract:

Run `.venv-native/bin/python -I examples/native_forcing.py` from the repository root.
The contract checks this displayed source against that executable file.

```python
"""Check an exact adoption pause, equivalent forcing and tracked ABM agents.

The synthetic covariate cancels both Bass rates until time three. A later event
raises innovation at time six. These piecewise-constant simulation controls are
distinct from fitted covariate effects or empirical intervention estimates.
"""

import json
from itertools import pairwise

from innovate import NativeError, execute


def main() -> None:
    """Exercise covariates, additive events, agent conservation and input guards."""
    times = [0.0, 1.0, 3.0, 5.0, 6.0, 10.0]
    parameters = {"p": 0.2, "q": 0.3, "m": 100.0}
    options = {
        "engine": "kairos_abm",
        "seed": 42,
        "population": 100,
        "ticks_per_unit": 1_000_000,
        "covariate": {"time": [0.0, 3.0], "values": [-1.0, 0.0], "beta_p": 0.2, "beta_q": 0.3},
        "events": [{"time": 6.0, "delta_p": 0.1, "delta_q": 0.0}],
    }
    request = {"inputs": {"time": times}, "parameters": parameters, "simulation_options": options}
    response = execute("simulate_model", "bass", **request)
    result = response["result"]
    counts = result["counts"]
    assert len(counts) == len(times)

    # Before time three, p + beta_p*x and q + beta_q*x are both exactly zero.
    # At the boundary, adoption still needs at least one positive scheduler tick.
    assert counts[:3] == [0, 0, 0]
    assert counts[3] > 0
    assert all(0 <= count <= options["population"] for count in counts)
    assert all(left <= right for left, right in pairwise(counts))
    agents = result["agents"]
    assert len(agents) == options["population"]
    assert len({(agent["index"], agent["generation"]) for agent in agents}) == options["population"]
    assert sum(agent["adopted"] for agent in agents) == counts[-1]
    assert response == execute("simulate_model", "bass", **request)

    # The same rate schedule can be expressed entirely through additive events.
    # Identical boundaries and rates should consume the same seeded streams.
    event_options = {key: value for key, value in options.items() if key != "covariate"}
    event_options["events"] = [
        {"time": 0.0, "delta_p": -0.2, "delta_q": -0.3},
        {"time": 3.0, "delta_p": 0.2, "delta_q": 0.3},
        {"time": 6.0, "delta_p": 0.1, "delta_q": 0.0},
    ]
    equivalent = execute("simulate_model", "bass", **{**request, "simulation_options": event_options})["result"]
    equivalent_forcing = all(equivalent[key] == result[key] for key in ("counts", "values", "agents"))
    assert equivalent_forcing

    # A negative rate and two distinct changes rounded to the same tick must fail.
    invalid_options = [
        {**options, "covariate": {**options["covariate"], "beta_p": 2.0}},
        {
            **options,
            "ticks_per_unit": 1,
            "events": [
                {"time": 0.1, "delta_p": 0.0, "delta_q": 0.0},
                {"time": 0.2, "delta_p": 0.0, "delta_q": 0.0},
            ],
        },
    ]
    invalid_forcing_rejected = 0
    for invalid in invalid_options:
        try:
            execute("simulate_model", "bass", **{**request, "simulation_options": invalid})
        except NativeError as error:
            assert error.code == "invalid_request"
            invalid_forcing_rejected += 1
        else:
            raise AssertionError("Invalid rates or colliding changes were silently accepted.")
    assert invalid_forcing_rejected == len(invalid_options)

    print(
        json.dumps(
            {
                "counts": counts,
                "equivalent_forcing": equivalent_forcing,
                "invalid_forcing_rejected": invalid_forcing_rejected,
                "tracked_agents": len(agents),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
```

The covariate is right-continuous and piecewise constant, starting at time zero;
its last value persists. Rates are `p + beta_p*x + sum(delta_p)` and
`q + beta_q*x + sum(delta_q)` over events up to the current time. Effective rates
must remain finite and nonnegative. Zero pressure pauses adoption until a later
change. Covariate boundaries and event times round upward to ticks and precede
tied adoption events. Distinct change times that collide at the chosen tick
resolution are rejected. Waiting times are resampled at rate boundaries using
the exponential distribution's memorylessness. Adding boundaries can therefore
change seeded trajectories even if the resulting distributions agree.

This simulation contract does not replace the historical model-specific linear
covariate interpolation, fitting, market-potential effects or event-split curves.
Those ports remain required. Python, Julia and R accept the same JSON payload;
Polars can construct trajectory tables from returned native counts and levels.

## Historical Logistic covariates and event curves

Logistic now supports `model_options={"covariates": ["price"]}`. The Polars
`price` column supplies values aligned with `time`. Native JSON uses
`inputs.covariates.price`. Parameters add `beta_L_price`, `beta_k_price` and
`beta_x0_price` to the corresponding base parameters at each observation.
Multiple declared covariates are additive. Effective L and k must stay positive;
all inputs and coefficients must be finite. Forecasts need future covariate values.

For a fresh Logistic, Bass or Gompertz fit, set `event_candidates` to 2–16 strictly
increasing times in the constructor options. Each split must have at least
three observations on each side. The fit refits the model for every
candidate, selects the converged fit with the smallest residual sum of
squares, and keeps the earlier candidate on a tie. Weighted fits use the
normalized weighted residual sum of squares and skip candidates with
fewer than three positive-weight observations on either side. The fitted state stores the
selected `t_event`, and `event_profile` reports the candidate scores and
convergence. Each profile includes the winning row as `selected` and
its zero-based `selected_index`. The search is limited to the supplied times.
For scalar event-profile fits, `bootstrap_fit` repeats the candidate
search on each resample. Its `event_selection` reports the baseline
selection and each candidate's count and frequency among accepted
refits. These frequencies describe selection stability on the supplied
grid. They are separate from the percentile intervals on curve parameters.

A baseline fit with detected rank deficiency cannot produce parameter
bootstrap intervals. If a converged refit reports rank deficiency,
the bootstrap excludes it and records it under
`failure_counts.rank_deficient`. The percentile intervals describe the
accepted refits. Check the failure counts when interpreting those
intervals, especially when resampling can remove transition observations.

Centered residual resampling keeps the observed time grid and requires
a converged baseline because it generates observations from that fitted
curve. Paired case resampling can return intervals when the baseline
optimizer stops early. Inspect `baseline_optimizer.converged` alongside
the refit counts.

The requested interval mass sets the percentile cutoffs. Empirical
coverage also depends on the data and resampling design. In a seeded
study with four Logistic transition observations and four plateau
observations per dataset, nominal 95% capacity intervals covered the
generating value in 33 of 40 paired-case datasets and 27 of 39 datasets
with a converged residual-bootstrap baseline, using 200 draws per
dataset. A large draw count or a low refit failure rate can coexist with
undercoverage.

For Norton–Bass and the competitive ecosystem models, paired case
resampling applies the same sampled row weights to all observed series.
It reports candidate counts for event, launch, or joint profiles under
`event_selection`, `launch_selection`, or `joint_time_selection`.
Centered residual resampling centers each series and samples the entire
residual row together. If the first observed row sets the initial
condition, it is retained and excluded from the residual pool.

Pass `forecast_inputs` to predict on up to 100 times from each refitted
state. The forecast includes the baseline rows and pointwise percentile
intervals in time-by-series order. These intervals describe variation
among the refitted mean paths. Forecast failures are counted separately
among successful fits, and fewer than two forecast draws yield null bounds.

In Python, pass the bootstrap operation's `result` to
`bootstrap_forecast_frame` to get one Polars row per time and series.
Pass that frame to `export_ipc_file` for a file readable by Julia Arrow
and R Arrow. Missing bounds are stored as nullable Float64 columns.



Norton–Bass, Lotka–Volterra, complementary goods, lock-in and multi-product
fits accept the same candidate list with a zero-based `event_index`
into `parameter_events`. The fitter varies that event's time while preserving
its overrides and the other events. The selected event
list is stored in the fitted state.

To profile an existing override, add `event_override_parameter` and
`event_override_candidates` alongside `event_index`. Supply a time list
as well to fit the event time and override together. The fit evaluates
at most 16 total combinations and records the winning override in
the fitted event list. The optimizer diagnostics describe each
conditional base fit. The profile records the override search.

For successive-generation Norton–Bass, use `launch_candidates` with a
zero-based `launch_index` greater than zero to profile one later launch.
The selected launch is stored in `launch_times`, and the candidate
scores appear in `launch_profile`. To fit a launch and an event together,
provide both candidate lists and indices. The fitter evaluates up to 16
launch/event pairs and returns their scores in `joint_time_profile`.
Ties favor the earlier launch, then the earlier event. Python Polars fits
expose these profiles as `FitResult.launch_profile` and
`FitResult.joint_time_profile`.

`t_event` selects independent pre/post curves using `L_post`, `k_post`,
`x0_post` at and after the event. These historical curves can jump at the event;
continuity is not imposed. If covariates and an event are combined,
`event_covariates="historical_ignored"` preserves the original behavior: the
event branch ignores covariates. Metadata records that mode; fit diagnostics warn
that its coefficients are unidentifiable. Select `event_covariates="additive"`
to apply the same covariate coefficients to both segments in the redesigned API.

Rate predictions hold controls fixed by default. To include their local
time derivatives, supply `inputs.covariate_rates` in native JSON or use
`quantity="rate"` and `covariate_rate_columns={"price": "price_slope"}` in
Python's Polars prediction API. Supply one aligned slope column per declared
control. At an event boundary the post-event curve determines the reported
rate, with jump impulses excluded.

Advanced fits accept complete `fit_options.initial_parameters` or generate
starts from the observed levels and time span of each event segment. Automatic
starts set covariate coefficients to zero. Rust reports convergence and local
rank. Rank deficiency limits parameter identification. Analytic time-varying
adoption-rate helpers and the other models' historical interpolation and event
semantics remain migration work. This
port preserves the Logistic pointwise curve contract; it is not an ODE forcing
or interpolation rule. Historical source: pre-migration commit `56d91dd7`.

Both Kairos simulation engines also accept `initial_adopters` from zero through
population. ABM initial adopters are selected through its separate identity
stream and retained as adopted components. Counts include these agents at time
zero; `result.agents` still describes the final requested time. This does not
supply network or heterogeneous agent hazards.

## Complementary goods

`complementary_goods` ports the historical two-good equations:
`dy1/dt = k1*y1*(1-y1) + c1*y1*y2`, with the symmetric second equation using
k2 and c2. All four coefficients are nonnegative. Prediction requires explicit
`initial_values`; fitting conditions on the first observation and retains that
origin in fitted state. Covariates/events are still unavailable for this family.
Levels may exceed one because positive interactions enlarge equilibrium levels;
outputs are not clipped or normalized. Extreme interactions can cause numerical
domain/step-budget errors. Independent SciPy trajectories, analytic limits and
native parameter recovery validate the current bounded implementation.

## Network imitation with resource service

Use `simulation_options.engine="kairos_network"` with Bass parameters p/q/m:

Run `.venv-native/bin/python -I examples/native_network.py` from the repository root.
The contract checks this displayed source against that executable file.

```python
"""Check contagion, service completion and FIFO resource accounting.

One synthetic time unit equals one scheduler tick so the five-tick service time
is visible. The separate queue fixture uses a very high innovation rate to expose
the scheduler's minimum one-tick delay; it is not a calibrated adoption model.
"""

import json

from innovate import NativeError, execute


def main() -> None:
    """Verify graph-dependent spread, independent service timing and graph errors."""
    times = [0.0, 5.0, 20.0, 100.0]
    parameters = {"p": 0.0, "q": 3.0, "m": 3.0}
    options = {
        "engine": "kairos_network",
        "seed": 42,
        "population": 3,
        "edges": [[0, 1], [1, 2]],
        "initial_agents": [0],
        "capacity": 1,
        "service_ticks": 5,
        "ticks_per_unit": 1,
    }
    request = {"inputs": {"time": times}, "parameters": parameters, "simulation_options": options}
    response = execute("simulate_model", "bass", **request)
    chain = response["result"]
    assert response == execute("simulate_model", "bass", **request)
    assert chain["counts"][0] == 1 and chain["counts"][-1] == options["population"]
    assert all(agent["state"] == "adopted" for agent in chain["agents"])

    # With p=0, node two can request only after its sole neighbor one adopts.
    completions = {event["agent"]: event["tick"] for event in chain["completions"]}
    requests = {event["agent"]: event["tick"] for event in chain["requests"]}
    assert set(completions) == set(requests) == {1, 2}
    assert requests[2] > completions[1]
    assert chain["counts"][-1] == len(options["initial_agents"]) + len(completions)
    disconnected = execute("simulate_model", "bass", **{**request, "simulation_options": {**options, "edges": []}})[
        "result"
    ]
    assert disconnected["counts"] == [1] * len(times)
    assert disconnected["requests"] == disconnected["completions"] == []

    # A second fixture creates demand without peers and queues three agents.
    queue_options = {**options, "edges": [], "initial_agents": []}
    queued = execute(
        "simulate_model",
        "bass",
        inputs={"time": [0.0, 5.0, 20.0]},
        parameters={"p": 1e12, "q": 0.0, "m": 3.0},
        simulation_options=queue_options,
    )["result"]
    assert queued["counts"] == [0, 0, 3], "A pending request is not yet adoption."
    assert len(queued["requests"]) == len(queued["completions"]) == 3
    free_at = 0
    expected_occupied_ticks = 0
    expected_wait_ticks = 0
    for demand, completion in zip(queued["requests"], queued["completions"], strict=True):
        # Independent single-server FIFO recurrence; no native queue helper used.
        start = max(demand["tick"], free_at)
        free_at = start + queue_options["service_ticks"]
        assert completion["agent"] == demand["agent"]
        assert completion["tick"] == free_at
        expected_occupied_ticks += queue_options["service_ticks"]
        expected_wait_ticks += start - demand["tick"]
    metrics = queued["resource_metrics"]
    assert metrics["peak_in_service"] == queue_options["capacity"]
    assert metrics["occupied_slot_ticks"] == expected_occupied_ticks
    assert metrics["admitted_wait_ticks"] == expected_wait_ticks
    assert metrics["configured_slot_ticks"] == metrics["horizon_ticks"] * queue_options["capacity"]
    assert metrics["idle_slot_ticks"] + metrics["occupied_slot_ticks"] == metrics["configured_slot_ticks"]
    assert metrics["over_capacity_slot_ticks"] == 0

    try:
        execute("simulate_model", "bass", **{**request, "simulation_options": {**options, "edges": [[0, 0]]}})
    except NativeError as error:
        invalid_edges_rejected = error.code == "invalid_request"
    else:
        invalid_edges_rejected = False
    assert invalid_edges_rejected, "An undirected self-loop must fail the native graph contract."

    print(
        json.dumps(
            {
                "chain_counts": chain["counts"],
                "disconnected_counts": disconnected["counts"],
                "fifo_completions": [event["tick"] for event in queued["completions"]],
                "invalid_edges_rejected": invalid_edges_rejected,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
```

Edges are unique undirected pairs of node positions, without self-loops. Agent
records retain node order and Kairos identities. Population is 1–1000; capacity
is 1–population. Initial agent indices are unique and already adopted. A
susceptible node requests service at hazard `p + q*adopted_neighbors/degree`;
isolated nodes have zero imitation. Native simulation now permits p=0, allowing
pure contagion seeded by initial agents. Deterministic Bass fitting/prediction
retains its previous positive-p domain.

Kairos ABMContext owns entity state and dispatch. Its PriorityResource owns the
waiting order. Innovate controls admission and the active service count.
Optional population-length `agent_priorities` supplies signed 32-bit integers. Higher values serve first among waiters, with FIFO ties. The
default is zero for every agent. An active service always runs to completion. An agent becomes adopted at service completion. Pending agents
do not transmit and cannot request twice. Completion events precede tied requests.
Waiting times round up to at least one tick. Service ticks must be positive and
at most 10^12; ticks per unit are 1–10^9, with a default of 10^6. Requested sample
times must map into 0–10^15 ticks. Counts include initial agents. Returned
`requests` and `completions` retain ticks/identities; agent and resource state is
at the final sample in a completed run.

For `kairos_network`, pause with either `pause_at_tick` (a nonnegative
integer below the horizon) or `pause_after_completions` (a positive
integer). A paused result has `status: "paused"` and an opaque
`checkpoint` string. Remove the pause option and pass the string as
`simulation_options.resume_checkpoint` with the same inputs, effective
parameters and simulation configuration. Preserve the string exactly
when saving or transferring it. The RNG metadata alone is insufficient
to resume a run.

Python, Julia and R have exchanged checkpoints for the same native
build on a continuous-hazard network fixture. Compatibility across
versions requires separate qualification.

Optional population-length `innovation_multipliers` and `imitation_multipliers`
scale each susceptible agent's p and q terms. They default to one and must be
finite and nonnegative; zero disables that hazard component. Optional
`agent_service_ticks` overrides the scalar duration per agent, including agents
waiting in the service queue. Each duration is an integer in 1–10^12. Defaults retain
the homogeneous engine and RNG stream. Metadata retains the effective arrays.

The network engine currently rejects continuous mechanistic covariate forcing and
arbitrary hazard functions. Checkpoint compatibility across versions remains open.
The separate `network_diffusion` family has its own scientific qualification.

## Historical lock-in

`lock_in` uses alpha1/alpha2, beta1/beta2, gamma1/gamma2 and positive market
potential m. Predictions require initial_values, with initial_time defaulting to
zero. Fitted states retain their conditioning origin. The historical equations
clip each state to [0,m] before evaluating the RHS and clip reported trajectories
to the same range. Reported rates are the historical RHS evaluated at reported
levels, not derivatives of output clipping. No simplex constraint is imposed.

Interior trajectories identify only alpha1, alpha2 and four feedback/competition
to-market combinations. Free m is confounded with beta/gamma parameters. Native
diagnostics explicitly warn about this structural ambiguity; a numerical full
Jacobian rank does not qualify unique parameter recovery. Boundary information
and calibrated uncertainty require separate qualification. Event splits
and broader lock-in variants remain required migration work.

## Historical deterministic network spillover

`network_diffusion` is a separate family from `kairos_network` event simulation.
Constructor options include square `adjacency`, optional unique `node_labels`
and coordinates, `base_model` (Bass by default), and spillover strength (0.15 by
default). Adjacency diagonals are cleared. Zero-sum rows have no peer contribution;
other rows are normalized by their sum. The native wrapper supports scalar Bass,
Logistic, Gompertz and Fisher–Pry bases without base covariates/events.

Parameters are `spillover_strength` and `node_0_p`, `node_0_q`, `node_0_m`, etc.
for Bass, or the corresponding names for the chosen base curve. Predictions are
base levels plus strength times row-normalized peer levels, clipped nonnegative
and cumulatively maximized within each requested batch. This historical batch
rule can matter for nonmonotone adjusted curves. Raw native `increments` preserve
historical sample differences with initial zero; they are not time derivatives.
Polars prediction columns use time_node rows. Observation orientation defaults
to time_node; node_time and historical_auto are explicit alternatives, with
historical_auto preferring node_time when dimensions are ambiguous.

Fitting preserves the historical algorithm: nodes are fitted independently and
spillover strength stays fixed. Adjusted training predictions may consequently
not fit observations; diagnostics flag that limitation. Native `node_fit_options`
can specify options per node. Nonempty global fit_options are rejected. Fitted
state retains graph, labels, coordinates, base model and node parameters.
No-time summaries retain graph statistics without inventing predictions.

Targeted interventions, full base-model composition/covariates, stochastic draws,
uncertainty and general scenario integration remain required migration work.
Unknown option fields fail explicitly. This deterministic wrapper does not
inherit the event engine's resource semantics or qualify full network parity.

## Historical policy-timing wrapper

`policy_hazard` reuses a scalar Bass, Logistic, Gompertz or Fisher–Pry baseline.
Constructor options are base_model, event_times, event_effects, optional
aligned event_labels and decay (default 0.5). Parameters use decay and base_p,
base_q, base_m for Bass, or the corresponding base-prefixed curve parameters.
At each event boundary and afterward, an effect contributes
`effect * exp(-decay*(time-event_time))` to a level multiplier of one plus the
summed contributions. Negative adjusted levels are clipped and a cumulative
maximum is applied within the requested batch. Sample increments start at zero
and are not time derivatives. This preserves the historical deterministic
wrapper, not a calibrated survival hazard or causal policy effect.

Fitting preserves the historical algorithm: fit the baseline to observations,
hold decay fixed, then apply policy multipliers. Diagnostics retain the baseline
fit and warn that adjusted predictions need not fit observations. Optional
base_fit_options apply to that baseline; nonempty global fit_options are rejected.
Summaries retain event times/labels, mean effect and decay without inventing
predictions when no times are requested. Fitted state retains policy inputs.
Native R transport preserves singleton event vectors as arrays.

Full base composition/covariates, estimating policy effects or decay,
stochastic draws, uncertainty and causal validation remain required work.
Unknown fields and malformed event arrays fail explicitly. Negative effects,
zero decay, pre-existing events and simultaneous events retain the analytic
level-adjustment interpretation; no event scheduler is copied here.

## Historical latent residual process

`latent_process` supports a scalar Bass, Logistic, Gompertz or Fisher–Pry base
curve. Native baseline least-squares fitting runs first; smoothing is held fixed
(`constructor_kwargs.smoothing`, default 0.3, domain [0,1]). The first latent
value is the first residual; subsequent values are
`smoothing * residual + (1 - smoothing) * previous_latent`. `latent_noise` is the
sample standard deviation of baseline residuals with one degree of freedom
removed. This is the historical residual proxy, not probabilistic state-space
inference or calibrated forecast uncertainty.

Saved state retains `base_*` parameters, `latent_smoothing`, `latent_noise` and
constructor arrays `latent_time`/`latent_values`. The arrays must be finite,
aligned and strictly increasing in time, with finite interval spans. Forecasts
linearly interpolate latent offsets and extend the first/last offsets outside
the fitted range; no latent arrays gives zero offset. Fit requests estimate new
latent arrays and reject supplied saved arrays. `base_fit_options` accepts native
base-fitting options on the JSON route. General wrapper `fit_options` must be
empty because smoothing and noise are not jointly optimized.

Predictions add baseline and offset, clip below zero and take the cumulative
maximum within each requested batch. Consequently, separate forecast batches
may differ from one combined batch if an earlier adjusted value was larger.
`increments` preserves historical sample differences, including the unusual
singleton behavior of returning the level; it is not a continuous derivative.
The simulation operation currently returns the deterministic point path with
explicit metadata. Noise draws, uncertainty, arbitrary nested base models,
base covariates and complete state-space inference remain required work.

Python `FitResult.predict` carries saved residual arrays through Polars. Julia
and R consume the same versioned state; R preserves even singleton latent arrays.
Summaries can omit time, and diagnostics warn about in-sample residual smoothing.

Adjusted-fit AIC/BIC are unavailable until effective degrees of freedom for the
residual smoother are qualified. Baseline optimizer diagnostics remain separate.

## Regime-switching diffusion

`regime_switching` fits independent scalar Bass, Logistic, Gompertz or Fisher–Pry
curves before a continuity adjustment. Constructor `split_index` explicitly
selects a zero-based boundary inside the observed series. Alternatively,
`split_time` selects the first observation at or after that time and retains the
exact time for fixed-time continuity, including boundaries between samples.
Do not supply both; each must leave observations on both sides. Without either, native
unpruned dynamic programming minimizes penalized L2 segment error on a grid:
`detector_penalty=3`, `detector_jump=5`, minimum segment length two. It selects
the first detected boundary, with midpoint fallback when no valid interior
boundary is detected. This independently implemented grid objective aligns with
the historical [ruptures Pelt defaults](https://centre-borelli.github.io/ruptures-docs/code-reference/detection/pelt-reference/).
Fixtures from ruptures 1.1.10 qualify constant, multiple-step, off-grid, short,
trend, small-step and smooth-curve cases; universal floating-point/tie or PELT performance
parity is not asserted. Native detection supports at most 2000 observations;
provide an explicit split for larger fits. Jump is 1–2000 and penalty positive.

Prediction mode is explicit:

- `historical_index` (default) applies the fitted boundary index to each requested
  batch. At the boundary, the second curve is shifted to equal the preceding
  first-curve sample. If the index reaches the penultimate batch position, only
  the first curve is used. This preserves the historical batch dependence.
- `fixed_time` uses saved `split_time`, shifting the second curve to equal the
  first curve at that exact time. Samples at or after the boundary use the second
  curve. Use this mode for forecasts that retain the same time-based switch
  across batches. Direct parameter requests must supply `split_time`.

Both modes clip below zero and take a batch cumulative maximum. Saved state
includes all regime parameters, boundary index/time, regime count, detector
configuration/breakpoints and residual sample deviation. Segments with fewer
than two observations are omitted; if neither segment remains, fit the whole
series as one regime. Each retained base fit still enforces its own numerical
and domain requirements. Fisher–Pry now allows two interior share observations
for an identified point fit, while reporting no residual degrees of freedom and
unavailable information criteria. Other scalar fit minima remain four.

Optional JSON `regime_fit_options` provides one native fit-option object per
retained regime; `base_fit_options` applies common options. Do not supply both.
General `fit_options` must be empty because boundary selection is separate.
Adjusted-fit AIC/BIC remain unavailable until split-selection degrees of freedom
are qualified. Segment optimizer diagnostics remain separate: continuity-adjusted
predictions need not fit the observations. The residual deviation is not
calibrated uncertainty. Sample increments preserve the historical singleton
return-level behavior, not a derivative. Summaries may omit time.

Python Polars, Julia and R retain the same state and boundary. Stochastic draws,
probabilistic regime inference, other changepoint objectives/search methods,
full nested base composition, base covariates and complete small-sample scientific
qualification remain required. Simulation currently returns a labelled
deterministic point path.

## Bayesian inference for scalar forecasts

Quadrature and Metropolis use aligned scalar observations with independent
Gaussian errors and a known positive `noise_sd`. The bounds define the
support. The optional `gaussian_priors` map takes a mean and positive standard
deviation for each specified parameter. Parameters omitted from the map have
uniform priors within their bounds.

Run `.venv-native/bin/python -I examples/native_bayesian.py` from the repository root.
The contract checks this displayed source against that executable file.

```python
"""Check a bounded four-chain sampler against an independent Gaussian posterior."""

import json
from math import isfinite
from statistics import mean, variance

from innovate import execute


def main() -> None:
    """Check support, draws, diagnostics and a predeclared Monte Carlo tolerance."""
    result = execute(
        "bayesian_metropolis_chains",
        forecast_model="logistic",
        parameter="L",
        bounds=[80, 120],
        proposal_step=4,
        gaussian_priors={"L": {"mean": 96, "sd": 2}},
        noise_sd=1,
        observed=[50],
        prediction={"inputs": {"time": [2]}, "parameters": {"k": 0.5, "x0": 2}},
        draws=4000,
        warmup=1000,
        chains=[
            {"seed": 41, "initial": {"L": 83}},
            {"seed": 42, "initial": {"L": 94}},
            {"seed": 43, "initial": {"L": 106}},
            {"seed": 44, "initial": {"L": 118}},
        ],
    )["result"]
    chains = result["chains"]
    assert len(chains) == 4
    assert [chain["seed"] for chain in chains] == [41, 42, 43, 44]
    for chain in chains:
        assert chain["draws"] == 4000 and chain["warmup"] == 1000
        assert len(chain["samples"]) == 4000
        assert 0 < chain["acceptance_rate"] < 1
    samples = [sample for chain in chains for sample in chain["samples"]]
    assert all(isfinite(sample) and 80 <= sample <= 120 for sample in samples)

    # At the midpoint, y=L/2. Combining N(96, 4) with y=50, noise variance 1,
    # gives N(98, 2), truncated to [80, 120]. Both bounds exceed 12 SD from 98;
    # truncation is negligible at the declared tolerances, not mathematically absent.
    # These fixed tolerances check a seeded fixture; they are not interval coverage.
    assert abs(mean(samples) - 98.0) < 0.2
    assert abs(variance(samples) - 2.0) < 0.3
    assert len(result["diagnostics"]) == 1
    diagnostic = result["diagnostics"][0]
    assert diagnostic["parameter"] == "L"
    assert diagnostic["effective_sample_size"]["draws_used"] == 16000
    rhat = diagnostic["split_rhat"]["rhat"]
    assert isfinite(rhat) and 0 < rhat < 1.05
    for name in ("bulk_ess", "tail_ess"):
        effective_samples = diagnostic["effective_sample_size"][name]
        assert isfinite(effective_samples) and effective_samples > 1000
    # Diagnostic thresholds are warning criteria, not proof of convergence.
    print(json.dumps({"chains": len(chains), "draws": len(samples), "posterior_oracle_passed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
```

At the Logistic midpoint the likelihood is `50 ~ Normal(L / 2, 1)`.
Combining it with the `Normal(96, 4)` prior, with variance as the second argument,
gives a `Normal(98, 2)` posterior truncated to `[80, 120]`. Both bounds are more
than twelve posterior standard deviations from the mean. The example checks
the pooled sample mean within 0.2 and variance within 0.3 of this independent
target. These are predeclared tolerances for a seeded numerical fixture; they
do not establish interval coverage or validate a real population model.

The result includes each chain's retained draws and per-parameter rank-normalized
split R-hat and bulk and tail effective sample sizes. To sample jointly, supply
two to sixteen ordered `axes` entries, each with `parameter`, `bounds`, and
`proposal_step`, and initialize every sampled parameter in each chain.
`bayesian_parameter_grid` instead uses `points` per axis for one or two axes,
with at most 10,000 cells, and returns normalized masses and credible intervals.
Julia and R accept the same operation and payload through their kernel bridges.
R-hat and effective sample size are diagnostic warning criteria. Passing the
example's thresholds does not prove convergence in a different inference problem.

## Hierarchical grouped point models

`hierarchical` supports scalar Bass, Logistic, Gompertz or Fisher–Pry base curves
and 1–128 unique nonempty `groups` labels. The label `global` is reserved for
parameter names. It provides grouped point fits; partial pooling, hierarchical
Bayesian inference and calibrated uncertainty remain required work.

Grouped observations may be a JSON mapping from each label to a time-aligned
series, or a time-by-group matrix in declared group order. Python Polars uses
`observed_columns` to select the matrix columns. A single observed vector is a
shared **per-group series**: fit once, set each group adjustment to zero and sum
the resulting group curves for the aggregate. It is not interpreted as an
already aggregated population series.

Parameterization is explicit:

- `historical_additive` preserves the historical grouped-data fit:
  independently fit each group, store its absolute parameters under `group_*`,
  and store their arithmetic means under `global_*`. Prediction uses
  `global_* + group_*`, including the original extra mean contribution.
- `centered` (default) stores each independent group estimate minus its global mean under
  `group_*`. The same addition therefore recovers the independently fitted group
  curve. This is a point-fit encoding, without shrinkage or joint inference.

Shared-series fitting sets all group adjustments to zero in both variants.
Direct parameter prediction requires all global/group keys and finite effective
parameters valid for the chosen base curve. Global means divide before summing
to avoid unnecessary overflow; centered subtraction can still lose precision
for extreme parameter scale differences, so broad conditioning qualification
remains open. Aggregate or effective-parameter overflow is an explicit error.

`output="aggregate"` returns the summed `values` and individual `group_rows`.
`output="groups"` returns time-by-group `rows` and separate `aggregate_values`;
Python Polars preserves every group as `prediction_1`, `prediction_2`, etc.
Saved state retains labels, parameterization and output selection. Summaries may
omit time and report effective group parameters. Simulation currently returns
labelled deterministic point paths; stochastic draws remain required.

`fit_options` accepts common native scalar fitting options. JSON
`base_fit_options` is an alias, and `group_fit_options` supplies a mapping of one
option object per group for grouped observations. Conflicting common/per-group
options are rejected. Baseline group diagnostics remain separate from wrapper
metrics. Shared-series wrapper metrics compare the summed prediction with the
shared observed vector, preserving the historical aggregate scoring convention.
Grouped metrics compare each predicted group with its own series.

AIC/BIC remain unavailable because effective degrees of freedom for global/group
decomposition are unqualified. Historical adoption increments prepend the first
sample difference; the previously failing singleton case is defined as zero.
They are not continuous derivatives. Native Polars rate output remains unavailable.
Group base covariates, nested base families, shrinkage and full hierarchical
inference remain required. R preserves singleton group labels and mapped series.
Diagnostics now compute means without overflowing sums of large constant values;
undefined or unrepresentable total variation yields unavailable R².

## Weighted scalar fits and mixtures

Scalar Bass, Logistic, Gompertz and Fisher–Pry fitting accepts
`fit_options.weights`: finite nonnegative precision weights aligned with time.
The objective is `sum(weight * residual²)`; zero weights exclude observations.
At least as many positive weights as curve parameters are required, and the
weighted local Jacobian still reports numerical rank. Optimization divides
square-root weights by the square root of their maximum for scale invariance;
parameter scaling uses positively weighted observations. Model-specific seed
estimation may still use the whole series; explicit `initial_parameters` remains
available. Descriptive residual metrics are unweighted, while optimizer/rank and
`weighting.weighted_rss_normalized_by_max` describe the weighted objective.
AIC/BIC and calibrated uncertainty remain unavailable until precision/variance
likelihood conventions are qualified. Independent SciPy 1.18.1 least-squares
fixtures, zero-weight outliers and common-weight rescaling are tested.

`mixture` accepts 1–16 scalar `components` (Bass, Logistic, Gompertz or Fisher–Pry),
with optional aligned nonnegative weights summing to one. Parameters use
`model_0_*`, `model_1_*`, etc. and `weight_0`, `weight_1`, etc. Component rows are
retained alongside the weighted-mean `values`. Both model variants are explicit:

- `gaussian_residual` (default) models each time-point observation as a Gaussian
  mixture around the component curve values, with a shared positive
  `noise_scale`. Responsibilities use log-sum-exp of the residual Gaussian
  densities. M-steps use weighted native curve fits, mean responsibilities for
  weights, and the corresponding responsibility-weighted residual variance.
- `historical_adoption` preserves the historical adoption-level heuristic:
  responsibilities normalize `weight * (component_prediction + 1e-9)` without
  an observation residual. Component updates use responsibility + 1e-9 as
  precision weights. Its objective is labelled an adoption log score, not an
  observation likelihood, and its pre-update stopping convention is explicit.

Gaussian `noise_floor` (default 1e-6, in observation units) bounds the fitted
standard deviation. This regularization prevents variance collapse, affects the
solution and is reported when active; it does not calibrate uncertainty.
`max_iter` defaults to 100 (1–200), and positive `tol` defaults to 1e-6. Gaussian
stopping compares successive log-likelihoods; historical stopping compares
successive pre-update adoption scores. Global optimality and class
identifiability are not established by local convergence.

By default each component is initialized with a native fit to the full series;
Gaussian mode then applies a deterministic parameter/time spread to break exact
symmetry. JSON `component_fit_options` supplies per-component initial-fit seeds
and iteration limits. Alternatively `fit_options.initial_parameters` supplies a
complete mixture state as a warm start, including `noise_scale` for Gaussian
mode. `fit_options.component_max_iterations` controls the default inner limit
(1–2000). Initial component fits, all attempted updates and objective history
remain inspectable. A failed/non-converged component update rolls back the entire
iteration and reports non-convergence; a Gaussian likelihood decrease beyond
the explicit numerical guard also rolls back. Update records identify whether
that iteration was accepted. Zero-weight components retain their parameters.

Saved state retains components, mode, weights, noise scale and numerical options.
Python Polars, Julia and R fit and forecast through the same Rust implementation;
R preserves singleton component/weight arrays. Summaries can omit time. Prediction
without observations returns the weighted mean; diagnose/fit with observations
also returns mode-specific responsibilities and objective. Residual diagnostics
describe the mean curve, which is distinct from latent-class observation
likelihood. Mixture AIC/BIC remain unavailable pending singularity/effective-
degrees-of-freedom qualification. Summary calls do not infer optimizer convergence
from saved parameters alone.

This is not joint least-squares decomposition of one aggregate trajectory.
Nested base families, component-rate helpers, stochastic draws, calibrated
uncertainty, class-selection diagnostics, heteroscedastic models and full mixture
inference/scientific qualification remain required. Simulation currently returns
the deterministic weighted point path. No new native runtime dependency is added.

## Composite growth ODEs

`composite` combines 1–16 scalar Bass, Logistic, Gompertz or Fisher–Pry growth
ODEs with signed off-diagonal linear interactions:
`dy_i/dt = growth_i(y_i) - sum(alpha_i_j * y_j)`. Parameter names retain one-based
suffixes (`p_1`, `m_1`, `L_2`, etc.) and `alpha_1_2`, `alpha_2_1`, etc. Positive
interaction coefficients subtract peer levels; negative coefficients stimulate
growth. No row/share normalization or nonnegative output clipping is applied.
Negative levels remain visible with a warning, and rates are the signed ODE RHS.

Origin behavior is explicit:

- `fixed` (default) uses a retained initial time/state. Prediction defaults to
  time zero and zero state. Fit without explicit initial values conditions on
  the first observed row at that row's time, excludes it from the fit objective
  and diagnostic counts, and retains that state for forecasts. An explicit
  `initial_time` must equal that first time when state is inferred.
- `historical_batch` preserves zero initial state at each requested batch start.
  It rejects explicit initial time/values and intentionally resets forecasts.
  Singleton batches return their well-defined zero initial row.

Explicit initial values must be finite nonnegative and match component count;
initial time must be finite nonnegative and no later than sampled times. Stored
fixed state permits forecast batches to start later without resetting adoption.
Matrix observations and predictions use time-by-product order. Python Polars
preserves all level/rate columns; Julia and R consume identical native state.

The ODEs differ from their scalar closed forms: Logistic `x0`, Gompertz `b` and
Fisher–Pry `t0` do not enter the growth RHS. Those named parameters are retained
but held fixed during fitting and listed as structurally absent. Gompertz uses
zero own-growth when its current level is nonpositive, preserving the historical
boundary rule. The native Fisher–Pry adapter applies `alpha * y * (1-y)`;
the old generic composite call did not match that Python method's signature.
Zero-initialized Logistic/Gompertz/Fisher–Pry components have no own-growth until
an external interaction or explicit initial condition seeds adoption.

Fitting optimizes active positive growth coefficients/capacities and signed
interactions through native least squares. Local rank diagnostics cover active
parameters. `fit_options.initial_parameters` supplies the complete named
state. Capacity bounds default to `positive`, allowing stimulated trajectories
to exceed an intrinsic capacity. `capacity_bounds="historical_observed"` retains
the old observed-maximum capacity floors. Optional zero-diagonal square `alpha`
is a fitting seed; complete parameter values govern prediction. Midpoints and
other absent parameters retain their supplied seed (or native
heuristic) and are not estimated. Initial-condition uncertainty is not propagated.
Composite information criteria remain unavailable pending conditional/structural
degrees-of-freedom qualification. Rank covers active ODE parameters only.

The solver is the shared adaptive step-doubling RK4 with Richardson correction,
absolute/relative tolerances 1e-10 and a 200000-step budget. Nonfinite trajectories
or budget/step failures are explicit errors. Closed-form decoupled cases and an
independent SciPy DOP853 heterogeneous signed-interaction reference pass. Full
stiff-system/BDF workload parity, nested models, covariates/events, uncertainty,
stochastic paths, weighted composite fitting and scientific qualification remain
required. Simulation currently returns the deterministic ODE path.

Mechanistic Bass and Gompertz, historical Logistic, Norton–Bass, and the
competition models also accept `fit_options.weights`. For multi-output fits,
one weight per time row applies to every output. Zero-weight rows contribute
no residuals. In competition fits, a row used to condition the initial state
still sets that state and is excluded from the weighted residual objective.
Weighted loss and local rank are reported separately from unweighted descriptive
metrics. Information criteria are unavailable for weighted fits.
All sixteen model families now have bounded native support. Every original and
planned feature remains a release obligation, and the ledger still blocks release.

## Mechanistic Bass/Gompertz covariates and events

Declaring `covariates`, `t_event` or `prediction_mode` for Bass/Gompertz selects
an ODE execution variant. Plain requests without those options retain the current
closed-form scalar contract. `ode_fixed` is the ODE default: initial time zero,
Bass initial value zero, Gompertz initial value 1e-6. Override `initial_time` and
`initial_value` explicitly. `ode_historical_batch` instead uses those historical
initial values at each requested batch start and rejects custom initial settings.
Gompertz ODE `b` does not set the initial value; it is absent from the RHS, unlike
the plain scalar closed-form shape parameter.

Each declared covariate requires `beta_p_name`, `beta_q_name`, `beta_m_name` for
Bass, or `beta_a_name`, `beta_b_name`, `beta_c_name` for Gompertz. Coefficients
linearly modify the corresponding parameters using linearly interpolated controls
with endpoint hold. An optional finite `t_event` switches to required `*_post`
base parameters at that exact time. State remains continuous; returned event
rates use the right-hand regime. Integration stops at controls and event
boundaries, and each event segment uses its own fixed regime at the endpoint.
Gompertz `b`, `b_post` and `beta_b_*` remain named but do not enter the ODE; fitting
holds them fixed and reports their structural absence.

`coefficient_layout="corrected"` uses the shared named beta block in both regimes.
Explicit `historical` reproduces the old pre-event indexing that could read post
parameters as coefficients. This is separate from origin behavior. The old
Gompertz scalar interpolation-grid failure is corrected, not asserted as working
legacy covariate parity. The default `effective_domain="physical"` requires
finite nonnegative effective innovation/imitation or Gompertz growth, and positive
capacity. Explicit `historical` allows signed effective coefficients and returns
zero own-growth for nonpositive capacity, preserving the old boundary convention.
Base parameter domains remain validated in both modes.

Forcing state is `covariate_history={"time": [...], "values": {"name": [...]}}`.
It must cover the initial time and match declared names with finite aligned
values. Fit retains the full training forcing. Forecast controls append new
knots; overlapping controls must exactly match the saved interpolation. Revision
requires an explicit replacement history/state. JSON floating-point round trips
are qualified bit-for-bit with the pinned serializer's `float_roundtrip` feature;
unchanged controls are not mistaken for revisions because of parser rounding.
R preserves singleton named history values.

Python Polars fits/forecasts use declared covariate columns. Supply every new
control knot needed between the saved history and forecast endpoint: one future
point defines one linear segment from the saved last control, not an unobserved
sequence of intermediate controls. JSON callers may omit new controls to hold
saved endpoints. Prediction returns an updated `state` for explicit carry-forward;
Polars `FitResult.predict` returns a table and does not mutate its fitted history.
Use the native JSON result/state or a full future-control table when advancing
multiple forecast batches. Past queries reuse saved forcing rather than rewriting
it. Controls, initial conditions and their uncertainty are treated as fixed inputs.

`projection="raw"` is the default. Optional `cumulative_max` reproduces sample
maximum projection: historical batch mode uses requested samples only; fixed
mode includes saved forcing knots, event and query samples. This is a discrete
reporting-grid maximum, not a continuous-time maximum, so a new query sample can
expose an unseen peak. `raw_values` remain inspectable. Rates are the signed
underlying ODE RHS at raw state, not derivatives of a projected report. Request
Polars `quantity="rate"` only with the ODE variant; the plain scalar rate API
remains separately unqualified.

Mechanistic fitting estimates active coefficients with native least squares
and reports local rank and convergence. Without `fit_options.initial_parameters`,
it scales starting capacities to the observed levels, tries a deterministic
rate grid, and starts covariate slopes at zero. An explicit start must supply
the complete parameter map. The unused Gompertz shape parameters default to
one for automatic starts and retain their supplied values for explicit starts.
The optimizer reports each start's outcome in `start_reports` and identifies
the selected entry with `selected_start_index`, a zero-based index. Information criteria and calibrated uncertainty remain unavailable;
`fit_options.weights` accepts one finite nonnegative weight per observation
and requires at least four positive rows and at least as many positive rows
as active parameters. Zero weights exclude residuals from the objective.
The fit still conditions on its initial state. Descriptive metrics remain
unweighted, and the weighted objective and local rank have separate diagnostics.
Closed-form Bass/Gompertz forcing/event cases and a separately integrated SciPy
DOP853 fit/forecast reference pass. The explicit RK4 solver's stiff/LSODA workload
parity, multiple events, further model families, forcing
interoperability with richer Kairos networks, stochastic paths and full scientific
qualification remain required. No numerical binding fallback is used.

### Directed and weighted Kairos contacts

Set `simulation_options.directed` to true to interpret each edge as
`[source, target]`. Otherwise each edge contributes in both directions.
Optional `edge_weights` supplies one finite positive value per edge, in
edge order. Imitation uses the completed adopters' share of incoming weight;
isolated agents have zero imitation exposure. Pending agents still do not
transmit and adoption still occurs at service completion. Unweighted
undirected requests retain their original simulation traces.

A Kairos network can also take a `schedule` of up to 64 changes at strictly
increasing positive integer ticks. Each change can replace the entire contact
graph with `edges` and optional `edge_weights`, set a new `service_ticks`
duration for subsequent service starts, or do both. Direction stays fixed
for the run. At a shared tick, completions and their queue handoffs run before
the change. The new graph governs subsequent request draws, and services
already underway finish at their original times. The result lists the
schedule ticks applied through the requested horizon.


A schedule entry can replace `agent_priorities` with one signed 32-bit integer
per agent. Waiting agents are reordered by the new priorities, with their
original queue entry times breaking ties. Existing waiting deadlines and
active services are preserved.

Set scheduled `agent_service_ticks` to one duration per agent, each an integer
from 1 through 10^12 ticks. It replaces an earlier scalar duration override and
applies when a waiting or future agent starts service. Each schedule entry
accepts one duration form, `service_ticks` or `agent_service_ticks`.

```json
[
  {"tick": 4, "agent_priorities": [0, 0, 10]},
  {"tick": 5, "agent_service_ticks": [2, 3, 4]}
]
```

Set `max_wait_ticks` to a positive integer up to 10^12 to give every
queued agent a deadline. An agent who starts service at that tick keeps
the slot. Other expired agents appear in `abandonments` and enter `withdrawn`
unless a retry is configured. Without a deadline, agents keep waiting for service.


Initial and scheduled `capacity` can range from zero to the population.
An increase immediately admits eligible waiters in priority order. A decrease
allows active services to finish and pauses admissions while occupancy meets
or exceeds the new limit. At a shared tick, existing completion handoffs
and waiting deadlines are processed before the scheduled capacity change.
The result reports the initial `capacity`, `current_capacity`,
`in_service`, and nonnegative `available` slots.

This `simulation_options` example starts closed and opens two slots at
tick 7. With three agents, Bass parameters `p=1e12, q=0, m=3` and
observations through tick 11, services complete at ticks 9, 9 and 11.

```json
{
  "engine": "kairos_network",
  "population": 3,
  "seed": 42,
  "edges": [],
  "capacity": 0,
  "service_ticks": 5,
  "ticks_per_unit": 1,
  "schedule": [{"tick": 7, "capacity": 2, "service_ticks": 2}]
}
```


Set `max_queue_length` to limit the number of waiting agents. Zero permits
immediate service only. Requests that find all slots and queue places occupied
appear in `balks`. By default those agents withdraw. Optional `retry` sets
positive `delay_ticks` (up to 10^12) and `max_retries` (1–16 per agent) for both
balks and waiting expiry. After cooldown the agent becomes susceptible again
and competes through the same hazard as other eligible agents. It does not
retain its previous queue position. `retry_eligibility` records the cooldown
end, and `requests` records all attempts, including balks.

A schedule entry can replace `retry` with a new delay and budget or set
`"retry": null` to disable future retries. The budget counts retries already
scheduled for each agent. A pending retry keeps its original wake tick.

The `service_starts` trace records each admitted agent, admission tick,
request tick, wait in ticks, assigned service duration and scheduled
completion tick. A service that extends beyond the simulation horizon
appears in this trace. The `completions` trace records observed completions
through the horizon.

The `resource_metrics` totals cover tick zero through the last requested
observation. Occupied slot-ticks include service in progress, clipped at the
horizon. Configured slot-ticks integrate scheduled capacity over the same
interval. Occupied slot-ticks can exceed configured slot-ticks when service
continues through a closure. `over_capacity_slot_ticks` integrates service
above scheduled capacity. `idle_slot_ticks` measures unused configured
capacity, so occupied plus idle equals configured plus over-capacity.
Queue wait is split into admitted, abandoned,
and still waiting at the horizon. Balks and retry cooldown do not count as
queue wait. The peaks `peak_queue_length` and `peak_in_service` count
waiting agents and concurrent service after dispatch. An immediate service
start contributes to concurrent service without adding to the queue peak.


A scheduled change can replace either Bass hazard coefficient using `p` or
`q`. Values must be finite and nonnegative. The new coefficient applies
until replaced again and still uses each agent's multiplier. This supports
piecewise constant interventions, including turning a hazard off. It leaves
active service and queue positions intact. The result reports `current_p`
and `current_q` at the requested horizon.

A change can independently replace `innovation_multipliers` or
`imitation_multipliers`. Each replacement contains one finite nonnegative
value per agent in agent-index order. The active vectors are returned in
simulation metadata. Changes apply to subsequent adoption attempts, including
after retry cooldown, while existing requests keep their place in service or
in the queue.

For continuous forcing, set `hazard_forcing` with aligned `time`,
`innovation` and `imitation` arrays. Times are in model units, start at zero
and increase strictly. The nonnegative factors interpolate linearly and hold
their last values beyond the final knot. They multiply the active scheduled
Bass coefficients and compose with agent multipliers. The simulator integrates
the total hazard up to the next deterministic event, selects the requesting
agent at the continuous crossing time, and rounds dispatch up to a Kairos
tick. Existing deterministic events retain precedence on tick ties.

### Historical Norton–Bass covariates

The `legacy_cannibalization` variant accepts declared `covariates` and named
`beta_p1_price`, `beta_q1_price`, `beta_m1_price` coefficients for each
generation and declared control. The effective p, q and m are additive linear
functions of the controls. This preserves the historical signed coefficient
semantics, including zero derivative where effective market capacity is
nonpositive. Signed rates can arise through cannibalization or forcing.

Controls use the same linearly interpolated `covariate_history` as mechanistic
Bass and Gompertz. History must cover the initial time. Outside its grid,
endpoint values hold. The returned `state` carries the history for later
forecasts, and overlapping input values must agree with that saved history.
Integration includes all control knots. Without explicit initial parameters,
least squares starts from a grid of base coefficients with all beta slopes
initially zero and estimates the declared slopes. An explicit start must
include every beta coefficient. The
`successive_generations` variant has the fraction-space forcing semantics
described below. Event inference, uncertainty and broader independent fitting qualification remain open.

### Historical Lotka–Volterra covariates

The two-technology model accepts declared `covariates` with coefficients
`beta_alpha1_price`, `beta_beta1_price`, `beta_alpha2_price` and
`beta_beta2_price` for a control named `price`. Each control adds to the four
base rates using linear interpolation and endpoint holds. Signed control
effects and effective rates preserve the historical equations. Base rates
remain nonnegative. Shares are integrated without output clipping.

Provide an explicit `initial_time` and `initial_values`, or fit conditionally on
the first observed row. A saved `covariate_history` must cover that initial time.
The returned prediction `state` carries the history into later requests. New
control knots may extend it, while overlapping values must agree with the saved
history. The Python Polars fit path accepts declared controls as frame columns.
Conditional fitting accepts a complete named `initial_parameters` map or
starts from a deterministic base rate grid with zero covariate slopes. Each
fit reports local identifiability and conditions on the forcing and initial state.

The independent reference uses SciPy DOP853 with every control knot split into
its own integration interval. Initial-state uncertainty and
broader scientific qualification
remain required work.

### Historical multi-product covariates

The multi-product model accepts declared controls with one named coefficient
for every base parameter and control. For a control named `price`, examples
include `beta_p1_price`, `beta_q2_price`, `beta_m1_price` and
`beta_alpha_1_2_price`. The default `coefficient_layout="corrected"`
adds each coefficient to its named parameter. The explicit `"historical"`
layout preserves the old flat indexing, which read the p/q/m coefficients in
product triples even though their names were listed in p, q and m groups.
Interaction coefficients retain their named order in both layouts. Saved
history and state retain the chosen layout.

Controls use linear interpolation with endpoint holds, and integration splits
at every control knot. Effective coefficients may be signed. A product whose
effective capacity is nonpositive has zero derivative, as in the historical
equations. Conditional fitting requires complete initial parameters and reports
local identifiability. Coefficients scale with the units of their corresponding
base parameter during optimization. The shared optimizer now tests relative
objective improvement so a small residual alone does not trigger the previous
absolute-improvement stop.

Independent SciPy references cover interacting trajectories, signed rates and
forecasts in both layouts. A perturbed conditional fit is checked against a
held-out forecast. Automatic starts, event splits, constrained scientific
domains and uncertainty remain required qualification work.

### Complementary goods covariates

The complementary-goods equations now accept the shared declared-control
contract. A control named `price` has `beta_k1_price`, `beta_k2_price`,
`beta_c1_price` and `beta_c2_price` coefficients, added to intrinsic growth
and the two cross-product effects. Base parameters are nonnegative, while
control effects may be signed. Effective cross-product effects can therefore
represent inhibition during part of a scenario.

Controls use linear interpolation with endpoint holds. Saved state carries
the initial values, origin and forcing history for forecasts. An overlapping
control value must agree with the saved history. Conditional fitting accepts
an explicit parameter map or initializes covariate slopes at zero. Fits
report local identifiability.
The Polars fit path reads controls from declared frame columns.

This extends the original complementary-goods equations, whose historical
implementation had no covariate path. Analytic uncoupled growth and an
independent interacting SciPy reference verify levels, rates, conditional fit
and a held-out forecast with both starting modes. Uncertainty and broader
scientific qualification remain required.


### Lock-in covariates and instantaneous capacity

Lock-in now uses the shared declared-control contract for all seven base
parameters. A control named `price` adds `beta_alpha1_price`,
`beta_alpha2_price`, `beta_beta1_price`, `beta_beta2_price`,
`beta_gamma1_price`, `beta_gamma2_price` and `beta_m_price`.
Controls interpolate linearly and hold their endpoint values. The effective
capacity must remain finite and positive across the integration interval.

The historical equations clip levels inside the RHS. Reported levels are also
clipped to the capacity at each requested time. This reporting projection
preserves the integrated state, so levels can recover when capacity rises again.
The returned rates retain their historical meaning as the RHS at clipped levels.
They exclude changes caused solely by the reporting projection. For example,
constant integrated levels of 95 report 90 when capacity falls to 90, with a
zero RHS rate. Metadata identifies this projection explicitly.

Saved state retains the original initial values, time origin and forcing history.
Conditional fitting requires complete explicit initial parameters. With forcing,
the base m may be lower than observed levels because controls add to capacity.
The unforced model retains its original lower bound on m. The known interior
capacity/feedback confounding remains in fit diagnostics, with a rank upper
bound based on the expanded parameter count. Numerical convergence alone
cannot establish unique parameter recovery.

An independent interacting DOP853 reference covers predictions, RHS rates,
conditional fitting and a held-out forecast. Additional tests cover capacity
decline/recovery, invalid capacity and forced fits with a small base m.
Event splits, initial-state and control uncertainty, calibrated parameter
inference and wider scientific qualification remain required.

### Fixed parameter events for ecosystem ODEs

Lotka–Volterra, multi-product, complementary-goods and lock-in models accept
`parameter_events` in their constructor options. For example,
`[{"time":3.0,"parameters":{"alpha1":0.4}},{"time":6.0,"parameters":{"alpha1":0.2}}]`
changes Lotka–Volterra growth at two known times. Each override persists until
another event changes the same parameter. Other parameters retain their active
values. Events can also override declared beta coefficients, and compose with
the existing linear control history. Discovery advertises
`supports_parameter_events` for these four models.

Provide at most 32 events at finite, strictly increasing nonnegative times.
Each event must contain at least one existing named parameter. Base parameter
overrides retain the model's base domains, while beta coefficients may be
signed. Events at or before the initial time select the active parameters at
that initial state. The initial state itself remains the supplied or fitted
conditioning value.

The solver splits exactly at event boundaries, carrying the integrated state
continuously. Boundary states use the pre-event integration and reported rates
use the parameters active on the right. For lock-in, a capacity jump may change
the clipped report immediately while retaining the integrated trajectory.
Both sides of each used lock-in interval require positive finite capacity.
Saved state retains the complete event list for forecasts.

These are fixed interventions. Conditional fitting estimates the original
parameter map while keeping overrides fixed, and diagnostics explicitly record
that distinction. Fixed overrides can change structural identifiability, so
lock-in reports the need for event-specific analysis instead of reusing its
unforced confounding rank bound. Estimating event times, estimating override
values and calibrated intervention uncertainty remain required inference work.
Independent DOP853 references cover two off-grid events, control composition,
signed rates and saved-state forecasts for all four models.

### Historical Norton–Bass parameter events

The `legacy_cannibalization` ODE variant supports the same fixed
`parameter_events` contract as the ecosystem models. Overrides can change
named generation parameters such as `p1`, `q2` and `m2`, or declared
coefficients such as `beta_q2_price`. Values persist until replaced again.
The integrated generation levels remain continuous, and boundary rates use
the active right-hand parameters. Control history and the complete event list
persist in saved state for forecasts.

Event validation and continuous-state segmentation now share one Rust
implementation across the ecosystem and Norton–Bass models. Both Norton–Bass
variants accept fixed events. Unforced successive generations retain their
existing launch and closed-form conventions.

An analytic two-generation case verifies cumulative hazard, boundary rates
and forecasts after a coefficient override. An independent interacting DOP853
reference covers two off-grid events and a perturbed conditional fit. Event
times and overrides are fixed inputs. Estimating them, propagating uncertainty
and wider scientific qualification remain required.


### Successive-generation controls and events

The `successive_generations` variant now accepts declared controls and fixed
parameter events. Each generation starts with zero fraction at its entry in
`launch_times`. Control history must cover the first launch. From each launch
onward, effective innovation and imitation
must be finite and nonnegative, and capacity must be finite and positive.
Control coefficients may be signed when those effective domains hold.
Pre-launch levels and rates are zero.

The forced path integrates log survival `u`, with fraction `F=1-exp(-u)`
and `du/dt=p(t)+q(t)*F`. This keeps fractions in their physical range.
All launch, control and event boundaries split integration. Events carry the
fraction continuously. The successive market inheritance equations then map
those fractions to reported generation levels.

Reported rates include the derivative of effective capacity, including its
contribution to inherited markets. At control knots they use the right-hand
slope, and after the last control knot the endpoint hold has zero slope.
Capacity derivatives are evaluated for the report at each requested time.
The calculation reorders scaling if an intermediate control slope overflows
or underflows, so a representable capacity rate can still be returned.
A fixed capacity override can jump reported levels while fractions remain
continuous. Rates describe the right-hand evolution between events and exclude
impulses from those jumps. Metadata states this convention.

The analytic case checks two launch times, variable capacity, rate terms and
an endpoint-held forecast. The independent reference integrates fractions
directly with DOP853, while Rust integrates log survival. It covers an off-grid
launch, two events, conditional fitting and a held-out forecast. Launch times
and interventions remain fixed inputs. Their inference, calibrated uncertainty,
and wider scientific qualification remain required.

### Norton–Bass threshold summaries

The native summarize, diagnose and fit responses include
`threshold_diagnostics` for each generation. These offsets use the named
base innovation and imitation rates. The historical `half_life` label
denotes the Bass inflection offset. Takeoff clips at zero, and saturation
retains its signed historical offset. The reference time is the launch for
successive generations and the integration origin for historical generations.
Zero rates retain the historical zero values with a degenerate status.
These are baseline heuristics. Thresholds under forcing, interactions and
interventions require separate trajectory analysis.

## Batch fitting and forecasting

A native `fit_batch` or `predict_batch` request has one model key and
an ordered list of datasets. Each dataset carries its own inputs and
options or fitted state. Forecast grids can have different lengths.
The response has one row per input, with a result on success or an
error for that dataset.

Run `.venv-native/bin/python -I examples/native_batch.py` from the repository root.
The contract checks this displayed source against that executable file.

```python
"""Verify ordered batch results, analytic values and worker-count validation."""

import json
from math import exp, isclose

from innovate import NativeError, predict_batch


def main() -> None:
    """Compare one and two workers using two distinct synthetic Logistic curves."""
    datasets = [
        {"inputs": {"time": [0, 1]}, "parameters": {"L": 100, "k": 1, "x0": 0}},
        {"inputs": {"time": [0]}, "parameters": {"L": 200, "k": 1, "x0": 0}},
    ]
    rows = predict_batch("logistic", datasets, workers=2)
    assert rows == predict_batch("logistic", datasets, workers=1)
    assert len(rows) == 2 and all(row["error"] is None for row in rows)
    assert rows[0]["result"]["values"][0] == 50
    assert isclose(rows[0]["result"]["values"][1], 100 / (1 + exp(-1)), abs_tol=1e-12, rel_tol=0)
    assert rows[1]["result"]["values"] == [100]
    rejected = False
    try:
        predict_batch("logistic", datasets, workers=0)
    except NativeError as error:
        rejected = error.code == "invalid_request"
    assert rejected
    print(
        json.dumps(
            {"batch_items": len(rows), "invalid_workers_rejected": rejected, "worker_invariant": True}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
```

Julia and R can call these operations through `kernel_call` with
`datasets` and optional `workers` in the payload. Python callers
working with Polars frames can use `fit_frames` and `predict_frames`
to apply the native frame entry points in input order.

## Scenario grid rankings

For a complete grid with policy, competition, and substitution examples, see
[Scenario experiment workflows](/innovate/tutorials/scenario-experiment-workflows/).

In `execute_scenario_grid`, pass `rank_scenarios` with at least two
distinct scenario IDs. Each selected scenario must return a completed
scalar path on the same strictly increasing time grid. When supplied,
scenario specifications must agree on the time unit and reference year.
The response includes those values in `ranking.time_basis`.

Set `rank_metric` to `final_adoption` (the default) to rank by the last
sample or `adoption_area` to rank by trapezoidal area over the common
grid. Both put larger scores first. For `time_to_threshold`, supply
`rank_threshold`. This ranks earlier crossings first, interpolating
between samples, and reports the requested value in `ranking.threshold`.
Paths that never reach it have null scores and places. Equal scores share
a place, and ranking errors leave the individual executions available.

## Execution modes

Call `discover_execution_modes` to inspect the native execution modes.
Batch fit, batch prediction, and scenario grids accept up to 32 local
CPU workers and return results in input order. The worker count is a
request limit. Accelerator, distributed, and scheduler execution are
listed as unqualified.

Within a Slurm allocation, run the native benchmark runner with
`--slurm-allocation` and an explicit `--output` path. It uses
`SLURM_CPUS_PER_TASK` as the worker budget, capped at the native limit.
This option consumes an existing allocation. The runner remains a
single-process workload with local CPU workers.

To split independent benchmark cases across Open MPI ranks, run from
the repository root with the repo-local Python environment and native
wheel installed. Choose a rank count no greater than the number of
cases in the selected policy.

```bash
receipt_dir=$(mktemp -d)
mpirun -n 2 .venv-native/bin/python scripts/run_native_benchmarks.py \
  --mpi-shard --policy fast --workers 1 \
  --output "$receipt_dir/rank-{rank}.json"
```

For an existing Slurm allocation, launch a step with a shared receipt
directory and the same native wheel available on every task. The
example requests two tasks and one CPU per task.

```bash
mkdir -p "$receipt_dir"
srun --ntasks=2 --cpus-per-task=1 .venv-native/bin/python \
  scripts/run_native_benchmarks.py --slurm-shard --slurm-allocation \
  --policy fast --output "$receipt_dir/rank-{rank}.json"
```

Set `receipt_dir` to a fresh directory on a filesystem visible to all
tasks. The runner uses `SLURM_PROCID` and `SLURM_STEP_NUM_TASKS` to select
cases. This path has been exercised with simulated step variables.
Live Slurm qualification remains open.

Each receipt records its rank, world size, and selected case IDs.
The runner refuses to overwrite rank receipts. Verify the shard set
against a serial run of the same policy and installed build.

```bash
.venv-native/bin/python scripts/run_native_benchmarks.py --policy fast \
  --output "$receipt_dir/serial.json"
.venv-native/bin/python scripts/check_benchmark_shards.py \
  "$receipt_dir/serial.json" "$receipt_dir"/rank-*.json
```

The verifier checks rank coverage, case order and results, kernel
version, and native extension digest. The worker budget applies per rank. Local runs
qualified both fast and scheduled case selections. Multi-node
execution and scheduler allocation still require qualification.

## Run a native remote worker subprocess

Build the single-request worker with
`cargo build --locked -p innovate-core --bin innovate-remote-worker`.
Pass one remote execution envelope on standard input and close the input.

```bash
target/debug/innovate-remote-worker <<'JSON'
{"schema_version":"2.0","context":{"request_id":"demo","tenant_id":"local","principal":"example","auth_scope":"kernel:execute"},"kernel_request":{"schema_version":"2.0","operation":"predict_model","model_key":"logistic","payload":{"inputs":{"time":[1]},"parameters":{"L":100,"k":1,"x0":1}}}}
JSON
```

The worker exits zero when it emits a JSON response. Check the response's
`error` and the nested `result.kernel_response.error` before consuming a
result. Malformed JSON and raw input over 2,000,000 bytes produce a
diagnostic on standard error and a nonzero exit status. The kernel
also limits the encoded nested request to 1,000,000 bytes.

A transport must authenticate the caller and construct the context
before invoking the worker. Hosted transport and tenant isolation
require separate qualification.

## Comparing Kairos resource use

Add a pair like this to an `execute_scenario_grid` request with two
successful Kairos network scenarios.

```json
{"baseline": "base", "alternative": "closed", "compare_resources": true}
```

The pair must share an observation grid, tick resolution and horizon.
`result.resource_differences` reports alternative minus baseline for
configured, occupied, idle and over-capacity slot-ticks and for admitted,
abandoned and pending wait ticks. It reports differences in peak queue
length and peak concurrent service as agent counts, and differences in completed
services, balked requests and abandoned requests as event counts. It also
compares the numbers waiting and in service at the horizon. The object
also reports the shared
`ticks_per_unit` and `horizon_ticks`. A mismatch returns
`invalid_comparison` for that pair while retaining the scenario executions.

## Exact sign-flip inference for paired scenarios

Add this comparison to the `comparisons` array in an
`execute_scenario_grid` request. Each ID refers to a scenario in the
same grid.

```json
{
  "pairs": [
    {"baseline": "control-1", "alternative": "policy-1"},
    {"baseline": "control-2", "alternative": "policy-2"}
  ],
  "interval_mass": 0.95,
  "exact_sign_flip": true
}
```

Use two to sixteen pairs, with each scenario ID appearing once in the
comparison. The scalar paths must share a time grid and use comparable
units. By default, the test uses alternative minus baseline at the
final sample. Set `"evaluation_time": 1` to evaluate at time 1 when it
appears on the shared grid. The mean, empirical interval, and any
requested bootstrap or sign-flip results all use effects at that
sample. The response reports the actual sample as
`result.evaluation_time`. An off-grid time returns
`invalid_comparison` for that row and preserves the scenario executions.

Every pair must provide two successful scalar paths. If a path is
missing, failed, or paused, the comparison returns
`invalid_comparison` with the scenario ID and its role (baseline or
alternative). The comparison result is null and all individual
execution records remain available. After resolving the failed
execution or resuming the paused scenario, compare the complete
set of pairs.

The comparison's `result.sign_flip` reports the number of assignments
enumerated, the number with an absolute sum at least as large as the
observed absolute sum, and their ratio as the two-sided `p_value`.
For final effects of 50 and 25, two of the four assignments are at least
as extreme, so the p-value is 0.5. The `interval_mass` still controls
the descriptive empirical interval.

Use this test when pairs are independent and the sign of each paired
difference is exchangeable under the null. With paired simulation
seeds, `require_distinct_pair_seeds` can check for matching effective
seeds within pairs and distinct seeds across pairs. The study design
must still justify sign exchangeability.

## Check treatment overlap in cross-fitted contrasts

Use `cross_fitted_stratum_ate` or `cross_fitted_stratum_att` with
outcomes, binary treatment indicators, stratum labels and fold IDs.
For clustered observations, pass cluster labels and assign every
observation in a cluster to the same fold.

Run `.venv-native/bin/python -I examples/native_overlap.py` from the repository root.
The contract checks this displayed source against that executable file.

```python
"""Check cross-fitted overlap and contrasts against six hand-worked observations.

These synthetic outcomes do not identify an empirical treatment effect. Meeting
an overlap count threshold does not establish exchangeability, adequate precision
or a suitable analysis design. Threshold one is deliberately small in this demo.
"""

import json
import math

from innovate import NativeError, execute


def main() -> None:
    """Validate held-out predictions, an independent score oracle and overlap guards."""
    payload = {
        "method": "cross_fitted_stratum_ate",
        "outcomes": [1, 2, 3, 4, 5, 6],
        "treatment": [0, 0, 1, 0, 1, 1],
        "strata": ["a"] * 6,
        "folds": [1, 1, 1, 2, 2, 2],
        "minimum_training_arm_count": 1,
    }
    result = execute("summarize_causal", **payload)["result"]

    # Fold one trains on outcomes 4,5,6; fold two trains on outcomes 1,2,3.
    # Their respective treatment fractions are 2/3 and 1/3.
    expected_propensities = [2 / 3] * 3 + [1 / 3] * 3
    expected_control_means = [4.0] * 3 + [1.5] * 3
    expected_treated_means = [5.5] * 3 + [3.0] * 3
    for actual, expected in (
        (result["propensities"], expected_propensities),
        (result["control_predictions"], expected_control_means),
        (result["treated_predictions"], expected_treated_means),
    ):
        assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(actual, expected, strict=True))
    overlap = result["overlap"]
    assert overlap["min_training_arm_count"] == 1
    assert overlap["min_training_arm_clusters"] is None
    assert math.isclose(overlap["min_propensity"], 1 / 3)
    assert math.isclose(overlap["max_propensity"], 2 / 3)

    # For score mu1-mu0 + T*(Y-mu1)/e - (1-T)*(Y-mu0)/(1-e), the six
    # hand-calculated scores are below. This oracle never calls a model fitter.
    scores = [10.5, 7.5, -2.25, -2.25, 7.5, 10.5]
    expected_ate = sum(scores) / len(scores)
    expected_se = math.sqrt(sum((score - expected_ate) ** 2 for score in scores) / (len(scores) * (len(scores) - 1)))
    assert math.isclose(result["contrast"]["estimate"], expected_ate, rel_tol=1e-12)
    assert math.isclose(result["contrast"]["influence_standard_error"], expected_se, rel_tol=1e-12)
    assert result["contrast"]["observations"] == len(scores)

    # Repeated observations within one arm still count as one distinct cluster.
    clusters = ["fold1-control", "fold1-control", "fold1-treated", "fold2-control", "fold2-treated", "fold2-treated"]
    clustered = execute("summarize_causal", **payload, clusters=clusters, minimum_training_arm_clusters=1)["result"]
    assert clustered["overlap"]["min_training_arm_clusters"] == 1
    assert math.isclose(clustered["contrast"]["estimate"], expected_ate, rel_tol=1e-12)
    assert clustered["contrast"]["cluster_count"] == len(set(clusters))

    invalid_designs = [
        ({**payload, "minimum_training_arm_count": 2}, "minimum_training_arm_count"),
        ({**payload, "clusters": clusters, "minimum_training_arm_clusters": 2}, "minimum_training_arm_clusters"),
        (
            {**payload, "clusters": ["split", "c1", "t1", "split", "t2", "t2"]},
            "single cross-fitting fold",
        ),
    ]
    rejected_designs = 0
    for invalid, expected_message in invalid_designs:
        try:
            execute("summarize_causal", **invalid)
        except NativeError as error:
            assert error.code == "invalid_request" and expected_message in str(error)
            rejected_designs += 1
        else:
            raise AssertionError("An invalid overlap or fold design was silently accepted.")
    assert rejected_designs == len(invalid_designs)

    print(
        json.dumps(
            {
                "ate": round(result["contrast"]["estimate"], 6),
                "minimum_training_arm_count": overlap["min_training_arm_count"],
                "minimum_training_arm_clusters": clustered["overlap"]["min_training_arm_clusters"],
                "propensity_range": [round(overlap["min_propensity"], 6), round(overlap["max_propensity"], 6)],
                "rejected_designs": rejected_designs,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
```

This returns a minimum training-arm count of one and held-out
propensities ranging from one third to two thirds. The default minimum
is one. Raising it to two rejects this request because a held-out
stratum relies on a training arm with one observation. The guard
rejects the full estimate without dropping any strata. Choose the
minimum as part of the analysis design. A passing guard establishes
only that the requested count was met.

With cluster labels, `min_training_arm_clusters` reports the smallest
number of distinct clusters in a training arm. Use
`minimum_training_arm_clusters` to require a positive minimum. Multiple
rows from the same cluster count once per arm. Both minimums apply when
supplied together.

## Counterfactual forecasts with parameter events

Use the `counterfactual_forecast` method of `summarize_causal` to
compare forecasts on a shared prediction grid. The baseline accepts
the same payload as native prediction, including a saved fitted state
or explicit parameters and constructor options.

This payload adds a Lotka–Volterra parameter event at time 1 and
compares both products at time 2.

```json
{
  "method": "counterfactual_forecast",
  "forecast_model": "lotka_volterra",
  "baseline": {
    "inputs": {"time": [2.0]},
    "parameters": {"alpha1": 0.2, "beta1": 0.0, "alpha2": 0.1, "beta2": 0.0},
    "constructor_kwargs": {"initial_values": [0.1, 0.2]}
  },
  "parameter_event_override": [{"time": 1.0, "parameters": {"alpha1": 0.4}}]
}
```

The event override replaces the whole schedule on the alternative.
Use an empty array to remove events. Use `parameter_overrides` for
named model parameters, `covariate_overrides` for named values on the
prediction grid, and `covariate_history_overrides` for named values
on the existing constructor history grid.

The result contains both trajectories and the alternative minus the
baseline. Multi-product results include column labels and row matrices.
Percentage differences are null where the baseline is zero.

## Re-ingesting a dataset artifact

Pass the complete artifact so Rust can verify the checksum with its
recorded encoding and recompute dataset diagnostics.

Run `.venv-native/bin/python -I examples/native_reingestion.py` from the repository root.
The contract checks this displayed source against that executable file.

```python
"""Verify both dataset encodings, complete-artifact replay and tamper rejection."""

import json
from copy import deepcopy

from innovate import NativeError, execute


def main() -> None:
    """Re-ingest a synthetic adoption dataset while preserving its checksum format."""
    dataset = {
        "kind": "adoption",
        "time": [0, 1],
        "adoption": [1, 2],
        "provenance": {
            "source": "synthetic survey α",
            "license": "CC0-1.0",
            "extraction_time": "2026-11-15T00:00:00Z",
        },
    }
    encodings = ["python_sorted_json_v1", "rust_sorted_json_v1"]
    rejected_count = 0
    for encoding in encodings:
        artifact = execute("ingest_dataset", dataset=dataset, encoding=encoding)["result"]
        assert artifact["checksum_encoding"] == encoding
        verified = execute("ingest_dataset", artifact=artifact)["result"]
        assert verified == artifact
        assert verified["dataset"]["adoption"] == [1, 2]
        assert verified["dataset"]["time"] == [0, 1]
        tampered = deepcopy(artifact)
        tampered["dataset"]["adoption"][1] = 3
        rejected = False
        try:
            execute("ingest_dataset", artifact=tampered)
        except NativeError as error:
            rejected = "checksum does not match" in str(error)
        assert rejected
        rejected_count += int(rejected)
    assert dataset["adoption"] == [1, 2]
    print(json.dumps({"encodings": encodings, "tampered_artifacts_rejected": rejected_count}, sort_keys=True))


if __name__ == "__main__":
    main()
```

This works for artifacts recorded with either `python_sorted_json_v1`
or `rust_sorted_json_v1`. For a bare dataset, pass `dataset` and an optional
`encoding`. The default for a bare dataset is `rust_sorted_json_v1`.

For datasets without a checksum, ingestion assigns omitted policy event
labels as `event_0`, `event_1`, and so on, and fills omitted network edge
weights with `1.0` before calculating the digest. For checksummed inputs,
preserve the generated fields and their JSON numeric types. Removing a
field causes checksum verification to fail.

### Historical interaction-matrix competition

Call `predict_competition_interaction_matrix` through the JSON envelope with
`time`, `p`, `q`, and `market_potentials`. Supply 2 to 64 products and a
square signed interaction matrix `q`. Row `i`, column `j` gives the
coefficient of product `j`'s adoption share in product `i`'s rate.
The prediction starts at zero adoption at the first supplied time and returns
`rows` and instantaneous `rates` in time-by-product order. Optional
`observed` values in the same shape and product order add a pooled
`r_squared` across all times and products. Constant observations score zero.

`competition_cross_elasticity` accepts the same inputs and preserves the
historical first-product seed of `1e-6`. It returns the historical
absolute-coefficient proxy in product-by-time order.

`fit_competition_interaction_matrix` uses the same payload with required
`observed` rows and treats `p`, `q`, and `market_potentials` as initial
parameter guesses. It returns fitted parameters, predictions, solver diagnostics,
and local identifiability. Fitting supports 2 to 8 products with at least
`n + 2` time points. Supply all three guesses for a single start, or omit them to try three
capacity scales with observation-derived innovation rates. The solver reports
the termination status and local identifiability of the selected fit.

Historical Norton–Bass summaries retain requested-sample milestones and expose
`refined_crossings` for the same fixed baseline market fractions. The search
adds 128 uniform intervals across the requested window, plus event times and
covariate knots, then bisects the first detected upward bracket. Excursions
shorter than the probe spacing can still be missed.

`refined_rate_peaks` reports the best signed rate found on the probe grid and
in the two adjacent intervals. Event and launch boundaries are evaluated from
both sides where they fall inside the window. The rate excludes instantaneous
jump effects. The search can miss narrow peaks outside the selected intervals.
