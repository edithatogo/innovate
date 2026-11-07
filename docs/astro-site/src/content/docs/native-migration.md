---
title: Native Rust and Polars migration
description: Current development API, scientific conventions and remaining obligations
---

Innovate 0.7 is a breaking development line. Rust owns computation and scientific
validation. Python, Julia and R are the supported bindings; Polars owns tables.
The installed Python package requires Polars and its native extension. There is
no Python numerical fallback, pandas or xarray runtime.

## Current scope

Eleven native model families expose deterministic fitting, prediction, simulation,
summary and diagnostics: Bass, Logistic, Gompertz, Fisher–Pry, Norton–Bass,
Lotka–Volterra, multi-product competition, complementary goods and lock-in. Their support remains bounded.
Covariates, events, uncertainty, other model families and full Kairos simulation
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

## Native Kairos Bass simulation

`simulate_model` accepts an explicit `simulation_options` object for Bass:

```python
from innovate import execute
import polars as pl

result = execute("simulate_model", "bass",
    inputs={"time": [0.0, 1.0, 5.0, 20.0]},
    parameters={"p": 0.2, "q": 0.3, "m": 100.0},
    simulation_options={"engine": "kairos_des", "seed": 42, "population": 100})
trajectory = pl.DataFrame({"time": [0.0, 1.0, 5.0, 20.0],
                           "adopters": result["result"]["counts"],
                           "level": result["result"]["values"]})
```

Kairos revision `d56863505d938feb069af88a737eb7668113d14b` owns scheduler
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

```python
simulation_options = {
    "engine": "kairos_abm", "seed": 42, "population": 100,
    "covariate": {"time": [0.0, 3.0], "values": [-1.0, 0.0],
                  "beta_p": 0.2, "beta_q": 0.3},
    "events": [{"time": 6.0, "delta_p": 0.1, "delta_q": 0.0}],
}
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

`t_event` selects independent pre/post curves using `L_post`, `k_post`,
`x0_post` at and after the event. These historical curves can jump at the event;
continuity is not imposed. If covariates and an event are combined,
`event_covariates="historical_ignored"` preserves the original behavior: the
event branch ignores covariates. Metadata records that mode; fit diagnostics warn
that its coefficients are unidentifiable. Select `event_covariates="additive"`
to apply the same covariate coefficients to both segments in the redesigned API.

Advanced fits currently require complete `fit_options.initial_parameters`,
including post-event and covariate coefficients. Rust reports convergence and
local rank; rank deficiency does not establish parameter uniqueness. Default
initialization, analytic time-varying adoption-rate helpers and the other models'
historical interpolation/event semantics remain required migration work. This
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

```python
simulation_options = {
    "engine": "kairos_network", "seed": 42, "population": 3,
    "edges": [[0, 1], [1, 2]], "initial_agents": [0],
    "capacity": 1, "service_ticks": 5, "ticks_per_unit": 1000000,
}
```

Edges are unique undirected pairs of node positions, without self-loops. Agent
records retain node order and Kairos identities. Population is 1–1000; capacity
is 1–population. Initial agent indices are unique and already adopted. A
susceptible node requests service at hazard `p + q*adopted_neighbors/degree`;
isolated nodes have zero imitation. Native simulation now permits p=0, allowing
pure contagion seeded by initial agents. Deterministic Bass fitting/prediction
retains its previous positive-p domain.

Kairos ABMContext owns entity state and dispatch; its Resource owns the FIFO
queue and leases. An agent becomes adopted at service completion. Pending agents
do not transmit and cannot request twice. Completion events precede tied requests.
Waiting times round up to at least one tick. Service ticks must be positive and
at most 10^12; ticks per unit are 1–10^9, with a default of 10^6. Requested sample
times must map into 0–10^15 ticks. Counts include initial agents. Returned
`requests` and `completions` retain ticks/identities; agent and resource state is
at the final sample. RNG checkpoint metadata is provenance, not resume support.

Optional population-length `innovation_multipliers` and `imitation_multipliers`
scale each susceptible agent's p and q terms. They default to one and must be
finite and nonnegative; zero disables that hazard component. Optional
`agent_service_ticks` overrides the scalar duration per agent, including agents
waiting in the FIFO queue. Each duration is an integer in 1–10^12. Defaults retain
the homogeneous engine and RNG stream. Metadata retains the effective arrays.

The network engine currently rejects covariate/event forcing, weighted/directed
edges and arbitrary hazard functions. Dynamic networks, richer agent attributes,
resource priorities/balking/reneging, resume and general network-model parity
remain required migration work. A successful Bass network simulation does not
qualify the separate `network_diffusion` model family. Python, Julia and R use
the same versioned native request and retain the same service/adoption semantics.

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
and calibrated uncertainty require separate qualification. Covariates/events
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
