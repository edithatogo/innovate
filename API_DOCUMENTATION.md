> Historical pre-0.7 API reference. These Python APIs are temporarily unavailable.
> The current Rust/Polars API is documented in README.md and
> docs/astro-site/src/content/docs/native-migration.md. Requirements remain in
> migration-ledger.json; this document is not a current support claim.

# Innovate Library API Documentation

## Overview
The Innovate library provides tools for modeling innovation and policy diffusion, with implementations of classic diffusion models and advanced fitting techniques.

## Canonical Public API

For new user code, prefer the stable package-level imports:

```python
from innovate import BassModel, GompertzModel, LogisticModel, ScipyFitter
from innovate.compete import MultiProductDiffusionModel, LotkaVolterraModel
from innovate.substitute import CompositeDiffusionModel, FisherPryModel, NortonBassModel
from innovate.ecosystem import ComplementaryGoodsModel
from innovate.backends import use_backend
```

Legacy imports such as `innovate.backend` and `innovate.compete.competition` remain importable for compatibility, but the package-level imports above define the intended public API surface.

## Core Modules

### 1. Diffusion Models (`innovate.diffuse`)
- **Bass Model** (`innovate.diffuse.bass.BassModel`): Implementation of the classic Bass diffusion model
- **Logistic Model** (`innovate.diffuse.logistic.LogisticModel`): Logistic growth model
- **Gompertz Model** (`innovate.diffuse.gompertz.GompertzModel`): Gompertz growth model

#### Bass Model
The Bass model is defined by the differential equation:
```math
\frac{dF(t)}{dt} = (p + q \cdot F(t)) \cdot (m - F(t))
```
Where:
- `p` is the coefficient of innovation
- `q` is the coefficient of imitation
- `m` is the market potential

**Key Methods:**
- `initial_guesses(t, y)`: Provides initial parameter guesses
- `bounds(t, y)`: Defines parameter bounds
- `predict(t, covariates=None)`: Predicts cumulative adoption
- `score(t, y, covariates=None)`: Calculates R² score
- `params_`: Property for accessing/modifying model parameters

### 2. Base Classes (`innovate.base`)
- **DiffusionModel** (`innovate.base.base.DiffusionModel`): Abstract base class for diffusion models

### 3. Fitting Algorithms (`innovate.fitters`)
- **ScipyFitter** (`innovate.fitters.scipy_fitter.ScipyFitter`): Fitting using scipy optimizers
- **BayesianFitter** (`innovate.fitters.bayesian_fitter.BayesianFitter`): Bayesian parameter estimation
- **BlackJaxFitter** (`innovate.fitters.blackjax_fitter.BlackjaxFitter`): JAX-based MCMC fitting

### 4. Competition Models (`innovate.compete`)
- **MultiProductDiffusionModel** (`innovate.compete.MultiProductDiffusionModel`): Stable matrix-form multi-product diffusion API

### 5. Backend Management (`innovate.backends`)
- **Backend switching**: Support for NumPy and JAX backends
- **`use_backend(name)`**: Switch between computational backends

### 6. Dynamics Models (`innovate.dynamics`)
- **Growth Models**: Various growth curve implementations
- **Competition Models**: `innovate.dynamics.competition` - Lotka-Volterra, Market Share Attraction, Replicator Dynamics
- **Contagion Models**: `innovate.dynamics.contagion` - SIR, SIS, SEIR models for contagion dynamics

#### Competition Dynamics
- **LotkaVolterraCompetition** (`innovate.dynamics.competition.lotka_volterra.LotkaVolterraCompetition`): Models competition between two species using Lotka-Volterra equations
- **MarketShareAttraction** (`innovate.dynamics.competition.market_share_attraction.MarketShareAttraction`): Determines market share based on relative attractiveness
- **ReplicatorDynamics** (`innovate.dynamics.competition.replicator_dynamics.ReplicatorDynamics`): Models evolution of strategy proportions based on relative fitness

#### Contagion Dynamics
- **SIRModel** (`innovate.dynamics.contagion.sir.SIRModel`): Susceptible-Infected-Recovered model
- **SISModel** (`innovate.dynamics.contagion.sis.SISModel`): Susceptible-Infected-Susceptible model  
- **SEIRModel** (`innovate.dynamics.contagion.seir.SEIRModel`): Susceptible-Exposed-Infected-Recovered model

**Common Methods for Dynamics Models:**
- `compute_spread_rate(**params)` or `compute_interaction_rates(**params)`: Calculate instantaneous rates
- `predict_states(time_points, **params)`: Predict states at specified time points
- `get_parameters_schema()`: Get parameters schema

## Common Usage Patterns

### Basic Model Fitting
```python
from innovate import BassModel, ScipyFitter

# Create model and fitter
model = BassModel()
fitter = ScipyFitter()

# Fit to data
t_data = [0, 1, 2, 3, 4, 5]  # time points
y_data = [10, 25, 45, 70, 90, 95]  # adoption values

# Perform fitting
fitted_model = model.fit(fitter, t_data, y_data)

# Make predictions
predictions = fitted_model.predict([6, 7, 8])
```

### Model with Covariates
```python
# Create model with covariates
model = BassModel(covariates=["marketing_spend", "price"])

# The model will automatically include parameters for covariate effects
# like beta_p_marketing_spend, beta_q_marketing_spend, etc.
```

### Backend Switching
```python
from innovate.backends import use_backend

# Switch to JAX backend for GPU acceleration
use_backend('jax')

# Switch back to NumPy
use_backend('numpy')
```

## Key Features

1. **Flexible Parameterization**: Models support covariates and structural breaks
2. **Multiple Backends**: NumPy and JAX support with automatic differentiation
3. **Advanced Fitting**: Classical, Bayesian, and MCMC fitting methods
4. **Competition Modeling**: Multi-product models with competitive effects
5. **Performance Optimized**: Designed for computational efficiency

## Error Handling

- **Unfitted Model Errors**: Calling `predict()` on unfitted models raises `RuntimeError`
- **Parameter Validation**: Automatic parameter bounds checking
- **Numerical Stability**: Built-in checks for numerical issues

## Configuration and Settings

### Backend Configuration
```python
# Use NumPy backend (default)
from innovate.backend import use_backend
use_backend('numpy')

# Use JAX backend (for advanced features)
use_backend('jax')
```

## Mathematical Background

The library implements diffusion processes based on ordinary differential equations:

- **Bass Model**: `(dN/dt) = p(M-N) + q(M-N)(N/M)` where N is cumulative adoptions, M is market size
- **Logistic Model**: `(dP/dt) = rP(1-P/K)` where P is population, r is growth rate, K is carrying capacity
- **Gompertz Model**: `(dP/dt) = rP*ln(K/P)` with similar parameters as logistic

## Performance Considerations

- Use JAX backend for maximum performance and GPU acceleration
- Consider using batched fitting for multiple time series
- For large datasets, consider using the Bayesian fitters
- Memory usage grows with the number of parameters and time points

## Extending the Library

To create a new diffusion model, inherit from `DiffusionModel` and implement the required abstract methods:

```python
from innovate.base.base import DiffusionModel

class NewModel(DiffusionModel):
    def predict(self, t):
        # Implement prediction logic
        pass
    
    def score(self, t, y):
        # Implement scoring logic
        pass
    
    @property
    def params_(self):
        # Implement params property
        pass
    
    @params_.setter
    def params_(self, value):
        # Implement params setter
        pass
```

## Stratified treatment contrast

Call `summarize_causal` with method `stratified_treatment_contrast` and
aligned arrays `outcomes`, `treatment` (0 or 1), and `strata` (nonblank
string labels). The point estimate weights each within-stratum mean
difference by its share of the supplied sample. The response also
reports each stratum’s weight and arm counts. Every stratum requires
both arms. Strata must be supplied by the caller. Causal interpretation
requires exchangeability within strata, positivity and consistency.

Set `estimand` to `att` to weight strata by their share of treated
observations. Control-only strata are omitted, and each stratum with
treated observations requires controls. The default `ate` uses the full
sample as described above.

The `conditional_standard_error` assumes independent observations and
holds the stratum weights fixed. It combines the two arm sample variances
divided by their respective arm sizes, with squared stratum weights.
It is null when an included arm has only one observation.

Call `summarize_causal` with method `describe_treatment_contrasts` to
discover the supported adjustment methods, inputs and estimands.

Each included stratum reports its observed `treated_fraction`. The
top-level `excluded_control_observations` counts controls in strata
outside the ATT target population.

## Propensity weighted contrast

Call `summarize_causal` with method `propensity_weighted_contrast` and
aligned arrays of outcomes, binary treatment indicators and caller-supplied
treatment propensities strictly between zero and one. The operation
normalizes inverse probability weights separately in each arm and returns
their weighted means and difference. This ATE point estimate requires
valid propensities, exchangeability, positivity and consistency.

The `weight_diagnostics` object reports each arm’s observation count,
largest normalized weight and effective sample size. Effective sample
size is the reciprocal of the sum of squared normalized weights.

The default `estimand` is `ate`. Set `estimand` to `att` to give
treated observations equal weight and weight controls by their treatment
odds, p/(1−p), before normalizing within each arm.

The `fixed_propensity_standard_error` is a sandwich standard error
for independent observations and a fixed propensity function. It
combines squared weighted residuals with an arm-specific n/(n−1)
correction. It is null when either arm has fewer than two observations.
Uncertainty from fitting the propensity function requires separate inference.

Call `summarize_causal` with method `estimate_stratum_propensities`
and aligned `treatment` and `strata` arrays to estimate treatment
probabilities from observed arm frequencies. It returns probabilities
in input order and arm counts by stratum. Every stratum requires both
arms. When those fitted probabilities feed the weighted contrast, its
fixed-propensity standard error does not account for the fitting step.

## Matched treatment contrast

Call `summarize_causal` with method `matched_treatment_contrast`,
aligned `outcomes`, binary `treatment`, scalar `scores`, and a
nonnegative `caliper` in score units. The method matches each treated
observation to the nearest control with replacement. Ties use the
earliest control in input order. It returns the mean paired difference
and zero-based pair indices. A treated observation outside the caliper
causes the request to fail. Identification depends on the supplied score
and the assumptions of the study design.

The `control_diagnostics` object reports the number of unique matched
controls and their effective sample size. A control’s normalized weight
is its share of matched pairs. Effective sample size is the reciprocal
of the sum of squared control weights.

Set `allow_partial` to true to estimate the contrast over treated
observations that match within the caliper. If any treated observations
are dropped, the result labels the estimand `matched_subset_att` and
lists their zero-based indices in `unmatched_treated_indices`. At least
one match is required.

Supply aligned, nonnegative `outcome_variances` to obtain a
conditional standard error for independent outcomes with the matched
pairs held fixed. Each reused control contributes its variance times
the square of its total weight. Without supplied variances the field
is null. This calculation conditions on the matching design.

For numeric covariates, call `summarize_causal` with method
`fit_logistic_propensities`, binary `treatment`, a row matrix
`covariates`, and positive `ridge`. The fit minimizes mean logistic
loss plus ridge times half the squared slope norm after scaling each
column by its maximum absolute value. The intercept is unpenalized.
The result contains fitted probabilities in input order, the intercept
followed by scaled slopes, column scales and the iteration count.
A fit that fails to converge returns an error.

Call `summarize_causal` with method `predict_logistic_propensities`,
the saved `fit` result and a new `covariates` row matrix. Prediction
uses the original column scales and coefficients. Columns must follow
the training order. Probabilities are returned in prediction-row order.

Use `score_propensities` with held-out binary `treatment` and aligned
`propensities` in (0, 1). It reports mean Brier score, mean log loss,
the observed treatment fraction and the mean predicted probability.

Use `cross_fit_logistic_propensities` with the fitting inputs and
aligned unsigned integer `folds` labels. Each of the two to 32 folds
is scored using coefficients and scales fitted on its complement.
The result preserves row order. Every training complement requires
both treatment arms. The caller supplies the split design.

## Augmented treatment contrast

Use `augmented_treatment_contrast` with aligned outcomes, binary
treatment, propensities in (0, 1), and treated and control predictions
for every observation. The ATE averages the predicted differences
plus inverse-probability-weighted observed-arm residual corrections.
For independent rows with suitably estimated nuisance predictions,
the influence standard error is the sample standard deviation of
these scores divided by the square root of the sample size. Supply
out-of-fold predictions when fitting nuisance models on the same data.

For `estimand="att"`, provide control predictions only. The estimate
subtracts treatment-odds-weighted control residuals from treated
residuals, with both totals divided by the treated count. The influence
standard error accounts for estimating the treated fraction.

Use `cross_fitted_stratum_ate` with aligned `outcomes`, binary
`treatment`, string `strata` and unsigned integer `folds`. For each
held-out stratum, the training complement must contain both arms.
Rust estimates the treatment frequency and both outcome means in
that complement, then computes augmented ATE. The response includes
the contrast, folds and nuisance predictions in original row order.

Use `cross_fitted_stratum_att` with the same inputs to target the
treated population. It uses the same training-complement overlap
rule and computes the augmented ATT from control predictions and
propensities.

Supply aligned string `clusters` to use independent-cluster inference
for the augmented contrast. The standard error sums influence scores
within clusters and applies the G/(G−1) variance correction for G
clusters. At least two clusters are required. The estimate is unchanged.

The cross-fitted stratum methods also accept `clusters`. Every
cluster must belong to one fold so its outcomes stay together during
training and validation. The augmented contrast then uses the same
cluster labels for its standard error.

## Balanced-panel event study

Use `panel_event_study` with strictly increasing `periods`, a zero-based
`baseline_index`, and `treated` and `control` matrices with one row per
unit and one column per period. Each contrast subtracts the control
mean change from the treated mean change relative to the baseline.
Standard errors use independent units and are null for singleton arms.
These are pointwise estimates for a common treatment timing design.
Causal interpretation requires parallel trends and no anticipation.

Use `staggered_panel_att` with `periods`, a unit-by-period `outcomes`
matrix, and zero-based `adoption_indices`. Use null for never-treated
units. Each cohort uses its last pre-adoption observation as baseline
and compares changes with units untreated through the target period.
Rows with no eligible controls return a null estimate and zero controls.
Interpretation requires parallel trends and no anticipation.

Staggered cohort-period rows report pointwise standard errors for
independent units. The standard error is null when either arm has
fewer than two units. Each row uses its own eligible control set.

The `event_time` summary groups cohort estimates by the number of
observed period steps since adoption. It weights supported cohorts
by their treated counts and reports how many cohorts lack controls.
A step with no supported cohorts has a null estimate.

Each event-time row also reports a pointwise standard error for
independent units and fixed cohort weights. Contributions from the
same unit are summed before squaring, including when a unit appears
in multiple comparison sets. The standard error is null if any
contributing cohort has a singleton arm.

Set `include_event_time_covariance` to true to return the covariance
matrix in event-time row order, for up to 128 offsets. Its diagonal
contains squared pointwise standard errors. Entries involving an
offset with unavailable uncertainty are null.

Request `simultaneous_bands` with an integer `seed`, 100 to 10,000
`draws` and optional `confidence` (default 0.95). The multiplier
max-t procedure uses the same random sign for each unit across offsets
and returns bands for offsets with positive standard errors. Other
offsets have null intervals. These are approximate bands under
independent units and fixed cohort weights.

Set `balanced_horizon` to a nonnegative event-time offset to keep the
same cohort composition from adoption through that offset. The summary
includes only cohorts observed with eligible controls through the horizon.
The result reports included and excluded cohort counts. Cohort-period
contrasts remain available for the full panel.

Set `include_pre_trends` to true for cohort-specific placebo contrasts
before the reference period. These compare each earlier outcome with
the last pre-adoption observation, using controls that remain untreated
through that reference period. The separate `pre_trends` array carries
pointwise errors and does not enter the post-adoption summary.

Set `control_group` to `never_treated` to use only units with null
adoption indices for all contrasts and uncertainty calculations.
The default `not_yet_treated` also uses units that adopt after the
comparison period.

Use `staggered_long_panel_att` for aligned `unit_ids`, `time` and
`outcomes` columns and an `adoption_times` map keyed by unit label.
Adoption dates must identify observed periods, with null for never
treated. The operation sorts periods, rejects duplicate or missing
unit-period rows, and returns the sorted `unit_order`. It accepts the
same inference options as the matrix operation.

Python callers can pass a Polars DataFrame or LazyFrame to
`staggered_att_polars(frame, adoption_times, **options)`. The frame
must contain `unit_ids`, `time` and `outcomes`. The function returns
the native staggered result, including cohort and event-time summaries.

## Scenario grid ranking

Supply `rank_scenarios` to `execute_scenario_grid` with at least two
distinct scenario IDs. Scalar paths must share the same time grid and
time basis. The result ranks final adoption in descending order.
Ties share a place and retain request order. A ranking error leaves
the scenario executions and pair comparisons available.

Set `rank_metric` to `adoption_area` to rank by the trapezoidal area
under each adoption path over the shared observation horizon. The
score is measured in adoption units times time units. This metric
requires at least two samples. The default is `final_adoption`.

Set `rank_metric` to `time_to_threshold` and supply a numeric
`rank_threshold` to rank earlier crossings first. Crossing times use
linear interpolation between adjacent samples. Paths that never
reach the threshold are listed last with null times and places.

Set `evaluation_metric="adoption_area"` on a paired ensemble
comparison to integrate each alternative-minus-baseline path over
the common horizon before computing the empirical summary. The
response includes `evaluation_horizon`. This metric and
`evaluation_time` are mutually exclusive.

Set `include_path_summary` on a paired ensemble comparison to obtain
the mean difference, standard error and empirical interval at each
shared sample time. These are pointwise summaries. The request is
limited to one million pair-by-time samples.

Set `include_path_covariance` to return the covariance matrix of the
paired mean differences and the associated path summary. The matrix
follows sample order and uses the same independent-pair assumption as
the pointwise standard errors. It supports up to 128 sample times.

Both matrix axes use the times in `path_summary.rows`. For `n`
replicate pairs, entry `[i][j]` is the sample covariance of the paired
effects at times `i` and `j`, divided by `n`. Each diagonal entry
therefore equals the square of that time's `mean_standard_error` up
to floating-point rounding. The empirical intervals describe the
observed paired effects at each time. Simultaneous
confidence bands for the mean path require a separate inference procedure.

If a covariance request exceeds 128 samples, its comparison row
returns `invalid_comparison`. For a longer path, request
`include_path_summary` alone, subject to the pair-by-time limit.

For inference on the selected paired effect, supply `bootstrap` with
`draws` and an unsigned integer `seed`. The Rust engine resamples
whole pair differences with replacement and returns the percentile
interval for the mean in `bootstrap.mean_interval`. It uses the
comparison's `interval_mass`, assumes independent replicate pairs,
and allows 2 to 10,000 draws with at most ten million sampled pairs.

Set `exact_sign_flip` to enumerate the sign assignments of up to
16 pair differences. The two-sided `sign_flip.p_value` is the
fraction of assignments whose absolute mean is at least the observed
absolute mean. The test assumes independent pairs and exchangeable
signs under the null.

### Kairos Bass adoption checkpoints

Set `simulation_options.pause_after_adoptions` to the cumulative number
of new adoptions at which to pause. If that event occurs within the
requested horizon, the result contains `status: "paused"` and a
`checkpoint` with the event history and Kairos RNG position. If the
horizon is reached first, the result is the ordinary simulation output.

Pass the returned checkpoint as `simulation_options.resume_checkpoint`
with the same effective model configuration and tick horizon. Omit the
pause option to finish, or supply a higher cumulative count to pause
again. The completed result includes the original adoption history.
Keep checkpoints as trusted artifacts. Version 1 supports the
`kairos_des` and homogeneous `kairos_abm` engines at dispatched adoption
boundaries. ABM checkpoints also retain the ordered susceptible agents
and identity RNG state.
Event ticks in the checkpoint are decimal strings to preserve their
full integer precision across JSON consumers.

For `kairos_network`, use `pause_after_completions` to pause after a
cumulative number of service completions and any resulting admissions.
Keep the returned checkpoint string intact and supply it as
`resume_checkpoint` with the original network request and time grid.
Omit the pause option to finish. The checkpoint preserves pending
service, waiting deadlines, retry state and resource accounting.

For `kairos_des` or `kairos_abm`, set `pause_at_tick` to a
nonnegative integer below the final tick horizon. The checkpoint
includes decisions through that tick and can be returned before the
first adoption. Use either the tick cutoff or the adoption count per
request. To pause again by tick, supply a cutoff later than the one
in the saved checkpoint.

The network engine also accepts `pause_at_tick` below its horizon.
It processes events through that tick and saves the state needed to
reproduce the next hazard draw. Use either the tick cutoff or the
completion count per request.

In the Arrow scenario execution export, `successful` denotes a
completed execution without an error. A paused row retains its
checkpoint and status in `execution_json`.
For exact R Arrow export of execution rows, obtain the response with
`kernel_call(operation, payload = payload, raw = TRUE)` and pass that JSON to
`kernel_scenario_executions_ipc`. This preserves the native JSON arrays
and numeric representations through export.

For `matched_treatment_contrast`, supply `balance_covariates` as a map
of covariate names to numeric arrays aligned with the input outcomes.
The response reports the treated and control means and their difference
for each covariate over retained pairs. A reused control contributes
once per retained pair.
Each covariate also includes `before_matching` means and their
difference over the original treatment and control groups.
Both standardized mean differences divide the signed difference by
the same pooled pre-match sample standard deviation. The standardized
values are null when either original group has fewer than two rows,
the pooled spread is zero, or the result exceeds the finite range.

Optional `control_predictions` supplies one control outcome prediction
per input row. The `adjusted_estimate` averages the observed paired
differences minus the corresponding prediction differences over the
same retained pairs. The conditional standard error treats the
predictions and matches as fixed.
