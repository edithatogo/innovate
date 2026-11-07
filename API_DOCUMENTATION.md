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
