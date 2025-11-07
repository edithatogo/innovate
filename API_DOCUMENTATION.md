# Innovate Library API Documentation

## Overview
The Innovate library provides tools for modeling innovation and policy diffusion, with implementations of classic diffusion models and advanced fitting techniques.

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
- **MultiProductDiffusionModel** (`innovate.compete.competition.MultiProductDiffusionModel`): Multi-product diffusion with competition effects

### 5. Backend Management (`innovate.backend`)
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
from innovate.diffuse.bass import BassModel
from innovate.fitters.scipy_fitter import ScipyFitter

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
from innovate.backend import use_backend

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