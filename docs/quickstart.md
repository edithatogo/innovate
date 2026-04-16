# Quickstart Guide

This guide will help you get started with the Innovate library quickly.

## Basic Usage

The Innovate library follows a familiar API pattern similar to scikit-learn. Here's how to use it:

```python
from innovate import BassModel, ScipyFitter

# Your time and adoption data
t = [0, 1, 2, 3, 4, 5]  # time points
y = [10, 20, 35, 60, 80, 90]  # adoption values

# Create model and fitter
model = BassModel()
fitter = ScipyFitter()

# Fit the model
fitted_model = model.fit(fitter, t, y)

# View fitted parameters
print("Fitted parameters:", fitted_model.params_)

# Make predictions
future_times = [6, 7, 8]
predictions = fitted_model.predict(future_times)
print("Predictions:", predictions)

# Evaluate model performance
score = fitted_model.score(t, y)
print("Model R² score:", score)
```

## Using Different Models

The library includes several diffusion models:

```python
from innovate import GompertzModel, LogisticModel

# Logistic model
logistic_model = LogisticModel()
# ... same fitting process as above

# Gompertz model
gompertz_model = GompertzModel()
# ... same fitting process as above
```

## Adding Covariates

You can include external variables (covariates) that affect model parameters:

```python
from innovate import BassModel, ScipyFitter

# Model with covariates (e.g., marketing spend, price)
model = BassModel(covariates=["marketing_spend", "price"])

# Your data with covariates
t_data = [0, 1, 2, 3, 4]
y_data = [5, 15, 30, 50, 70]
covariates_data = {
    "marketing_spend": [100, 150, 200, 180, 160],
    "price": [10, 9.5, 9, 8.5, 8]
}

# Fit the model with covariates
fitter = ScipyFitter()
fitted_model = model.fit(fitter, t_data, y_data, covariates=covariates_data)

# Make predictions with covariates
future_covariates = {
    "marketing_spend": [170, 180, 190],
    "price": [8, 7.5, 7]
}
future_t = [5, 6, 7]
predictions = fitted_model.predict(future_t, covariates=future_covariates)
```

## Using Different Backends

The library supports both NumPy and JAX for computations:

```python
from innovate.backends import use_backend

# Switch to JAX backend for GPU acceleration (if available)
use_backend('jax')

# Switch back to NumPy
use_backend('numpy')
```

Install `innovate[jax]` when you want the accelerator backend, and
`innovate[bayesian]` when you want the Bayesian/BlackJAX path. The base
install remains NumPy/SciPy-only for portability.

## Model Comparison

Compare different models to find the best fit for your data:

```python
from innovate import BassModel, GompertzModel, LogisticModel, ScipyFitter

models = {
    "Bass": BassModel(),
    "Logistic": LogisticModel(),
    "Gompertz": GompertzModel()
}

fitter = ScipyFitter
t_data = [0, 1, 2, 3, 4, 5]
y_data = [10, 25, 45, 70, 90, 95]

model_scores = {}
for name, model in models.items():
    fitted_model = model.fit(fitter, t_data, y_data)
    score = fitted_model.score(t_data, y_data)
    model_scores[name] = score
    print(f"{name} model R²: {score:.3f}")

# Find the best model
best_model_name = max(model_scores, key=model_scores.get)
print(f"Best model: {best_model_name} with R² = {model_scores[best_model_name]:.3f}")
```

This quickstart guide covers the basic functionality of the Innovate library. For more advanced features, see the API documentation or tutorials.

For compatibility, older imports like `innovate.backend` and deep module imports such as `innovate.diffuse.bass` continue to work. New examples should prefer the package-level imports shown above.
