---
title: Prediction & Forecasting
description: Generate forecasts and predictions from fitted diffusion models.
slug: latest/user-guide/forecasting
---

Once a model is fitted, Innovate provides several functions for generating predictions and forecasts.

## Basic Forecasting

The `predict_model` function generates forecasts from a fitted model:

```python
from innovate import fit_model, predict_model

result = fit_model(data, model='bass')
forecast = predict_model(result, horizon=12)

print(forecast)
```

### Specifying the Horizon

The `horizon` parameter controls how many periods to forecast:

```python
# 6 periods
forecast_6 = predict_model(result, horizon=6)

# 24 periods
forecast_24 = predict_model(result, horizon=24)
```

### Confidence Intervals

Generate forecasts with confidence intervals:

```python
forecast = predict_model(
    result,
    horizon=12,
    confidence_level=0.95,  # 95% confidence intervals
    n_simulations=1000      # Bootstrap simulations
)

print(forecast.confidence_lower)
print(forecast.confidence_upper)
```

## Prediction Types

### Cumulative Adoptions

Default. Returns the cumulative number of adopters at each time period.

```python
forecast = predict_model(result, horizon=12, type='cumulative')
```

### New Adoptions (Per Period)

Returns the number of new adopters in each period.

```python
forecast = predict_model(result, horizon=12, type='new')
```

### Adoption Rate

Returns the instantaneous adoption rate at each time point.

```python
forecast = predict_model(result, horizon=12, type='rate')
```

## Forecasting with Covariates

Include external covariates in the forecast:

```python
covariates = pd.DataFrame({
    'time': range(13, 25),
    'marketing_spend': [50000, 52000, 48000, 55000, 60000, 58000, 62000, 65000, 70000, 68000, 72000, 75000],
    'competitor_entries': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
})

forecast = predict_model(result, horizon=12, covariates=covariates)
```

## Scenario Analysis

Run what-if scenarios by modifying forecast parameters:

```python
# Baseline forecast
baseline = predict_model(result, horizon=12)

# Optimistic scenario: higher imitation coefficient
optimistic = predict_model(
    result._replace(params={'q': result.params['q'] * 1.2}),
    horizon=12
)

# Pessimistic scenario: lower market potential
pessimistic = predict_model(
    result._replace(params={'m': result.params['m'] * 0.8}),
    horizon=12
)
```

## Peak Timing

For models like Bass, identify the peak adoption period:

```python
from innovate import peak_time

t_star = peak_time(result)
print(f"Peak adoption occurs at period: {t_star:.1f}")
```

## Model Summaries

Get a comprehensive summary of the fitted model and its forecasts:

```python
from innovate import summarise_model

summary = summarise_model(result, forecast=forecast)
print(summary)
```

This includes:

* Fitted parameter values with standard errors
* Information criteria (AIC, BIC)
* Forecast values with confidence intervals
* Goodness-of-fit statistics (R², RMSE, MAP)
