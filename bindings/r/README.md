# innovate R bindings

This package provides a thin R-facing adapter over the Python `innovate` functional kernel.

## Invocation path

The package shells out to the kernel bridge script at `inst/python/kernel_bridge.py` and passes
JSON request/response envelopes between R and Python.

The bridge expects the Python source tree to be available at `../src` relative to the repository
root and uses `uv run python` by default. Set `INNOVATE_PYTHON_COMMAND=python3` if you want to
override the launcher.

## Current scope

- Stable kernel discovery
- Versioned request construction
- Thin operation wrappers for the functional kernel

The R layer intentionally avoids duplicating any model logic.

## Example workflow

```r
source("bindings/r/R/kernel_bridge.R")

time <- c(0, 1, 2, 3, 4)
observed <- c(0.02, 0.06, 0.12, 0.25, 0.41)

discovery <- kernel_discover_models()
bass <- discovery[discovery$key == "bass", ]

fit <- kernel_fit_model(
  kernel_request(
    operation = "fit_model",
    model_key = bass$key[[1]],
    payload = list(
      inputs = list(time = time, observed = observed),
      model_kwargs = list()
    )
  )
)

diagnostics <- kernel_extract_diagnostics(fit)
prediction <- kernel_predict_model(
  kernel_request(
    operation = "predict_model",
    model_key = bass$key[[1]],
    payload = list(
      inputs = list(time = time),
      state = fit$state
    )
  )
)
```

## Installation

The bindings are intentionally light-weight and can be installed from the repository checkout
with standard R tooling:

```r
install.packages("devtools")
devtools::install_local("bindings/r")
```
