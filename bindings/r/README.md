# innovate R bindings

This package provides a thin R-facing adapter over the Python `innovate` functional kernel.

## Invocation path

The package shells out to the kernel bridge script at `inst/python/kernel_bridge.py` and passes
JSON request/response envelopes between R and Python.

The bridge uses the installed `inst/python/kernel_bridge.py` helper and invokes Python through
`uv run python` by default. Set `INNOVATE_PYTHON_COMMAND=python3` to call an existing Python
environment directly. Installed use expects the Python `innovate` package to be available to that
Python environment.

## Current scope

- Stable kernel discovery
- Versioned request construction
- Thin operation wrappers for the functional kernel

The R layer intentionally avoids duplicating any model logic.

## API surface

The exported surface is intentionally small and maps directly to the kernel bridge:

- `kernel_schema_version()`
- `kernel_request()`
- `kernel_call()`
- `kernel_response_to_r()`
- `kernel_discover_models()`
- `kernel_fit_model()`
- `kernel_predict_model()`
- `kernel_simulate_model()`
- `kernel_summarize_model()`
- `kernel_diagnose_model()`
- `kernel_extract_diagnostics()`

Use `kernel_extract_diagnostics()` to read the diagnostics envelope from fit and summary
responses without depending on the Python response shape.

## Backend expectations

The bindings are a thin adapter over the Python kernel bridge, so the runtime expects:

- the Python `innovate` package available to the selected Python runtime
- `uv run python` as the default launcher, or `INNOVATE_PYTHON_COMMAND=python3`
- `INNOVATE_PYTHON_COMMAND` only when you need to override the launcher explicitly

The package does not reimplement model logic in R. It forwards requests, translates responses,
and preserves the kernel diagnostics payload for downstream inspection.

## Example workflow

```r
library(innovate.R)

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

## Release and package checks

Run package checks from the repository root so paths match CI:

```bash
Rscript -e 'install.packages("jsonlite", repos = "https://cloud.r-project.org")'
Rscript bindings/r/tests/run.R
R CMD build bindings/r
R CMD check --as-cran --no-manual innovate.R_*.tar.gz
```

Build the local PDF manual when reviewing documentation changes:

```bash
R CMD Rd2pdf bindings/r --output=innovate.R-manual.pdf
```

The package includes a source vignette under `vignettes/`. `R CMD build
bindings/r` builds it by default, and release candidates must not bypass
vignette checks unless the release notes explicitly call out a temporary
maintainer-only exception.

Generated release artifacts are inspection outputs, not source files. Do not
commit `innovate.R_*.tar.gz`, `innovate.R.Rcheck/`, `innovate.R-manual.pdf`,
or generated vignette build products. Keep source documentation in `README.md`,
`man/`, and future vignette sources.

CI publishes the source tarball as a workflow artifact from the R package job.
For a release candidate, retrieve `innovate.R_*.tar.gz` from the binding
publication workflow artifact, inspect the `R CMD check --as-cran` log, and
only then enable R-universe publication or prepare a CRAN submission. The R
publication quality gate is: integration tests pass, `R CMD build` succeeds,
`R CMD check --as-cran` has no errors or warnings, the manual builds locally,
and generated artifacts are not committed.
