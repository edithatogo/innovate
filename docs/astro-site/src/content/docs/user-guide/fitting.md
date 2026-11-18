---
title: Fit and assess a native model
description: Prepare observations, preserve model state, inspect diagnostics and distinguish parameter recovery from scientific validation.
---

The current table interface is [fit](/innovate/api/generated/fit/). It estimates
native model parameters from an ordered Polars table and returns a
[FitResult](/innovate/api/generated/fit-result/). This guide explains the choices
around that call; the [Norton–Bass tutorial](/innovate/tutorials/norton-bass/)
provides a complete executable recovery and forecasting example.

## Specify observations before selecting controls

| Input decision | Current contract | Analysis decision |
| --- | --- | --- |
| Time | Nonempty, finite, strictly increasing numeric `time`, without nulls | Choose the origin and unit and explain how dates became numeric time |
| Scalar observations | The default observation column is `observed` | State whether it measures a level, count, share or another quantity |
| Multiple series | Supply distinct `observed_columns` in model output order | Keep series identities and measurement units consistent |
| Model construction | Pass supported `model_options`; fitted state retains them | Specify launches, events or other structural assumptions before fitting |
| Fitting controls | Pass supported model-specific `fit_options` | Document starts, bounds, iteration limits and why they are appropriate |

Polars `LazyFrame` input is collected at the native boundary. Missing or invalid
required data fails; the wrapper does not silently choose an imputation model.
Column names do not establish compatible units or an appropriate observation
model.

## Start with a known answer

In the [two-generation example](/innovate/tutorials/norton-bass/), the native
predictions are checked against separate closed-form calculations before they
are used as synthetic observations. The fit then recovers six parameters, and
its predictions and signed rates are checked at times beyond the fitting window.
This sequence can expose interface, fitting and prediction discrepancies without
mistaking agreement with an unchecked data generator for independent evidence.

Run that example unchanged first. When adapting it, change one assumption at a
time and retain the original expected result as a regression case. A failure to
recover a known synthetic case should be investigated before fitting real data.

## Read the returned evidence

Inspect `parameters`, `diagnostics` and the complete `envelope`. Available
diagnostic fields depend on the model and operation; the wrapper does not
promise a universal AIC, BIC, likelihood or standard-error table. An optimizer's
convergence flag describes its stopping condition, not the identifiability of
every parameter or the adequacy of the model.

Use the returned result's `predict` method with an explicit new time table so
that stored model options are reused. Where profiling is supported, accessors
such as `launch_profile` expose the native candidate scores. Candidate scores
are not automatically calibrated confidence limits.

## Assess the analysis beyond the call

Consider how the observation process relates to the model quantity, whether the
observed window contains information about saturation or launch timing, and
whether several parameter combinations fit similarly. Examine held-out behavior,
residual structure and sensitivity to scientifically plausible assumptions.
These are analysis tasks; successful serialization and optimizer convergence do
not complete them.

For posterior sampling, use the separately documented
[native Bayesian operation](/innovate/tutorials/bayesian-fitter/). Its tutorial
checks a tractable posterior and chain diagnostics. A deterministic fit does not
acquire posterior uncertainty by changing its label, and returned uncertainty
still requires the assumptions and validation appropriate to the selected method.

## Failure handling and next steps

Preserve the native error code and validation details when present. A malformed
table, an unsupported feature and a scientific optimization failure can require
different remedies; consult the [fit exception contract](/innovate/api/generated/fit/)
instead of treating every failure as a request for more iterations.

Continue with [prediction and forecasting](/innovate/user-guide/forecasting/) and
[scenario experiments](/innovate/tutorials/scenario-experiment-workflows/).
The [earlier fitting guide](https://github.com/edithatogo/innovate/blob/6c226f92f14f01e771b1490c7fb3a7c77f4bc74a/docs/astro-site/src/content/docs/latest/user-guide/fitting.md) now points to retired source history; its removed `fit_model`, ensemble and backend examples
are not current API instructions.
