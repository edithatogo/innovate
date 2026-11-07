# Review Report: hierarchical grouped point-model port

## Summary

Bounded grouped point fitting and aggregate/group forecasts pass local qualification; full hierarchical inference remains required.

## Verification Checks

- [x] **Plan Compliance**: Shared and grouped-series fits, global/group parameter aggregation, historical variant, centered default, forecasts, summaries and diagnostics implemented.
- [x] **Style Compliance**: Cargo format, workspace Clippy, Ruff/types and normal hooks pass.
- [x] **New Tests**: Independent analytic group curves, historical absolute-plus-global encoding, shared fits, saved forecasts, malformed labels/shapes/parameters and large-constant R² regression.
- [x] **Test Coverage**: Core/C ABI line coverage 83.41%; hierarchical module 86.28%. No branch/mutation claim.
- [x] **Test Results**: 92 Rust (88 core/C ABI plus four Polars), 34 installed Python, 162 Julia assertions and R conformance pass.

## Findings and corrections

Large constant observations overflowed the old raw sum used to compute a mean,
producing R²=1 for data with zero total variation. A failing regression now
passes: scalar and per-series diagnostic means divide before summing, while
undefined or unrepresentable total variation returns unavailable R². Existing
numerical, rank and unsupported-family checks remain intact.

For the redesigned Rust API, centered parameters are the default, so grouped
fits recover their independently fitted curves. `historical_additive` explicitly
preserves the old absolute group estimates plus global mean contribution. The
regression exercises that variant separately; Rust and installed Python/Julia/R
checks exercise the new default. Historical shared-series fitting still fits
once with zero group adjustments and sums the group forecasts.

No remaining high-confidence finding in this bounded diff.

## Scope and limits

Reviewed new module/tests, dispatcher/discovery/ledger, shared diagnostic changes,
Polars consumer, Julia/R state transport and docs against preserved source
56d91dd7 and the approved breaking redesign. Both labeled group mappings and
time-by-group matrices are validated; groups are 1–128 unique nonempty labels,
with global reserved to prevent parameter-name collisions. All global/group
keys are required and effective curve domains are validated through the scalar
native predictor. Global means avoid unnecessary sum overflow. Extreme centered
subtraction conditioning remains unqualified and documented; nonfinite effective
parameters or aggregates are explicit errors.

Aggregate output retains group rows; group output preserves every Polars column
and exposes a separate aggregate. R preserves singleton group labels and named
observed/y mappings. No numerical binding fallback introduced. Diagnostics use
actual groups for grouped observations and the historical summed prediction for
shared observations. RSS overflow is rejected before multivariate diagnostics.
AIC/BIC remain unavailable pending decomposition degrees-of-freedom qualification;
independent scalar-fit diagnostics remain separate. Historical increment layout
is retained, with the previously failing singleton explicitly defined as zero.

This is grouped point fitting, without pooling/shrinkage or hierarchical Bayesian
inference. Stochastic draws, calibrated uncertainty, full inference, nested base
families, covariates and wide-conditioning/scientific qualification remain
required. Native derivative output remains unavailable.

Fourteen bounded families, mixture/composite unavailable; all 38 obligations remain
required and optimized-Python release guard fails intentionally. Local macOS
wheel installation, source archive metadata and docs (163 pages, valid links)
pass. Clean-source/platform/hosted packaging, main integration and scientific
acceptance remain open. No dependency/pin change, push or public release. Existing
unsuppressed bincode maintenance advisory retained without a new audit claim.
