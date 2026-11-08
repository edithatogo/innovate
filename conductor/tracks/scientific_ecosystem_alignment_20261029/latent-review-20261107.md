# Review Report: historical latent-process port

## Summary

Bounded native residual-smoothing port passes local consumer and scientific-contract checks; full migration remains incomplete.

## Verification Checks

- [x] **Plan Compliance**: Historical baseline fit, smoothed residual recursion, interpolation, endpoint continuation, state, summaries, diagnostics and increments retained.
- [x] **Style Compliance**: Cargo format, workspace Clippy, Ruff/types and normal hooks pass.
- [x] **New Tests**: Independent closed-form interpolation and residual recursion/sample-deviation oracle, forecast state, invalid inputs/overflow, summaries/clipping/singletons; Python Polars and Julia/R saved-state consumers.
- [x] **Test Coverage**: Core/C ABI lines 82.83%; latent module 91.75%. No branch/mutation claim.
- [x] **Test Results**: 79 Rust (75 core/C ABI plus four Polars), 32 installed Python, 155 Julia assertions and R conformance pass.

## Findings and corrections

Adjusted-fit AIC/BIC would use an unqualified parameter penalty for a residual
smoother estimated from the same observations. Both are now unavailable with an
explicit reason; baseline optimizer diagnostics remain separate. A regression
asserts this behavior. This does not remove the requirement to qualify effective
degrees of freedom and scientific comparisons.

The first broad Rust run caught stale discovery and unavailable-family tests.
The count is now twelve, and composite remains the fail-closed unavailable-model
oracle in Rust, Python, Julia and R. Both focused and full Rust reruns pass;
initial failed coverage output is retained as evidence.

## Scope and limits

Reviewed new Rust source/tests, dispatcher/discovery, ledger, binding transport
and docs against preserved source 56d91dd7 and the approved Rust redesign. Native
baseline optimization replaces SciPy; the oracle independently reconstructs
residuals from returned base parameters, smoothing, interpolation and forecast.
This is numerical/contract evidence, not external scientific validation.

Finite aligned latent arrays have strictly increasing finite-span times.
Smoothing is fixed in [0,1]; residual noise is nonnegative sample deviation.
No-time summary validates base parameters through the native scalar predictor.
Interpolation is convex within a finite interval and extends endpoint values.
Overflow is an explicit error. Historical nonnegative cumulative maximum is
batch-local; singleton increments retain the old return-level quirk. R preserves
singleton latent arrays; Python FitResult carries constructor state into Polars
forecasts. No numerical Python/Julia/R fallback was introduced.

Simulation currently returns a labelled deterministic point path. Gaussian draws,
calibrated uncertainty, full state-space inference, nested base families and
covariates remain required. All 38 ledger obligations remain required; twelve
families are bounded, four unavailable. No dependency/pin changes, main promotion,
push, hosted CI or release claim. Existing bincode maintenance advisory remains
unsuppressed; this unchanged graph was not re-audited here.

Local wheel installation, archive metadata and docs (163 pages, valid links) pass.
Clean-source/platform/hosted package qualification remains open.
