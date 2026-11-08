# Review Report: weighted scalar fits and native mixtures

## Summary

Weighted scalar least squares and bounded Gaussian/historical mixtures pass local qualification; full inference and scientific acceptance remain open.

## Verification Checks

- [x] **Plan Compliance**: Native weighting, mixture prediction/state, historical responsibilities and residual Gaussian EM implemented; remaining obligations retained.
- [x] **Style Compliance**: Cargo format, workspace Clippy, Ruff/types and normal hooks pass.
- [x] **New Tests**: Independent weighted SciPy reference, zero-weight exclusion/common scaling, closed-form responsibilities, synthetic component recovery with/without warm start, historical mode and failed-update rollback.
- [x] **Test Coverage**: Core/C ABI line coverage 84.13%; mixture 88.67%, shared scalar fitting 92.88%. No branch/mutation claim.
- [x] **Test Results**: 101 Rust (97 core/C ABI plus four Polars), 36 installed Python, 166 Julia assertions and R conformance pass.

## Findings and corrections

Clippy identified an overly complex parsed-parameter tuple; it is now a named
ModelParameters struct, without a suppression.

EM update records explicitly identify acceptance. Failed/non-converged component
updates or a Gaussian likelihood decrease beyond the stated numerical guard
roll back the whole iteration and cannot become successful convergence. The
regression asserts unchanged parameters, one objective-history entry, rejected
update and false convergence when an inner optimizer is deliberately limited.
Summary-only calls leave optimizer convergence unknown rather than inventing a
fit outcome from parameters. No remaining high-confidence finding in this diff.

## Review scope and limits

Reviewed new mixture module/tests, weighted-fitting changes, dispatcher/discovery,
ledger, Python Polars/Julia/R transport and docs against preserved source 56d91dd7.
No copied upstream algorithm, new native dependency or numerical binding fallback.
Scalar precision weights are finite, nonnegative and time-aligned, with positive
support at least equal to parameter count. Square-root/max normalization avoids
unnecessary common-scale sensitivity, and optimizer/local rank use the weighted
Jacobian. Raw predictions remain unweighted. Descriptive residual metrics are
unweighted; weighted RSS and representability are explicit. Precision-likelihood
AIC/BIC and calibrated uncertainty remain unavailable. Model-specific default
seed estimation may still use all observations; wide conditioning remains open.

The independent direct-logistic SciPy 1.18.1 reference records NumPy 2.5.4 and
Python 3.15.2 in an isolated oracle environment. Parameter and weighted-RSS checks
pass; that environment is separate from the Polars-only native runtime.

Mixtures accept 1–16 supported scalar curves, aligned nonnegative unit-sum weights,
finite ordered times and positive numerical bounds. Native prediction validates
all component parameters, including zero-weight components. Responsibilities use
stable log-sum-exp. Gaussian mode is a time-conditioned observation likelihood,
with a shared estimated standard deviation bounded by an explicit floor. It is
not aggregate-trajectory decomposition or calibrated class inference. Native
component local convergence and class identifiability are separate.

The historical variant preserves level-plus-epsilon responsibility equations,
epsilon precision weighting and pre-update stopping convention, while native
weighted optimization replaces old optimizer-specific sigma handling. Exact
legacy solver/basin parity is not asserted. Gaussian initialization fits global
curves with deterministic spread; complete warm starts remain available. Both
seeded and unseeded separated-component recovery are tested, without a global
optimum claim. Floor regularization, local convergence, failed updates and
objective history remain inspectable. Polars forecasts retain full native state;
R preserves singleton component and weight arrays.

Fifteen bounded families; composite unavailable. All 38 obligations remain
required and the optimized-Python release guard fails intentionally. Nested
families/covariates, rate helpers, stochastic draws, class selection,
heteroscedasticity, calibrated uncertainty and full inference/scientific
qualification remain open. Current macOS wheel installation, archive metadata
and docs (163 pages, valid links) pass. Clean-source/platform/hosted packaging,
main integration and release remain open. No push/publication performed. Existing
unsuppressed bincode maintenance advisory retained without a new audit claim.
