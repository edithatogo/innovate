# Review Report: native composite growth ODEs

## Summary

Bounded heterogeneous growth, signed interactions, fitting and fixed-origin state pass local qualification; full feature/scientific migration remains open.

## Verification Checks

- [x] **Plan Compliance**: Composite equations/suffixes, signed interactions, historical batch resets, fixed-origin forecasts, matrix levels/rates and fitting implemented.
- [x] **Style Compliance**: Cargo format, workspace Clippy, Ruff/types and normal hooks pass.
- [x] **New Tests**: Independent analytic and SciPy ODE references, signed coupled fitting, absent-parameter behavior, initial-condition forecasts, malformed inputs and precision-weight dispatch regression.
- [x] **Test Coverage**: Core/C ABI line coverage 84.32%; composite 87.00%. No branch/mutation claim.
- [x] **Test Results**: 110 Rust in the broad gates (106 core/C ABI plus four Polars), plus one later focused capacity-policy regression; 37 installed Python, 170 Julia assertions and R conformance pass.

## Findings and corrections

The scalar weighting change shared its option parser with competitive and
historical covariate/event fitters. Those paths accepted but did not apply
weights. A failing regression now passes: scalar_options alone admits precision
weights, while the shared options parser rejects unsupported weighting. This
preserves all existing weighted scalar/mixture checks and prevents silent loss.
The full native gates include both competitive and historical event regressions.

Capacity bounds default to positive for the redesigned API, because stimulating
interactions may raise observed levels above intrinsic capacity. Explicit
historical_observed retains the previous observed-maximum floors. A later focused
regression verifies both policies using a stimulated path above capacity. Only
that test file changed after the broad run; unchanged production sources and
broad results are reused. The reported line coverage is the unchanged-source
broad measurement, without incorporating this additional focused run.

Clippy's nested-option finding was fixed without a suppression. No remaining
high-confidence finding in this bounded diff.

## Review scope and limits

Reviewed new composite source/tests and fixture, shared fitting dispatch,
discovery/ledger, Python Polars and Julia/R consumers, and docs against preserved
source 56d91dd7. The native RHS is own growth minus signed off-diagonal linear
peer effects. Complete one-based suffixed parameters are validated; zero-diagonal
alpha is a fitting seed, not an implicit replacement for named prediction values.
All component domains and finite shape/time/origin bounds are checked. Negative
levels remain visible with warnings, and rates are signed derivatives.

Independent direct SciPy 1.18.1 DOP853 with NumPy 2.5.4/Python 3.15.2 qualifies
heterogeneous Bass/Logistic/Gompertz levels/rates. Decoupled analytic Bass,
Logistic and Fisher–Pry cases pass. Coupled fitting starts with a perturbed
interaction and recovers a low-residual trajectory while retaining absent
parameters. These are numerical/contract checks, not global scientific acceptance.

Logistic x0, Gompertz b and Fisher–Pry t0 are structurally absent from the ODE and
held fixed. Local rank covers active parameters only, and conditional/structural
information criteria remain unavailable. Inferred initial state is conditioned
on the first observed row, excluded from objective/statistics, and stored for
forecasts without uncertainty propagation. Explicit historical_batch resets zero
state at each requested start; singleton rows are defined. The Fisher–Pry native
adapter fixes the old generic-call signature mismatch. No numerical binding
fallback, copied solver or new runtime dependency introduced.

Current solver is adaptive explicit RK4 with step doubling/Richardson correction;
full stiff/BDF workload parity remains required. Nested base families, covariates/
events, weighted composite fits, stochastic paths, uncertainty and scientific
qualification remain open. All sixteen families are now bounded native, but all
38 original/planned obligations remain required and optimized-Python release
guard fails intentionally. Discovery scope text now reflects bounded extensions
without claiming the entire scientific migration complete. Unknown model and
real unavailable-covariate checks retain fail-closed behavior across the native
boundary and installed Python consumers.

Local macOS wheel installation, source archive metadata and docs (163 pages,
valid links) pass. Clean-source/platform/hosted packaging, main integration and
release remain open. No push/publication. Existing unsuppressed bincode
maintenance advisory retained without a new audit claim.

Final package recheck encountered an internal-volume output redirection failure.
Earlier package logs were not used as current proof. The source archive was
regenerated on the owned external volume, metadata/wheel checks passed, and
exact current source/fixture/test inclusion was verified. Current packaging logs
are final-sdist.log and final-package.log; numerical results were unchanged.
