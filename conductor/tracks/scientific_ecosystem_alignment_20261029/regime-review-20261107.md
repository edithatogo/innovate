# Review Report: regime-switching native port

## Summary

Bounded native regime fitting, detection and forecast modes pass local qualification; full migration remains incomplete.

## Verification Checks

- [x] **Plan Compliance**: Historical batch-index variant retained; separately named fixed-time variant, boundary selection, native detector, state and diagnostics implemented.
- [x] **Style Compliance**: Cargo format, workspace Clippy, Ruff/types and normal hooks pass.
- [x] **New Tests**: Independent rupture fixtures, closed-form predictions, exact boundary continuity, batch forecast equivalence, fitted segment parameter recovery, malformed state and short Fisher–Pry identification.
- [x] **Test Coverage**: Core/C ABI line coverage 83.22%; regime module 89.02%. No branch/mutation claim.
- [x] **Test Results**: 86 Rust (82 core/C ABI plus four Polars), 33 installed Python, 159 Julia assertions and R conformance pass.

## Findings and corrections

The shared scalar fitter originally rejected all samples below four. The
historical short-regime Fisher–Pry path needs its identified two-parameter fit
from two interior shares. A failing regression now passes with alpha=1,t0=2.
Only Fisher–Pry's minimum changes to two; other scalar minima remain four.
The just-identified fit reports no residual degrees of freedom and unavailable
AIC/BIC. No rank/domain/convergence checks, skips or tolerances were weakened.

A supplied fit-time split boundary was initially overwritten by automatic
detection. The failing regression found index 5 instead of the requested split
at 14.5 (index 15). Time selection now partitions at the first observation at or
after the boundary and retains its exact value for continuity. Simultaneous index
and time selection, or an empty side, are explicit errors.

No remaining high-confidence finding in this bounded diff. Split-selection
information criteria remain unavailable; independently fitted segment diagnostics
are separate from continuity-adjusted fit metrics.

## Review scope and limits

Reviewed Rust source/dispatcher/discovery, shared fitting/validation changes,
all new tests and fixture data, Python/Julia/R transport, ledger and docs against
preserved source 56d91dd7 and the approved redesign. No numerical binding fallback
or copied upstream implementation introduced. Native dynamic programming uses
Welford segment errors, positive penalties, minimum length two and a bounded
2000-observation grid; explicit splits bypass detection for larger fits. It is
unpruned, with ascending-predecessor ties. Seven reference cases from isolated
Python 3.15.2, ruptures 1.1.10, NumPy 2.5.4 and SciPy 1.18.1 match the upstream
Pelt outputs. The small-step case labelled "tie" is not universal mathematical
tie parity. Floating-point/tie/performance parity beyond these fixtures remains
unqualified. Oracle dependencies are isolated from the native runtime.

`historical_index` retains batch position and offset behavior, including first-
regime-only short forecasts. `fixed_time` anchors continuity at the saved exact
time and passes combined-versus-separate forecast checks. Both clip nonnegative
and accumulate maxima. Single retained regimes and no-time summaries are tested.
All regime parameters are validated even when a batch only uses the first curve.
Stochastic draws, calibrated uncertainty, full probabilistic regime inference,
other detector objectives/search methods, nested base families/covariates and
complete small-sample scientific qualification remain required.

Thirteen bounded families, three unavailable, all 38 obligations still required.
Release guard fails intentionally even with Python optimization. Current macOS
wheel installation and source archive metadata pass; docs build 163 pages with
valid links. Clean-source/platform/hosted qualification, repository integration
and scientific acceptance remain open. No main/push/publication performed.
Dependency graph unchanged; existing unsuppressed bincode maintenance advisory
retained without a new audit claim.
