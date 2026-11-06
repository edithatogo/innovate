# Native scientific feature continuation — 7 November 2026

Local implementation and self-review in the existing isolated writing lane,
starting at `2e0b9930`. This is a development checkpoint, not full migration,
independent scientific acceptance, main integration or a release.

## Implemented and checked

- Native Norton–Bass now handles multiple generations, historical ODE semantics,
  an explicit successive-generation variant, fixed launch times, signed rates,
  fitting, forecasts and diagnostics. Independent three-generation DOP853 and
  analytic two-/three-generation cases preserve equation and column semantics.
- Lotka–Volterra and multi-product competition preserve their historical equations
  and add native fit/predict/simulate/summary/diagnose paths. Independent SciPy
  fixtures, uncoupled analytic limits and parameter recovery cover bounded cases.
- All seven fitters report actual optimizer termination and local QR sensitivity
  rank. Non-convergence, rank deficiency and unrepresentable sensitivities are
  explicit. This does not establish global identifiability or calibrated intervals.
- Polars fitting accepts ordered observation columns; prediction/rates retain all
  series and row identity. Fitted forecasts retain origins and launch times.
- Installed Python, Julia and R consume the native implementation. R one-element
  vectors remain JSON arrays, and recursive conversion preserves optional nulls.
- The policy guard retains all 16 model and 22 feature obligations. Removal,
  duplicate identities or making an obligation optional are rejected.

## Review findings resolved

Unknown options and malformed quantity values could silently disappear; they now
fail explicitly. Falsy non-mapping Python options are forwarded to Rust validation
instead of becoming defaults. Custom scalar fit seeds no longer bypass fitter-option guards.
Model-option aliases survive direct-parameter normalization. Multi-output arrays
are never truncated to the first column. Historical negative levels and signed
rates are retained with explicit semantics rather than clipped or called incidence.

Competitive fits conditioned on the first observed row now exclude that row from
conditional fit diagnostics and record the conditioning in fitted state. Initial
state uncertainty remains unpropagated. Numerical Jacobian and normal-equation
overflow now produces an error rather than a misleading convergence result.

A Polars test fixture mixed integer and fractional time literals; its construction
now explicitly uses Float64. The first failure and focused failure-only rerun are
retained. An integer-versus-float JSON comparison in a Rust oracle was corrected
to compare numeric values. Neither fix relaxed the scientific tolerances.

An archive rebuild initially reused an older native extension from the shared
Cargo build cache. Refreshing only the extracted Rust source mtimes forced the
three project crates to rebuild. A revised overflow regression, using a scale
that exceeds the optimizer domain after normalization, fails against the old
wheel and passes in the final installed wheel and both C ABI bindings. Source
bytes remain identical to the archive and checkout.

## Evidence and limits

The [receipt](native-features-20261107.json) binds logs, source hashes, native
artifacts and coverage. Python's final checks use a wheel built from the extracted
source distribution, with cached Rust dependencies. R and Julia use the rebuilt
C ABI library. No original Python test suite is relabelled as passing.

New scientific modules exceed the workflow's 80% line-coverage target. Overall
core/C ABI coverage remains below 80%; branch, mutation, cross-platform and hosted
proof are not claimed. Unchanged dependency-audit and selected Voiage evidence
remain linked to the preceding checkpoint; they were not rerun or promoted.

Covariates/events, threshold/equilibrium helpers, zero-capacity competition,
remaining families, Bayesian/bootstrap uncertainty, native Kairos execution,
R/Julia IPC and distribution bundles, and all other ledger obligations remain
open. The track stays active. Main/shared promotion stays with its integrator.
