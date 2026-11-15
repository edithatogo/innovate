> Frozen candidate document completed before the ownership recheck. The handoff README records the current ownership hold; this snapshot does not change the active track or establish sign-off.

# Bounded Conductor review — Track 5

## Verdict

The allowed source migration and preservation tests are implemented. Independent
review found no remaining defect in the exact-constant repair or scope boundary.
The draft is **not ready for integration acceptance**: the shared dependency is
not declared in this branch and the candidate library fails the unchanged CDF
accuracy gate. Neither phase completion nor scientific/release sign-off is claimed.

## Method and scope

The Conductor implementation/review workflow was applied against the current
plan, specification, project workflow, bounded execution contract and scientific
obligations. A separate reviewer inspected live ownership, source, fixtures,
tests and receipts; the implementation agent executed the commands recorded
in evidence. Review is distinct from numerical execution and hosted checks.

Allowed paths are the two named modules, dedicated new lane fixtures/tests and
this track directory. Review confirmed no edits to `causal.rs`, `numerics.rs`,
Cargo manifests/locks, CI, shared registry, scientific thresholds or existing
tests. No new test is ignored. The dependency diff is a request artifact under
this track, not a change to the protected shared files.

## Findings and dispositions

| Finding | Disposition |
| --- | --- |
| Proposed distribution names did not correspond to actual helpers in bootstrap/causal. | Source audit recorded; retain empirical percentiles, fixed multiplier and model arithmetic. Causal stays reserved. |
| Replacing relative Gaussian kernels with normalized log-PDF subtraction loses near-centre scores. | Keep original arithmetic. Independent fixture and an isolated negative control detect the cancellation at `-5e-21`. |
| The private quantile helper's invalid-probability saturation differs from the dependency. | Prove the sole caller's validated Blom range; retain separate historical boundary characterization and explicit dependency-domain tests. No new public API is introduced. |
| `try_inverse_cdf` inherits panicking behavior for invalid probabilities in the pinned dependency. | Tests record actual panic behavior for both entry points; public valid-input proof is independent. |
| The CDF assertion stopped at the first failure. | Collect all 36 CDF/survival comparisons and report all 14 failures, retaining the original tolerances. |
| Exact JSON equality for posterior grid floats overstates portability. | Retain exact sampled draws/accepted counts on the recorded target; use the existing `1e-10` mean and `1e-12` grid-mass tolerances. Add target/libc provenance. Fixture outputs themselves are unchanged. |
| More precise quantiles expose rounded means for exactly constant splits. | Exact per-split equality guards preserve null ESS/R-hat. Different split constants are allowed; one varying split uses the original formula. R-hat retains its previous nonfinite error ordering. |
| Some long constant-chain artifacts predate this migration. | Baseline/candidate probe distinguishes those correctness fixes from the genuine 80-draw ESS regression. No blanket bitwise-preservation claim. |
| Test header implied later regression cases were original baseline tests. | Header now separates pre-migration characterization from post-migration regression tests. |

## Constant-split evidence

Before repair the half-40 regression returns `Some(2.0253164556962027)` where
the established null convention requires `None`. The regression suite includes
split lengths 3, 40 and 5,000; differing constant values; one split varying by
an adjacent representable float; and overflow retaining the previous R-hat
error. Public tests exercise 80- and 10,000-draw distinct/piecewise constant
chains and all associated nullable diagnostic fields.

The isolated negative controls reverse quantile probability and subtract
normalized log densities. Both are rejected by the independent tests. An initial
score-mutant invocation reused a stale build artifact; it is not counted. The
recorded score-mutant run explicitly recompiles the changed source and fails
on the near-centre reference. Candidate source bytes and timestamps were then
refreshed and recompiled for final qualification. These are two concrete
sensitivity checks, not a mutation-score claim.

## Acceptance still open

The handoff keeps shared dependency/lock/feature reconciliation, RustSec audit,
the CDF failure, exact integrated-head CI, full workspace and installed consumers,
coverage, native mutation qualification, performance, platform/package and
scientific-release gates open. Required gate settings and solo-maintainer
approval policy were not changed. No downstream planning-only track advances.
