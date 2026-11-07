# Lock-in migration review

Recovered historical equations from 56d91dd7. Native prediction separately clips
RHS states and reported levels to [0,m], retaining explicit initial origin for
forecasts. Historical rates are RHS values at reported levels, not derivatives
of clipped outputs. Independent SciPy DOP853 and analytic forecast limits pass.

Interior trajectories identify six polynomial combinations of seven parameters;
m is confounded with feedback/competition coefficients. Diagnostics explicitly
scope this structural warning to unclipped interior trajectories. Boundary
identification and calibrated uncertainty remain unqualified. Fitting preserves
m >= max(observed) across both series using an offset-positive transform,
rejects infeasible initial seeds, and initializes m at 1.5 times that maximum.
The bound regression failed before the fix and passes afterward.

Current-source verification: 62 Rust tests (58 core/C ABI coverage-run plus four
Polars), 28 installed Python checks, 144 Julia assertions and R native cases pass.
Core/C ABI line coverage is 81.48%. Clippy, lint/types, docs and package metadata
pass within local scope. A disk-space failure invalidated the first post-bound
binding attempt; only the fresh postbound logs support final binding results.
An owned stale build cache was removed; source, prior artifacts and evidence were
preserved. Dependency graph is unchanged from network-resource-20261107.json;
its unsuppressed bincode maintenance warning remains. Full clean-source/platform
packaging and scientific qualification remain open. No main promotion/publication
occurred. All 16 models and 22 feature areas remain required.
