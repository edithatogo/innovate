# Scientific alignment continuation — 6 November 2026

## Source and ownership reconciliation

Base: `b9998addf32717438df92b23a306903e4db51620`. The actual repository is
`/Volumes/PortableSSD/GitHub/innovate/innovate`; its outer wrapper has a different
Git history and no project remote. This lane uses the standard Git worktree and
scope recorded in `plan.md`. The earlier planning worktree is clean except for
its pre-existing untracked `.conductor/` state, which was preserved. No running
Innovate native operation was found. The earlier planning chat's retained app
history could not be retrieved; no ownership or native operation was taken over.

Archived evidence reconciliation is committed at `5cd07b7f`: one historical
inventory task and two retained-note Rust verifications were closed. Four human
Scenario verifications remain pending, with explicit metadata and registry
qualification. The retained 2 November release/installed Arrow-EVPI and
authentication-only receipts remain the baseline; they were not repeated or
promoted into broader acceptance.

## Actual Kairos call inventory

| Surface | Actual behavior before this continuation | Current qualification |
|---|---|---|
| `bindings/rust/Cargo.toml` | Seven `kairo-ecs-*` git dependencies at `fae901558f07b7b717a676adbafbe2cdc78dea1c` | Pin unchanged; candidate native macOS arm64 build passed |
| `bindings/rust/src/lib.rs` | No calls into Kairos APIs | Native binding kernel behavior is separate from Kairos integration |
| `examples/kairos_des_smoke.rs` | Commented intended imports and unconditional success prints | Replaced with actual upstream Scheduler, DeterministicStream and Resource calls; replay/order/FIFO assertions passed |
| `examples/kairos_abm_smoke.rs` | Commented intended imports and unconditional success prints | Replaced with upstream ABMContext calls; identity, component state, event order and replay assertions passed |
| `src/innovate/abm/kairos_adapter.py` | Python reference scheduler/agent loop; no native bridge dispatch | Existing reference behavior retained; bridge promotion remains gated |
| `KairosDependencyEvidence` | Manifest/example presence incorrectly claimed Kairos-backed simulation | Requires promoted bridge in addition to source prerequisites; collector cannot promote an absent dispatch path |

No scheduler was copied from upstream. The examples exercise upstream crates.
No new upstream revision, shared dependency, runtime or registry promotion occurs.
A full native Python adapter migration remains required before claiming its
scheduler/agent loop is delegated to Kairos.

## Regressions and measured native checks

Two existing status assertions were changed to require false for unsupported
native dispatch. Both failed against the original code, then both passed after
the fix; all 16 adapter contract tests passed. Ruff check/format on the affected
Python files and `cargo fmt --check` passed. The release-readiness adapter
artifact now uses the same fail-closed value.

The 89-test focused gate covered registry hygiene, adapter contracts, roadmap
status and all four Scenario test modules. Registry hygiene reports 94 registered
tracks with no missing, orphaned, stale or wrong-link targets. The 29 historical
Rust-ownership tests also passed on the current source; that is Python contract
coverage, not fresh Rust parity proof.

See `kairos-native-20261106.json` for exact argv, versions, input hashes, output,
wall times, disk observations and the process-tree memory high-water observation.
Python was explicitly verified as 3.15.1 and Rust/Cargo as 1.99.1. Builds used
two jobs and a 300-second command timeout. The DES and ABM examples passed.
`cargo test --lib` succeeded with **zero tests**, so it supplies compilation only.

A fresh checkout has no Rust lock: `bindings/rust/.gitignore` ignores Cargo.lock
and Git tracks none, despite older build-plumbing tests asserting otherwise.
The locally generated 104-package candidate is retained verbatim as
`kairos-candidate.Cargo.lock` with its digest in the receipt. It is evidence for
this build, not a promoted dependency lock or fresh-checkout reproducibility.
The repository integrator must decide and qualify the committed-lock policy.

## Installed consumer checks

A wheel was built with the repo-local Python 3.15.1 interpreter and installed
into a separate environment outside the checkout. `python -I` verified its
`site-packages` import origin and unsupported-native status. The adapter-only
receipt is `installed-adapter-20261106.json`; it deliberately installed no runtime
dependencies and is not a full base-install or dependency-check claim.

The selected forecast/diagnostics checks and first-failure evidence are recorded
separately in `installed-consumer-20261106.json`. Broader scientific acceptance,
optional Polars/JAX/Bayesian consumers, HPC graph/build/cluster gates, hosted
checks and final integrator acceptance remain open.

The installed CPU checks passed after installing the declared statsmodels 0.14.6
runtime dependency. Its Python 3.15.1 source build completed in 1m50s; no runtime
was downgraded. A first probe used the wrong Bass time origin and unrealistically
exact analytic tolerance. Source inspection confirmed that the initial state is
at the first supplied time. The corrected probe uses t=0..8 and the numerical
ODE comparison bounds rtol=1e-7, atol=1e-6. Maximum absolute forecast error was
1.0758947269096097e-05 for population 1000. Both initial failures are retained in
the receipt. Two installed Arrow/diagnostics tests and the analytic forecast
then passed. This is a selected CPU case, not all scientific scenarios or a full
base dependency check. Statsmodels emitted pandas deprecation warnings.

Current-source validation totals 134 distinct focused tests: 89 in the initial
gate, 29 Rust-ownership Python contract checks, six Arrow interchange tests and
ten Kairos smoke/dependency checks. Two old Kairos tests used pre-archive report
paths; both failed, were corrected to the existing archived report, and passed
in a failure-only rerun. The Arrow diagnostics failure was a missing declared
statsmodels dependency in the deliberately minimal environment, then passed in
a failure-only rerun after that dependency was installed. No retries or skips
were used to conceal failures.

## Next gates and terminal handoff

This lane completes source/ownership reconciliation and a regression-backed
correction, with bounded native and installed-consumer receipts. The Kairos
qualification and installed forecast tasks remain in progress because these
cases do not prove all accepted scientific/native requirements. The native pin
is unchanged. Its generated lock must not be silently adopted as the repository's
accepted dependency graph.

The repository integrator retains main/shared runtime and pin promotion. Next:
review this scoped branch, resolve and qualify the fresh-checkout Rust lock
policy, run the full `uv run nox -s lint types tests docs package` gate and hosted
checks, and qualify the remaining optional/native consumers. The full Nox gate was attempted twice; see the terminal gate record below. No release, merge, dispatch,
cluster acceptance or full-track archive is authorized by these local results.
The bounded Linux graph job identified on 2 November remains a later HPC step;
it must preserve the exact Spack and two EasyBuild generation inputs.

## Terminal full-gate result

The required Nox command was attempted. The first attempt terminated with exit
138 without a terminal unit result; its cause remains unknown. A direct bounded
rerun passed lint and types, then was stopped by this lane at pytest elapsed
15:08 near the same late-suite position. No test assertion failure was reported
before the stop, but there is no complete unit or coverage result to accept.
Nox continued: docs built 162 pages with valid internal links, and package built
wheel/sdist with successful Twine and wheel-content checks. Overall exit was 1.

`full-gate-20261106.json` retains both log digests, exact outcome counts and
collection diagnostics. Collection succeeds with 1536 selected of 1541 tests.
The next unreported node inferred from collection order is a diagnostic lead,
not a proven failing test. No further full-suite retry was made. The source
environment remains Python 3.15.1; `uv pip check` reports 164 compatible packages.
The isolated installed-wheel receipts retain their own narrower dependency scope.

Next operation for the integrator: investigate that exact numerical-test runtime
with a measured case budget, complete the full unit/coverage gate, and resolve
the Rust lock policy before native/runtime promotion. This is the terminal
handoff for this lane; main, other worktrees and native owners remain unchanged.

The collection diagnostic points to
`tests/unit/test_specific_modules_coverage.py::TestFitterCoverage::test_bootstrap_fitter_basic`
as the first node without a reported outcome in both attempts. Source inspection
shows five bootstrap fits with no seed specified. This is a runtime diagnostic
lead, not proof that the test or fitter is defective. Do not reduce its coverage,
skip it for speed or change scientific behavior to turn the gate green; qualify
a bounded reproducible case before a fitter/test-policy change.

Delivery commits: `5cd07b7f` (archive reconciliation) and `1a6e2752` (Kairos
source/examples, selected-consumer evidence and local review). This lane has not
pushed, merged, dispatched hosted workflows, changed release state or promoted
a dependency/runtime. The four Scenario manual gates remain pending.
