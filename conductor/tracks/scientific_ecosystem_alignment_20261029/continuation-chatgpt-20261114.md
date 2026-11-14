# Ordinary ChatGPT continuation — 14 November 2026

## Reconciled baseline and ownership

This continuing ordinary ChatGPT conversation is the integrator. The user reports
that the local owner is paused. No Codex/Work task, local-machine operation,
additional writer or other-repository change is authorised by this continuation.

Live main was read back as `276c2e505a0ce26910820fe8a94a6a498f505700`, tree
`3bd34207785fb23bd22969bb63535ca15c058c39`. The seven protected contexts remain:
`lint`, `package`, `CodeQL`, `Secret Scan`, `Workflow and policy lint`,
`Python quality harness`, and `unit-tests-base (3.15.1)`. The last name is retained
by branch protection; the current workflow runs Python 3.15.2.

PR #599 is merged and issue #571 is closed. Receiving-main CI `43567616550` and
CodeQL `43568829346` were read back successfully. Those are baseline receipts,
not proof of the new branch.

PR #590 was the only open PR at reconciliation and remains a separate draft.
Its runner, offline assets and scanner/agent qualification are not part of this
change. Preserve all 38 migration obligations, the failing full-release gate,
the four manual scenario checks, historical negative evidence, and all preserved
local stashes/ignored material. No release is requested. The four manual checks
remain unchecked in the archived scenario plan and are not automated acceptance.

The live native source supersedes the old Python-file taskbook paths. Those
legacy paths and receipts are historical, not a reason to reconstruct a retired
Python implementation. The current README, migration ledger and this track's
native migration plan govern this bounded continuation.

## Selected packet: native ensemble time-grid validation

Existing scope: native workflow/scenario validation and the explicit time-grid
contract. This does not close the broad scenario, calibration or release areas.

The native `ensemble::compose` validates nonempty aligned lengths, weights and
finite results, but did not reject duplicate or reversed times. Its dispatcher
checks schema version and calls it directly; no caller supplied the missing
validation. Calibration already enforces finite, strictly increasing time, and
the README declares that contract for native time columns.

Allowed files for PR #600, branch `chatgpt/ensemble-time-grid-20261114`:

- `rust/crates/innovate-core/src/ensemble.rs` — only the missing time validation.
- `rust/crates/innovate-core/tests/ensemble_time_grid.rs` — public Rust regressions.
- `tests/native/test_ensemble_time_grid.py` — installed Python boundary checks.
- This continuation record — ownership, commands and actual evidence.

No dependencies, workflow definitions, protected checks, numerical tolerances,
reference outputs or migration-ledger statuses change. The fix rejects a bad
grid before weighting or diagnostics; it does not sort or resample observations.

## Acceptance sequence and evidence

- [x] Read live main, current native architecture, dispatch and owning track.
- [x] Add tests for duplicate/signed-zero, descending and locally reversed times.
- [x] Add independent weighted-result, irregular-grid, singleton and positive
  affine-time controls. No sorting or resampling is permitted as a repair.
- [x] Observe behavioural failure against unchanged native production code.
- [x] Add the minimum native finite/strictly-increasing validation.
- [ ] Observe the unchanged regression tests and full required suites pass.
- [ ] Review the whole diff and verify required exact-head hosted checks.
- [ ] Record integration status separately from scientific/release acceptance.

The initial tests-only commit `a5c74cbdb955796419dbbce3397b064a5f6552d1`
contained a test declaration typo. Read-back caught it and tests-only commit
`44804f82b6d719c6457a338526513bd6f05ef31f` corrected it. The superseded initial
run is not counted as behavioural red evidence.

Actual red evidence: CI run `43571229444`, Ubuntu job `132376077441`, checked
out merge `f717a9d6806a6294e030cdc50e56c07cd77d525d` of corrected test head
`44804f82b6d719c6457a338526513bd6f05ef31f` against baseline `276c2e5`.
At 07:40:47 UTC on 14 November, all six new Rust tests executed: three valid-grid
controls passed and all three invalid-grid tests failed because `error.code`
was null instead of `invalid_request`. Compilation and the preceding native
unit/competition/complementary/composite tests passed. Cargo exited 101; later
Rust suites and the Python step were not completed, so no full-suite or Python
red result is claimed.

Commands: `cargo test --locked -p innovate-core --test ensemble_time_grid -j 2`
and `uv run --no-sync pytest tests/native/test_ensemble_time_grid.py -q`.
The existing CI additionally runs the complete core/tabular/FFI suites and
installed Python tests on Linux, macOS and Windows; no replacement workflow is
introduced. The existing Julia/R consumers remain part of ordinary CI.

Release Readiness run `43570857028`, job `132374708881`, independently retained
exactly 38 incomplete obligations at `scripts/check_migration.py --release`.
The new implementation does not change that ledger or imply release readiness.

The ChatGPT sandbox terminal and Python both returned infrastructure ClientError
before any code ran. Verification therefore uses actual hosted outputs; no local
red/green, build or numerical pass is claimed. At this fix checkpoint, green
verification, final-head acceptance and integration remain pending.

## Separate historical-alert reconciliation

Only historical incident #395 was closed after individual verification of its
original report and nine repeat reports. All ten source run records were read.
Matching scheduled CI run `42649041248` at `95579544ebf1bcc8b1455468ba311c486692761d`
passed, including the Linux scheduled source-distribution, remote-container,
benchmark and local-MPI steps. The issue comment contains the complete run table
and the distinction between workflow recovery and scientific revalidation.
No other historical alert is closed by that receipt.
