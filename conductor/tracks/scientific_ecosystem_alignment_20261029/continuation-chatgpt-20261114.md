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

PR #590 was the only open PR at reconciliation and remains a separate draft.
Its runner, offline assets and scanner/agent qualification are not part of this
change. Preserve all 38 migration obligations, the failing full-release gate,
the four reported manual scenario checks, historical negative evidence, and
all preserved local stashes/ignored material. No release is requested.

The live native source supersedes the old Python-file taskbook paths. Those
legacy paths and receipts are historical, not a reason to reconstruct a retired
Python implementation. The current README, migration ledger and this track's
native migration plan govern this bounded continuation.

## Selected packet: native ensemble time-grid validation

Existing scope: native workflow/scenario validation and the explicit time-grid
contract. This does not close the broad scenario, calibration or release areas.

The native `ensemble::compose` validates nonempty aligned lengths, weights and
finite results, but does not reject duplicate or reversed times. Its dispatcher
checks schema version and calls it directly; no caller supplies the missing
validation. Calibration already enforces finite, strictly increasing time, and
the README declares that contract for native time columns.

Allowed files for the branch `chatgpt/ensemble-time-grid-20261114`:

- `rust/crates/innovate-core/src/ensemble.rs` — only the missing time validation.
- `rust/crates/innovate-core/tests/ensemble_time_grid.rs` — public Rust regressions.
- `tests/native/test_ensemble_time_grid.py` — installed Python boundary checks.
- This continuation record — ownership, commands and actual evidence.

No dependencies, workflow definitions, protected checks, numerical tolerances,
reference outputs or migration-ledger statuses are to change.

## Acceptance sequence

- [x] Read live main, current native architecture, dispatch and owning track.
- [x] Add tests for duplicate/signed-zero, descending and locally reversed times.
- [x] Add independent weighted-result, irregular-grid, singleton and positive
  affine-time controls. No sorting or resampling is permitted as a repair.
- [ ] Observe behavioural failure against unchanged native production code.
- [ ] Add the minimum native validation and observe the unchanged tests pass.
- [ ] Review the whole diff and verify required exact-head hosted checks.
- [ ] Record integration status separately from scientific/release acceptance.

Commands: `cargo test --locked -p innovate-core --test ensemble_time_grid -j 2`
and `uv run --no-sync pytest tests/native/test_ensemble_time_grid.py -q`.
The existing CI additionally runs the complete core/tabular/FFI suites and
installed Python tests on Linux, macOS and Windows; no replacement workflow is
introduced. The existing Julia/R consumers remain part of the ordinary CI.

The ChatGPT sandbox terminal and Python both returned infrastructure ClientError
before any code ran. Verification must therefore use the actual hosted outputs;
no local red/green, build or numerical pass is claimed. At this tests-only
checkpoint, behavioural and final-head acceptance are still pending.
