# Issue #413: native testing and CI qualification

Status: implemented; hosted native execution and integration pending. This receipt contributes to
Q01/Q02; it does not close their broader acceptance requirements.

## Reconciliation and ownership

Baseline: `276c2e505a0ce26910820fe8a94a6a498f505700` (main).
[Issue #413](https://github.com/edithatogo/innovate/issues/413) and parent #415
record a July inventory. DST is interpreted here as deterministic simulation
testing and MT as metamorphic testing; parent #415 lists mutation separately.
The existing Rust-native suites already implement both techniques. The historical
Python/mutmut inventory is not current native execution evidence.

Allowed changes: one new native metamorphic test file, the existing CI/security
and binding-conformance workflows, and this owning-track receipt/plan entry.
PR #600 owns ensemble time-grid validation; PR #590 is an unqualified offline
security draft. Neither is a dependency or part of this change. Runtime source,
manifests, dependency/provider pins, scientific thresholds, required check names
and the migration ledger are preserved.

## Applicable techniques and exclusions

| Technique | Existing executable scope | Issue #413 increment |
| --- | --- | --- |
| Deterministic simulation | `tests/kairos.rs`: fixed-seed DES/ABM replay, sampling-grid invariance, checkpoint restoration and independent binomial expectations. `tests/network.rs`: checkpoint recovery at every tick, queued service, scheduling ties and resource conservation over fixed seeds. | Re-execute these public native integration targets and retain source/runtime/lock identity with results. |
| Metamorphic relations | `tests/network.rs`: common edge-weight scaling across 128 seeds. `tests/model_oracles.rs`: serial/parallel equivalence and prediction subsets. `tests/scientific_contract.rs`: Bass time/output-unit rescaling. | A named `tests/metamorphic.rs` target covers supported model time-unit covariance, amplitude scaling and mixture permutation, with analytic landmarks and finite case tables. |
| Mutation | Native scientific mutation remains unqualified in `conductor/workflow.md`; historical Python mutmut does not qualify Rust. | No mutation-score claim. The weekly native runner and >70% target remain open; this issue's MT evidence is not mutation evidence. |
| Workflow controls | Read-only defaults, scoped elevated permissions, SHA-pinned actions, update configuration, lint/security gates and scheduled expensive lanes already exist. | Verify the actionlint release archive before execution; add explicit job/step time limits and binding concurrency; retain native-test logs on success/failure. |

Rust test paths in the table are relative to `rust/crates/innovate-core/`.
The simulation engine, scheduler, RNG and replay implementation remain Kairos-owned.
These tests exercise Innovate's pinned consumer contract; no second simulator is
introduced. Fault-tolerant distributed/HPC, empirical model validity, unsupported
inference capabilities, live/offline security runner provisioning and deployment
are outside this bounded change. Their existing requirements remain intact.

Time-unit covariance changes time and inverse-time parameters together; it does
not imply arbitrary time-shift invariance for origin-dependent models. Amplitude
scaling is limited to models with an amplitude/capacity parameter. Fisher–Pry is a
probability, so an arbitrary output scale is inapplicable. No generic covariance
claim is made for fitted parameters, priors, covariates, event histories or
discretised simulation tick units. Metamorphic equivalence complements the
existing independent mathematical oracles; it cannot establish empirical validity.

## Runtime and evidence policy

Keep the fixed-case tests in the ordinary native suite on Linux/macOS/Windows.
Do not create a duplicate build or an artificial large PR sweep. Existing source
distribution, worker-container and benchmark/MPI lanes remain scheduled/manual.
Require zero failed native tests. Each numerical relation records its tolerance
before execution; same-input seeded simulation replay retains exact equality.

Native test execution is bounded at 20 minutes after dependency installation;
ordinary native and Julia/R jobs have 45-minute limits. Scheduled native jobs have
90 minutes for their additional existing installed-consumer and benchmark work.
These limits exceed observed main-run native job times (approximately 9–19 minutes
in run `43567616550`) without leaving GitHub's six-hour default as the bound.
The existing narrower container/MPI limits remain, and source-distribution and
benchmark steps each receive a 20-minute limit.

Every native test job retains its full Cargo log, actual checkout revision,
Cargo.lock SHA-256, compiler/Cargo versions and process exit status for 30 days.
The shell uses pipefail and propagates a failed Cargo/tee process; uploading
evidence cannot convert a failed test into success. A timeout/cancellation may
leave a partial log without an exit-code receipt and never counts as a pass.
GitHub's job conclusion remains authoritative for interrupted runs.

## Executed evidence

The new matrix has three tests and 119 public native invocations: 64 for time
covariance, 48 for amplitude covariance and seven mixture diagnostic calls. It
uses eight fixed curves, six time points and a predeclared mixed tolerance of
`1e-12 + 1e-12 * max(abs(actual), abs(expected))`. Analytic landmarks, finite/shape
checks, nonconstant curves and distinct mixture responsibilities prevent a zero
or uniform result from satisfying equivalence vacuously.

Local execution uses Rust `1.99.1 (58cf8f9c3 2026-10-13)` and a selected
`.venv-native` Python 3.15.2. The following checks passed:

- `cargo fmt --all --check` (after formatting the new test file).
- Verified actionlint 1.7.9 on all three changed workflows, with no findings.
  Its official Linux amd64 release archive SHA-256 is
  `233b280d05e100837f4af1433c7b40a5dcb306e3aa68fb4f17f8a7f45a7df7b4`,
  verified against both the upstream release API and the downloaded bytes.
- `node --test tests/workflows/ci-gate-monitor.test.cjs`: 13 passed.
- The actual native-test workflow shell was extracted and executed against
  synthetic child commands. Cargo exits 0/27 and tee exits 0/9 propagated
  correctly in all four combinations; stdout/stderr and exit receipts survived
  both failure paths. An independent reviewer separately checked Cargo exits
  0/1/27. These exercise CI failure handling, not native scientific computation.
- `.venv-native/bin/python scripts/check_migration.py`: policy valid; all 38
  feature obligations still block release.
- `.venv-native/bin/python scripts/conductor_registry_hygiene.py --json`:
  no missing, stale, orphaned or wrong-target entries.
- `git diff --check`.

Two local native builds exited 101 because the `innovate-core` compiler process
was killed by SIGKILL before tests executed. The first selected the existing
`kairos`, `network`, `scientific_contract` and `model_oracles` integration targets
with `--locked -j 2`; the second also selected `metamorphic`, used `-j 1`, disabled
incremental compilation and set development/test codegen units to 256. No
native test pass or application defect is inferred from these resource failures.
Required hosted CI must execute the exact proposed source before acceptance.

Independent source review found no blocking mathematical or workflow defect;
review is not a substitute for execution. Native mutation, scientific validation,
the 38 migration obligations and release acceptance remain unqualified.
