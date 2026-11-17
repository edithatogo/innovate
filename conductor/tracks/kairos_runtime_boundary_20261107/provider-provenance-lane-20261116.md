# Kairos provider provenance correction

- Owner: Codex chat `01a1e3d8-fb04-7010-a038-2a5478662a5d`.
- Worktree: `/Users/doughnut/.codex/worktrees/innovate-kairos-revision-fix/innovate`.
- Branch: `codex/innovate-kairos-provider-provenance`; base
  `b29a1d27f54fdd08d77881ee999772fceab7c03a`.
- Bounded outcome: make the existing DES, ABM and network result metadata report
  the actual locked Kairos provider revision and prevent manifest/lock/runtime
  drift from silently recurring. This is the current-layout equivalent of the
  track's claim-boundary and candidate-identity work, not full track acceptance.
- Allowed files: `rust/crates/innovate-core/src/simulation.rs`,
  `tests/native/test_python_consumer.py` and this unique lane handoff.
  Preserve the independent checkpoint-compatibility namespace, manifests,
  dependency pins, shared plans, registries and other owners' work.
- Native scope: local Rust simulation and installed Python consumer only; no
  native provider modification, publication, release or external service writes.
- Next operation: hand the scoped commit to the existing integrator for
  exact-head preflight and hosted qualification; keep this track open.

## Finding and fix

The root Cargo manifest and lock resolve all six Kairos packages, including
transitive `kairo-ecs-state`, at
`a1f6283575ad26918230adc122a99b5c08fc4da3`. After dependency PR #657, the
shared result metadata still emitted the previous provider revision
`e76b97be2ea3712645933d903822ff6381c8883e` for completed DES, ABM and network
simulations. The source change corrects that existing metadata value; it changes
no dependency, simulation behavior, schema or checkpoint configuration.

The new installed-consumer regression reads the five direct dependency pins and
their six resolved Cargo.lock entries, checks the exact provider URL and immutable
revision, then runs DES, ABM and network simulations and compares each reported
revision with the lock. Its known counts also check that each real engine ran.
It is in the already-packaged Python consumer test file so the scheduled
source-distribution lane can execute it. A future switch from Git dependencies
to published crates needs an explicit corresponding identity-contract update.

The older `CHECKPOINT_PROVIDER` value
`d56863505d938feb069af88a737eb7668113d14b` is a compatibility namespace, not
the reported current provider. It remains untouched. A read-only provider
comparison between that revision and `a1f6283` found no changes in the six
consumed crates' `src` directories; the only changes within those crate paths
were four Cargo description lines. Source equality alone does not prove replay,
so the retained old-provider fixtures were also executed below.

## Verification on the bounded source

The owned interpreter was CPython 3.15.2; the actual Rust compiler and Cargo
were 1.99.1. Maturin 1.16.0 built locked Darwin arm64 development wheels in an
isolated external target, and the selected wheel was installed into a separate
external target directory for the repo-local interpreter. Neither wheel is a
release artifact or cross-platform qualification.

| Check | Observed result |
| --- | --- |
| New revision regression against the baseline wheel | **3 failed as expected**, one per engine: the reported `e76b97b` differed from the locked `a1f6283`; the preceding native count and engine checks passed. |
| New revision regression plus existing replay and unknown-engine tests against the corrected wheel | **5 passed**. |
| Retained previous-provider DES, ABM and network checkpoint fixtures against the corrected wheel | All **3** resumed results equaled uninterrupted execution and the retained historical result after changing only its expected reported revision. |
| Extracted source-distribution regression using the corrected local wheel | **3 passed**. The extracted test, simulation source and Cargo.lock matched the corresponding checkout bytes; a wheel was not built from the extraction. |
| Review of the lock-selection oracle with disposable mutated locks | An extra unrelated Kairos version was accepted; changing the selected transitive provider revision was rejected. The real Cargo.lock was never modified. |
| Focused Rust formatting, Ruff lint/format, repository policy and registry hygiene | Passed within their reported local scopes. Live settings and full release gates were not checked. |

Wheel SHA-256 values: baseline
`069f8d0842e89cba2faa162e2414ff7346eabd842bf31e7c55371b1a1c39ac44`;
corrected `e5e1ab5a3636c764be660ac05bfd8ede759858dbed360740b4b1535247d91915`.
The reviewed source distribution is
`24e72493557c5c7a8226da5a5bde6de2f3733fed6dd0a12ed229abf6656e46ae`.
The Cargo.lock SHA-256 is
`2fbada0a3db416e66a1fb338ce82c4e77a53cecfed5529c20b98cc3228f00a44`.
The corrected simulation source and consumer test SHA-256 values are
`fd3ef8614f5f303adbe8d6267c15ad9544fc4705cf901c1db762832754fcb94d` and
`25c28543f5896575ec26fe9b20a8796ca65f99447bb43dbeb496f50193469102`.
Actual build, red/green, prior-checkpoint and source-distribution logs and the
artifacts remain in `/Volumes/PortableSSD/innovate-kairos-provenance.oIt1cV/`.

The focused pytest invocation selected
`test_kairos_metadata_identifies_the_locked_provider`,
`test_native_kairos_replay_and_engine_errors` and
`test_kairos_tick_checkpoint_resume` in `tests/native/test_python_consumer.py`.
The source-distribution invocation selected only the first test. No broad Rust
suite, Julia/R consumer run, hosted check, provider publication or full track
acceptance is inferred from this increment.

## Review and remaining work

The source fix changes only the emitted revision; a separate manifest/lock/runtime
oracle catches future drift without rejecting an unrelated second provider version.
Existing test oracles and checkpoint identities were not weakened. The track's
imported plan still names retired `bindings/rust` and
Python paths; its original leaves must be reconciled with the current workspace.
The I10 installed-consumer/replay and integration gates remain open, and the
optional I07 game profile has separate unmet prerequisites. Do not archive from
this bounded result. The integrator owns the shared plan, registries, provider
pin promotion and final review; recheck main and the other lanes at integration.
