# Bounded plan

All implementation leaves are pending. One behaviour and no more than five files
per leaf. Use the existing scientific-alignment integrator for shared source/pin
handoffs. Commands run from repository root with qualified tools; two commands
shown together are independent checks. A named future test is an explicit output,
not an assertion that the target exists today. Stop on missing contracts, toolchain
or ambiguous scientific assumptions; return a reproducer instead of guessing.

Leaf IDs are stable labels, not execution order. The explicit `Requires` fields
define the dependency graph: I01 precedes I00, then I02. I00 was inserted after
review to prepare the candidate and lock without renumbering the existing leaves.
Do not infer dependencies from numbering or document order.

The I10 receipt covers the native runtime consumer only. Optional game profile
I07 stays pending independently and does not block that receipt or a release
that claims no game capability. Game integration follows the existing
[PR570 G01–G04/A08 programme](https://github.com/edithatogo/innovate/pull/570)
and the game provider's T17 handoff; this track does not take over that branch.
A game claim needs its accepted profile and real solver-consumer evidence, with
native replay additionally required for dynamic execution. Static game acceptance
does not require the dynamic adapter. Keep the whole track open while I07 is
unfinished; a scoped runtime receipt must not mark optional game work complete.

## I01: Correct the native claim boundary

- [ ] I01 Implement and verify this bounded outcome.

Executor: specialist. Requires: none.

Files: `src/innovate/abm/kairos_contract.py`, `tests/unit/test_kairos_adapter_contract.py`.

Check: `python -m pytest tests/unit/test_kairos_adapter_contract.py -q`.

Acceptance: Source/revision/example filenames alone cannot make claims_kairos_backed_simulation true. Keep dependency-inclusion metadata distinct from actual executed native capability; coordinate with PR570 before editing.

## I00: Prepare the isolated candidate and lock

- [ ] I00 Bind the candidate dependency before locked checks.

Executor: specialist. Requires: I01 and current manifest-owner handoff.

Files: `bindings/rust/Cargo.toml`, `bindings/rust/Cargo.lock`, `conductor/tracks/kairos_runtime_boundary_20261107/candidate-selection.md`.

Checks: `cargo generate-lockfile --manifest-path bindings/rust/Cargo.toml`; `cargo metadata --locked --manifest-path bindings/rust/Cargo.toml --format-version 1`.

Acceptance: Work in an isolated consumer checkout. Select an explicit immutable candidate (initially the existing pin if appropriate), generate and review the previously absent lockfile, and record resolved provider SHA/features. I02–I09 must run against this configuration. A changed candidate remains on this branch until qualified; final promotion merges the already-tested manifest/lock, never an untested later pin edit.

## I02: Select an immutable provider and public API mapping

- [ ] I02 Implement and verify this bounded outcome.

Executor: specialist. Requires: I00 and I01.

Files: `conductor/tracks/kairos_runtime_boundary_20261107/provider-map.md`.

Check: `cargo tree --locked --manifest-path bindings/rust/Cargo.toml --edges normal`.

Acceptance: Inventory actual scheduler/entity/RNG APIs at the candidate and current pin, supported targets and consumed features. Do not invent imported types from the print-only examples. Record an upgrade/retain decision without changing pins in this leaf.

## I03: Execute a real DES example

- [ ] I03 Implement and verify this bounded outcome.

Executor: local. Requires: I02 and accepted provider APIs.

Files: `bindings/rust/examples/kairos_des_smoke.rs`.

Check: `cargo run --locked --manifest-path bindings/rust/Cargo.toml --example kairos_des_smoke`.

Acceptance: Schedule two events with a deliberate tie/cancel case through Rust, assert actual order/state and return nonzero on mismatch. Printed import strings or a successful build cannot satisfy acceptance.

## I04: Execute a real ABM state and RNG example

- [ ] I04 Implement and verify this bounded outcome.

Executor: local. Requires: I02 and accepted provider APIs.

Files: `bindings/rust/examples/kairos_abm_smoke.rs`.

Check: `cargo run --locked --manifest-path bindings/rust/Cargo.toml --example kairos_abm_smoke`.

Acceptance: Create an entity, mutate/read a component and consume a versioned Rust stream; assert deterministic identity/state/output and explicit missing-capability failure.

## I05: Specify one thin native model dispatch operation

- [ ] I05 Implement and verify this bounded outcome.

Executor: specialist. Requires: I03 and I04.

Files: `conductor/tracks/kairos_runtime_boundary_20261107/native-dispatch-contract.md`.

Check: `cargo test --locked --manifest-path bindings/rust/Cargo.toml`.

Acceptance: Name one model operation, native ABI or Rust symbols, ownership/errors and output versions. Emit separate <=5-file bridge and installed-consumer test leaves; host code only marshals/orchestrates. Do not implement an entire diffusion engine from this design task.

## I06: Make one heavy data dependency optional

- [ ] I06 Implement and verify this bounded outcome.

Executor: specialist. Requires: I02 and active manifest-owner handoff.

Files: `bindings/rust/Cargo.toml`, `bindings/rust/Cargo.lock`, `bindings/rust/src/lib.rs`, `bindings/rust/tests/minimal_kairos_features.rs`.

Check: `cargo test --locked --manifest-path bindings/rust/Cargo.toml --no-default-features --test minimal_kairos_features`.

Acceptance: New target: selected core model compiles/runs without Arrow when unused, and explicit data feature preserves the existing interchange. Qualify resolved normal/build closure; no blind dependency deletion.

## I07: Qualify the optional domain-game profile

- [ ] I07 Implement and verify this bounded outcome.

Executor: specialist. Requires: UOGTO executable profile and generic Rust game package.

Files: `specs/ecosystem/uogto/innovate-game-profile.md`.

Check: `python scripts/conductor_registry_hygiene.py --json`.

Acceptance: Declare diffusion/adoption players/actions/information/payoffs and model assumptions, map UOGTO IDs and solver capability/results. No generic solver in Innovate and no mandatory Kairos dependency for static games. Registry hygiene is structural only; require a real game fixture in follow-on leaves. This optional profile has its own game-acceptance scope and is not an I10 runtime prerequisite.

## I08: Add a distinguishing paired-outcome regression

- [ ] I08 Implement and verify this bounded outcome.

Executor: local. Requires: existing accepted uncertainty schema and I05.

Files: `tests/unit/test_kairos_outcome_pairing.py`, `specs/ecosystem/heoml/extensions/innovate/README.md`.

Check: `python -m pytest tests/unit/test_kairos_outcome_pairing.py -q`.

Acceptance: New target specifies strategy/draw/parameter identity, time units and provenance and rejects missing/reordered pairs and incompatible perspectives. An expected baseline failure is a valid defect reproducer, not accepted runtime behaviour. I08b owns the bounded exporter fix and must integrate jointly before this target becomes a passing required gate. Innovate exports adoption outcomes; the HEOR producer owns values and Voiage owns VOI.

## I08b: Implement the paired-outcome validation in Rust

- [ ] I08b Complete the correction identified by I08.

Executor: local after the I05/I08 schema and oracle are settled. Requires: I08's settled oracle/reproducer, not a passing future implementation, and I05's native dispatch interface.

Files: `bindings/rust/src/outcome_contract.rs`, `bindings/rust/src/lib.rs`, `bindings/rust/tests/kairos_outcome_pairing.rs`, `tests/unit/test_kairos_outcome_pairing.py`.

Checks: `cargo test --locked --manifest-path bindings/rust/Cargo.toml --test kairos_outcome_pairing`; `python -m pytest tests/unit/test_kairos_outcome_pairing.py -q`.

Acceptance: Add the declared Rust module/test target for the one agreed paired-outcome profile; preserve the numeric payload and validate identity/units/value metadata. Host tests exercise the qualified Rust adapter or test marshalling only with that scope labelled; they cannot replace native validation. The existing diffusion_v1 fixture is an input contract, not an already shipped exporter. Make the positive/negative cases pass and integrate I08's red reproducer with this fix. Do not create a second simulation engine or a broad generic export framework.

## I09: Execute native replay and fail-closed installed acceptance

- [ ] I09 Implement and verify this bounded outcome.

Executor: specialist. Requires: I05 implemented leaves, I06 and I08b.

Files: `bindings/rust/tests/kairos_replay.rs`, `tests/unit/test_kairos_installed_native.py`.

Check: `cargo test --locked --manifest-path bindings/rust/Cargo.toml --test kairos_replay; python -m pytest tests/unit/test_kairos_installed_native.py -q`.

Acceptance: New targets: real installed Rust dispatch outside checkout, seeded replay, changed-action effect and missing/wrong native provider errors. Python reference results cannot substitute. Capture both consumer/provider identities once.

## I10: Review the coherent consumer contract and pin decision

- [ ] I10 Implement and verify this bounded outcome.

Executor: specialist. Requires: I00, I01, I02, I03, I04, I05, I06, I08, I08b and I09.

Files: `conductor/tracks/kairos_runtime_boundary_20261107/acceptance.md`.

Check: `python scripts/conductor_registry_hygiene.py --json; cargo test --locked --manifest-path bindings/rust/Cargo.toml`.

Acceptance: One contract/native integration review of the runtime-only scope; record actual checks and unsupported scopes, including the separate status of I07. Promote only the exact I00 candidate manifest/lock already exercised by the consumer gates after acceptance; no release, game acceptance or broader scientific promotion. Optional game availability is not a prerequisite for this receipt.

## Proportional closeout

The plan is the one detailed work record. Routine commits carry the diff and
focused result; no separate per-leaf git note, approval or evidence dossier is
needed for this track. Review the numerical/public/replay boundary and coherent
installed-consumer milestone once. Preserve required repository/hosted checks
and the original scientific completion obligations; structural checks cannot
certify runtime capability. Larger unresolved implementation tasks are split
into bounded leaves before local-model dispatch.
