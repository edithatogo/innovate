# I01 native claim boundary: receiving lane

- Owner: Codex chat `01a1e897-d8a6-74d0-858a-06d6048ebdef`.
- Worktree: `/Users/doughnut/.codex/worktrees/innovate-kairos-i01/innovate`.
- Branch: `codex/kairos-i01-native-claim-20261117`; base:
  `cc2e4da4af9c848043b31f1191c9d96212048276`.
- Scope: reconcile the imported I01 boundary against current native dispatch,
  receive the prior provider-provenance correction, and verify that declared
  capability and executed simulation remain distinct.
- Allowed files: `rust/crates/innovate-core/src/simulation.rs`,
  `tests/native/test_python_consumer.py`, the preserved earlier lane receipt,
  and this receiving-lane receipt. The existing integrator retains the shared
  track plan and registry, manifests, dependency pins, promotion and hosted gates.
- Native scope: local Rust simulation and installed Python consumer only. No
  provider, external service, authentication, publication or release mutation.
- Coordination: the existing integration owner in chat
  `01a1e448-3a28-7653-a681-cd3fdb41f498` was notified of the scoped packet and
  requested to flag an overlap. This records a handoff, not an approval or an
  assertion of independent review.
- Next operation: verify the received source and installed-consumer checks, then
  hand the exact commit and this plan reconciliation to the integrator.

## Reconciliation for the owning plan

The current I01 plan refers to `src/innovate/abm/kairos_contract.py`,
`tests/unit/test_kairos_adapter_contract.py`, and
`claims_kairos_backed_simulation`. Those are historical: the production adapter
and its boolean are absent from this Rust-native checkout. Recreating them would
not test the current claim boundary. PR #570 has already merged; its broader
scientific and game-acceptance work remains separate.

The present boundary is `KernelBinding::simulate_model` and the Rust simulation
and network modules. Discovery lists supported engines; it is a capability
declaration, not a receipt that a particular request ran. A completed native
Bass simulation returns the engine, resolved provider revision and dispatch
count in its result metadata. A valid run can dispatch zero events. Unsupported
engines must fail rather than silently become a deterministic curve. Python,
Julia and R marshal the native contract; their availability is not independent
scientific qualification.

For integration into the shared plan, replace the I01 file/check references
with `rust/crates/innovate-core/src/simulation.rs` and the selected tests in
`tests/native/test_python_consumer.py`. Preserve the original acceptance intent:
declared inputs, filenames and dependency metadata alone cannot establish
execution; actual engine results identify the selected provider and unsupported
requests fail closed. Check the manifest and lock against all three engine
results, and retain the zero-dispatch case. Record I01 as a scoped candidate
until the integrator accepts its exact source and required hosted evidence.

The old I00–I10 file paths also need individual reconciliation against the root
workspace. Existing scientific-alignment receipts show bounded native examples
and bridge work; they do not automatically close those leaves, change their
dependency order, accept I07's optional game profile, or establish release
readiness. This lane does not edit those shared leaves.

## Received correction

The earlier lane's commit `b2d9bb7704d45531f7d7b6adcb018672497009cd` was
received as `44e33d06` after a read-only, exact-range integration preflight.
Its earlier ownership, red/green evidence, source and wheel hashes, and limits
remain in [provider-provenance-lane-20261116.md](provider-provenance-lane-20261116.md).
The preflight reported no dirty overlap, staged deletion or missing target file.
It did not itself verify executable behavior.

## Receiving-lane verification

The selected worktree-local interpreter reported CPython **3.15.2**; Cargo and
rustc were **1.99.1**. Tests used this lane's own environment and target
directory. The following results were observed on 17 November 2026:

| Check | Result |
| --- | --- |
| Original provenance test against the retained baseline wheel, using the exact source-packet test file | **3 failed as expected**: DES, ABM and network each reported `e76b97b` instead of the locked `a1f6283`. |
| Selected installed-Python tests against the retained corrected wheel | **6 passed**: three provider-identity cases, capability-versus-execution, unknown-engine/replay and checkpoint-resume cases. The new boundary case supplies a forged declaration and a valid zero-dispatch run. |
| `cargo test --locked --offline -p innovate-core --test kairos -j 2` | **17 passed**, including native scheduler, ABM and previous-provider checkpoint tests. |
| `cargo test --locked --offline -p innovate-core --test network -j 2` | **37 passed**, including native network replay and resource cases. |
| Ruff check and format check on the changed consumer test; `cargo fmt --all -- --check`; `git diff --check` | Passed. |
| `scripts/check_repository_policy.py` and `scripts/conductor_registry_hygiene.py --json` | Passed within their local scopes; live settings were not checked. |

The Cargo commands used `CARGO_PROFILE_TEST_DEBUG=0`,
`CARGO_PROFILE_DEV_DEBUG=0` and `CARGO_INCREMENTAL=0` with the worktree-local
`target` directory. The Python regression used the previously built corrected
wheel, not a wheel rebuilt from this receiving worktree. Its SHA-256 was
`e5e1ab5a3636c764be660ac05bfd8ede759858dbed360740b4b1535247d91915`;
the retained baseline wheel was
`069f8d0842e89cba2faa162e2414ff7346eabd842bf31e7c55371b1a1c39ac44`.
The earlier source base and receiving base have identical Rust, binding,
manifest and lock inputs. The received corrected simulation source SHA-256
is `fd3ef8614f5f303adbe8d6267c15ad9544fc4705cf901c1db762832754fcb94d`;
the lock SHA-256 is
`2fbada0a3db416e66a1fb338ce82c4e77a53cecfed5529c20b98cc3228f00a44`.
The new consumer-test file SHA-256 is
`8d5e1f261b566a76559d486bc773f75b1e851d4d0cec9711fda85ac7a6b5058a`.

The local result establishes this bounded implementation and regression
evidence. It does not establish a newly built exact-commit wheel, other
platforms or bindings, hosted checks, broader scientific qualification, I10
acceptance or release readiness. The integrator owns the shared-plan edit and
the decision to promote the packet.
