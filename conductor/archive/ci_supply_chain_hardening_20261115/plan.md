# Plan: CI & Release Supply-Chain Hardening

## Alert 44 isolated intake — 16 November 2026

Chat `01a1e448-3a28-7653-a681-cd3fdb41f498` owns the bounded candidate in
`/Volumes/PortableSSD/GitHub/innovate-security-alert-44`, branch
`codex/security-alert-44`, based on
`b29a1d27f54fdd08d77881ee999772fceab7c03a`. The allowed candidate paths are
the root and games Cargo manifests and locks, the core Cargo manifest,
`rust/vendor/`, `THIRD-PARTY-LICENSES`, `.gitattributes`, `Dockerfile`, `pyproject.toml`, `deny.toml`,
`osv-scanner.toml`, `.github/workflows/security-quality.yml`,
`.github/workflows/ci.yml`, `noxfile.py`, any narrowly required dependency
guard and its tests, the registry-readiness input record and its reassessment,
the source-derived Julia/R documentation inventory and generated reference
metadata affected by the CI workflow hash, the remote-worker closed-pipe test
needed for a deterministic native gate, and this plan. The existing repository integrator retains shared-manifest and
promotion authority; this isolated lane supplies a reviewable handoff. Other
owners and historical acceptance records remain unchanged. The next operation
is to qualify the paste-free source and distribution graphs and reconcile the
games pin with the accepted Paignia revision; this intake is not release or
scientific acceptance.

The formatting check now selects all five project workspace members without
traversing third-party path dependencies. Comparison of the derived Julia/R
inventory before refresh found only the `.github/workflows/ci.yml` source hash
changed; its Julia and R qualification commands and all structural fields are
unchanged. Refreshing that inventory does not assert fresh Julia/R execution
or validate any earlier receipt against the new hash.

The Python package still declares Apache-2.0 for project-owned code. The copied
upstream sources and notices include MIT, BSD and MPL terms; a complete review
of bundled dependency licences and the distribution-level SPDX expression is
still required before publication. This change does not qualify a release.

The first full native gate found `worker_reports_a_closed_response_pipe`
returning success in one run. The test closed the pipe reader only after
spawning the worker, leaving a window in which another concurrent spawn could
inherit a reader. The test now closes the reader before spawning, while retaining the
same expected error and exit code; no worker behavior is changed.

### Candidate evidence and remaining gates

- Paignia [PR #27](https://github.com/edithatogo/paignia/pull/27) passed its
  stable, Rust 1.75 and privacy checks at
  `6f27257400fb25f44603883615f30447fbc5d4e4`. Its receiving commit is
  `ad3e1932469fcdfe3c13a5182a4e79b4448e4fbf`; the two trees are identical.
  The games manifest and lock name the receiving commit. The same three checks
  passed on receiving main in runs `43878846017` and `43878846026`.
- Root, games and vendored-audit lock SHA-256 values respectively are
  `642a7176be4ac9fae0c633e468abfd5decdd2531c594fcdf8f3d4029c8784342`,
  `307c1c18b5d4e0beee7b9cd4d575c320045b4f71287b8fbfe1976f269d23750e`, and
  `b72b16b86c389ffb59e79044414b3501d8cfbefcae8fd6b559b17d75f0e20e6b`.
  Locked all-features metadata found `pastey 0.2.3` and no `paste` in either
  active graph. Unfiltered OSV scans of both locks and the 16-crate upstream
  audit inventory returned no findings.
- The independent review identified that ordinary RustSec checks skip path
  crates. `scripts/check_vendor_audit.py` verifies the complete upstream
  inventory and its registry-identity audit lock; the security workflow audits
  that lock and the games lock as well as the root. A seeded `paste` dependency
  failed the cargo-deny ban, and a seeded `nalgebra 0.27.0` audit entry produced
  `RUSTSEC-2021-0070`. Four inventory and licence-bundle tests passed.
- Local execution used CPython 3.15.2; the follow-up below explicitly verifies
  the Rust toolchain. The combined games suite
  passed three tests on the merged provider revision. A fresh Git-consumer
  prototype passed its closed-form control, and a wheel built from an extracted
  source distribution passed five focused NUTS and multi-generation tests.
  The development lint, types, documentation, package and security sessions
  passed. The first full Python run had 373 passes, five capability-based skips
  and a stale registry digest failure; after the documented reassessment, that
  exact failed test passed. The first `nox -s native` run failed the closed-pipe
  case described above; its five-test target passed after repair, followed by a
  successful full `nox -s native` run with one live-OECD test ignored. The
  warning-denied `cargo doc --locked --no-deps -p innovate-core -p
  innovate-tabular -p innovate-ffi` build also passed. A subsequent full
  Python run passed 374 tests with the same five capability-based skips.
- The initial wheel-content check rejected duplicate identical upstream
  licence files. The source distribution now retains each original file; the
  wheel carries a generated bundle that maps each source notice to its exact
  text, grouping only identical bytes. `check-wheel-contents` and `twine check`
  passed for the rebuilt wheel and source distribution, and five installed
  consumer tests passed. Bundle-drift tests preserve this mapping.
- Container review found that the source build context omitted the newly
  required notice bundle. The Dockerfile now copies it into both build stages
  and the remote-worker image; the scheduled container check includes and
  verifies it. No Docker client or running qualified container service was
  available locally, so container build and execution remain unverified here.
  The resulting CI workflow change alters only the archived container inputs
  and notice assertion; a structural comparison found no Julia/R inventory
  change other than the workflow hash before refreshing that inventory.
- Local actionlint 1.7.12 and the checksum-verified CI-pinned 1.7.9 both consumed
  CPU without producing a result and were interrupted after approximately two
  minutes and one minute respectively. Neither is a pass. Workflow tests passed;
  offline zizmor reported no unsuppressed findings. Hosted workflow lint on the
  exact candidate remains required. Registry publication remains blocked by
  `innovate-core`'s `publish = false` until paste-free published dependencies
  exist; release, multi-platform and scientific acceptance remain separate.
- On the first PR head, hosted workflow lint passed. Privacy run `43881472468`
  failed after Gitleaks reported zero findings; the privacy scanner emitted no
  result before the step exited, so its cause is not established. Release
  readiness run `43881472447` failed on the 38 open scientific obligations;
  the same gate failed on base `b29a1d27` in run `43864696589`. Neither is
  waived. Repeat required checks on the final head and preserve the release hold.

### Main-base and privacy follow-up — 17 November 2026

Main advanced to `cc2e4da4af9c848043b31f1191c9d96212048276` with no
overlapping paths. The read-only integration preflight compared it to candidate
`68707bddcaee4abc47f3faa9dcfbfad06490a6b2`: both worktrees were clean, with
zero overlaps, staged deletions, missing files or warnings. The app-managed
worktree request did not return a result; the verified repository's standard Git
fallback is the detached, read-only target
`/Volumes/PortableSSD/GitHub/innovate-alert44-main-preflight`. It is not an
app-managed checkout. This source branch incorporates the newer main commit;
promotion remains with the existing integrator.

Privacy run `43883637179` on the second PR head scanned all 770 selected files
without skips. Gitleaks reported no findings; Presidio reported three blocking
credit-card-pattern matches in public upstream complex-number test data.
Those fixture bytes match the published upstream crate, but the findings have
not been admitted through the shared scanner's exact nonidentifier review.
No scanner exclusion or finding waiver is applied. The hosted privacy action
also selected Python 3.12.16, whereas this repository requires Python 3.15.2;
the current [Presidio analyzer 2.2.366 metadata](https://pypi.org/project/presidio-analyzer/2.2.366/)
declares Python below 3.15. A qualified 3.15-compatible privacy route and
review of the exact nonidentifiers remain open. Do not count this job as a pass.

### Vendored sort and hosted follow-up — 17 November 2026

On candidate `25f6a76db7ca6274ff191a1a1a08d13c847aa771`, the PR's native
matrix, Julia/R matrix, package, RustSec and workflow-policy jobs passed. The
Rust CodeQL analysis completed, but its aggregate check failed on four new
high-severity invalid-pointer alerts (`#52`–`#55`) in the vendored faer sort.
Each points back to one of the empty cursor's null initializers. The PR privacy
scan covered 759 of 759 selected files and again reported three blocking matches;
release readiness retained the 38 existing scientific obligations. The offline
security jobs remained queued. None of these failures or queues was waived.

The exact-head manual CI run `43887883380` passed all jobs, including the
clean source-distribution consumer and remote-worker container build and runtime
checks. That receipt applies to `25f6a76d`, before the sort follow-up below.

The candidate initializes the two empty block-sort cursors directly from their
offset arrays. Previously, each null cursor was reset to its array in the loop
before it could be dereferenced. This changes the initializer and declaration
order only; the sort algorithm and comparisons are unchanged. `UPSTREAM.json`
records both source hashes, and `scripts/check_vendor_audit.py` verifies the
patched hash. A differential harness using Rust 1.99.1 compared upstream and
candidate output for 1,840 inputs and produced the same digest; each output was
also checked against independently sorted keys and the input value permutation.
The library-level `cargo test --offline -p innovate-core --test
vendored_faer_sort` passed its block-boundary and invalid-range tests before
the change; the `--locked --offline` rerun passed afterward. Five Python
inventory tests and repository policy passed. `nox -s security` passed its
three RustSec audits and inventory checks; `cargo deny --all-features --locked
check` and the games `check bans` both passed with existing duplicate warnings.
The full `nox -s native` session also passed with the host Rust 1.100.0
compiler, with its one live-OECD test ignored; `nox -s lint` passed. The
required rerun explicitly selected and verified both Cargo 1.99.1 and rustc
1.99.1 before running `nox -s native`; it passed with the same one ignored
live-OECD test.
The root and games all-features metadata remain paste-free. The first offline
games metadata attempt lacked the Paignia checkout; the locked online retry
passed. The root lock gained only the existing faer package as a test dependency;
all 285 package identities are unchanged, so the prior unfiltered OSV result
for that package set remains applicable. CodeQL and all relevant hosted gates
must run again on the new candidate; the findings have not been dismissed.

A fresh external Git consumer of `364d69e52a8e8ce04be05f796968128738c4d9ef`
resolved 197 packages with Cargo and rustc both verified at 1.99.1. Its lock
contains `pastey 0.2.3` and no `paste`; `cargo-audit` found no advisories. Its
closed-form Bass control passed and an invalid time was rejected. The same
consumer control also passed with the host Rust 1.100.0 compiler.

### Privacy receiving candidate — 17 November 2026

Chat `01a1e448-3a28-7653-a681-cd3fdb41f498` owns this isolated PR branch's
privacy receiving candidate. The allowed files are the privacy workflow and
this evidence plan; shared main integration and Innovate promotion remain with
their existing integrators. Native and authentication scope: none.

Shared standards draft PR #151 adds three exact reviews for the public matrix
fixture, with no change to scanner rules. Its candidate `cf6a3451c6828248bf99c1f53d35a246f54a4b9a`
passed hosted policy, privacy, real scanner, canary, CodeQL and workflow lint
checks. Independent review verified the three exact blob identities, the
complex-number source and real-Presidio blocking controls, including a changed
blob and a neighboring unreviewed value. The complete local Innovate changed-file
scan covered 760 of 760 files, with three reviewed matches and no blocking
privacy or secret findings; 1,145 advisory matches remained. The privacy workflow
here temporarily pinned that draft candidate solely to obtain an actual receiving
PR scan. The receiving privacy job passed on GitHub's synthetic merge of Innovate
head `3fb0cd6fa5275ff7f894dfdc8ad779f65792e35b` into base `cc2e4da4` using
Python 3.15.2; Gitleaks reported zero findings, while the hosted log did not
expose the Presidio JSON summary. A separate exact-head local scan covered 761
of 761 changed files, with three reviewed matches, zero blocking, 1,145 advisory
and no skips. Its value-free receipt SHA-256 is
`ea65a61e59e0c99a0c3ac47a487abbce3586ad1762f41c0e0fd53677fe36b537`.

At the user's direction, chat `01a1e448-3a28-7653-a681-cd3fdb41f498` owns the
bounded integration of shared PR #151 and Innovate PR #674 in the worktrees
recorded here and in the shared track; earlier integrators retain unrelated
work. Shared PR #151 merged as `c82b2658f6874127d05f825dbe9ed4747f146efd`,
whose tree is byte-identical to its tested head `4265b912`. This candidate pins
that receiving revision; post-merge shared checks and a fresh Innovate receiving
scan remain required. Release readiness and the offline security runner remain
separate open gates.

> **Dependencies:** `deny.toml` `[sources] allow-git = []` strict mode requires `rust_tabular_crates_io_20261115` and `rust_kairos_crates_io_20261115` to complete first (allow those two URLs temporarily in `deny.toml` if implementing earlier).

### Documentation and native diagnostic privacy — 17 November 2026

Chat `01a1e448-3a28-7653-a681-cd3fdb41f498` owns a bounded follow-up in
`/Volumes/PortableSSD/GitHub/innovate-security-alert-44`, based on
`fab4343b009e933ada4ed6460b20c336d1e5beed`. Allowed paths are the API
generator wrapper and its Node tests, the production documentation verifier and
its contract tests, the native CLI and its CLI contract tests, the documentation
ADR, and this plan. Native/authentication scope: none. The next operation is to
protect interpreter launch errors, artifact URL diagnostics and rejected native
request operations while preserving their checks, useful failure information,
and successful behavior. The scanner's coverage, image review and scientific
release acceptance remain separate open gates.

The documentation launcher now reports fixed launch diagnostics instead of the
configured interpreter path. Artifact receipt schema 3 fingerprints untrusted
URL diagnostics while retaining error codes, source routes and failure status;
invalid URLs and valueless canonical attributes also replace any old pass with a
failure receipt. Rejected native operations show the expected operation without
echoing the supplied request value. Independent review found the invalid-URL and
valueless-attribute cases; both were reproduced and covered by regression tests.

With repo-local Python 3.15.2, the 25 production-documentation contract tests
passed in both human and JSON receipt modes. All 48 Node documentation tests,
the actual API-generation `--check`, targeted Ruff and rustfmt checks, and the
locked offline Rust 1.99.1 `innovate-core` native CLI contract test passed. The
local repository-policy check passed within its documented scope. The strict
offline scanner and its parser coverage are not yet qualified; neither these
checks nor this diagnostic change establishes release readiness.

## Phase 1: `cargo-deny` & `poutine` in Security Quality Workflow
- [x] **1.1 Create `deny.toml` at repo root:** Configure `[advisories]`, `[licenses]` (SPDX allowlist), `[bans]` (duplicate/wildcards), and `[sources]` (`unknown-registry = "deny"`, `unknown-git = "deny"`). Run `cargo deny check` locally (or via `taiki-e/install-action` in CI) and verify baseline pass.
- [x] **1.2 Wire `cargo-deny` into `.github/workflows/security-quality.yml`:** Add a `cargo-deny` step alongside `cargo-audit` using SHA-pinned `EmbarkStudios/cargo-deny-action` or `taiki-e/install-action`.
- [x] **1.3 Add `poutine` workflow supply-chain check:** Add a SHA-pinned `poutine` scan step in `.github/workflows/security-quality.yml` alongside `zizmor`.
- [x] **1.4 Verify workflow lint & policy tests:** Run `uvx zizmor .github/workflows/security-quality.yml`, `python3 scripts/check_repository_policy.py`, and `pytest -o minversion=9.0 tests/workflows/test_repository_policy.py`.
- [x] **1.5 Conductor Review (Phase 1):** Run `conductor-review` on Phase 1 changes, apply any fixes, re-verify, and commit checkpoint.

## Phase 2: `Harden-Runner` on Publish Workflows
- [x] **2.1 Audit outbound endpoints for publish workflows:** Inspect `.github/workflows/pypi-publish.yml`, `crates-publish.yml`, `bindings-publish.yml`, and `conda-publish.yml` and list required allowed endpoints (GitHub, PyPI, crates.io, conda/anaconda, rustup).
- [x] **2.2 Add SHA-pinned `step-security/harden-runner`:** Add `harden-runner` (`egress-policy: audit` initially, then `block` with `allowed-endpoints` once verified) as the first step in each publish job.
- [x] **2.3 Verify workflow lint & release tests:** Run `uvx zizmor .github/workflows/*.yml` and `pytest -o minversion=9.0 tests/workflows/test_release_workflows.py`.
- [x] **2.4 Conductor Review (Phase 2):** Run `conductor-review` on Phase 2 changes, apply any fixes, re-verify, and commit checkpoint.

## Phase 3: Sigstore / in-toto / SLSA Provenance & Syft / Grype SBOM
- [x] **3.1 Add `actions/attest-build-provenance` (Sigstore + in-toto + SLSA):** Wire SHA-pinned `actions/attest-build-provenance` in `.github/workflows/bindings-publish.yml` (with `attestations: write` and `id-token: write` job permissions) for built FFI shared-library archives, and document `gh attestation verify` / `slsa-verifier` verification in release docs.
- [x] **3.2 Add `Syft` SBOM generation + `Grype` scan:** Add SHA-pinned `anchore/sbom-action` (`Syft`) and `anchore/scan-action` (`Grype`) in `.github/workflows/bindings-publish.yml` / release packaging, uploading SPDX/CycloneDX SBOMs as release artifacts.
- [x] **3.3 Run full workflow & policy verification:** Execute `uvx zizmor .github/workflows/*.yml`, `python3 scripts/check_repository_policy.py`, `pytest -o minversion=9.0 tests/workflows/test_release_workflows.py`, and `pytest -o minversion=9.0 tests/workflows/test_repository_policy.py`.
- [x] **3.4 Conductor Review (Phase 3 & Track Sign-off):** Run `conductor-review`, apply any fixes, re-verify, and mark track complete in `conductor/tracks.md`.

### Current-main reconciliation after PR 672 — 17 November 2026

Chat `01a1e448-3a28-7653-a681-cd3fdb41f498` owns the receiving PR 674
worktree `/Volumes/PortableSSD/GitHub/innovate-security-alert-44`, starting at
`ce7bd2acad48c1c41fc86584474bf93d1d29572b`. The bounded operation is to
integrate main `4b3d5cbb1c52b378ae2dbdd1c554d351e12f69a8`, reconcile its
generated documentation inventory, and record the resulting checks here. The
merge brings the committed main changes; manual edits are limited to the
generated inventory and this record. Native and authentication scope: none.
The scientific release and offline security gates remain open.

The read-only preflight compared clean worktrees and the merged main tree to
PR 672's tested tree; the tree IDs were identical. Git's merge preview found one
content conflict: the generated hash of `.github/workflows/ci.yml`. Regenerating
the inventory on the merged sources resolved it. With repo-local Python 3.15.2,
the inventory check, repository-policy check and pre-commit configuration
validation passed. The repository-policy, release-workflow and polyglot inventory
suites passed 19, 11 and 38 tests respectively; all 16 Node workflow tests
passed. Actionlint and Zizmor passed, with Zizmor's existing two ignored and 61
suppressed findings unchanged. These focused checks do not establish hosted
exact-head, offline-scanner or scientific release acceptance.

The post-merge main Security and Quality run `44066589564` reported a failing
RustSec advisory audit for unmaintained `chrono` (`RUSTSEC-2026-0543`) and
`chrono-tz` (`RUSTSEC-2026-0555`). These are separate from the `paste` fix and
must be rechecked on the PR head; no advisory exception has been added.

### Alert 44 temporal compatibility controls — 17 November 2026

Chat `01a1e448-3a28-7653-a681-cd3fdb41f498` owns this bounded candidate in
`/Users/doughnut/.codex/worktrees/innovate-pr672-security-pin/innovate`, based
on main `b74de7e678966865ea91fe00c5fe0eb15d31c20f`. Allowed files are
`rust/crates/innovate-core/tests/temporal_compatibility.rs` and this record.
Native/authentication scope: none. The next operation is to add independent
public-dispatch controls for expanded Gregorian years and timestamp handling,
using fixed expected results and malformed-input cases. These are migration
safety tests only; they do not change dependencies, waive advisories, qualify
Polars, or resolve the alert.

On 17 November the same owner extended this bounded test scope to the public
scenario-artifact import timestamp contract. The allowed files remain the test
file and this plan. The artifact stores the accepted timestamp unchanged; the
tests cover offset, fractional and leap-second forms plus invalid dates, seconds
and missing offsets. This is a migration guard, not a dependency replacement or
an assertion that the alert is fixed.

The Scorecard instance on main `b74de7e6` remains open for the informational
unmaintained advisories `RUSTSEC-2026-0543` (`chrono`) and `RUSTSEC-2026-0555`
(`chrono-tz`); both list all versions as affected and no patched version.
`cargo tree --locked --offline` confirms `innovate-core` directly uses Chrono,
while the Polars tabular dependency graph reaches both crates even with default
features disabled. A direct-only replacement therefore cannot clear the alert.
The [upstream Polars Jiff draft](https://github.com/pola-rs/polars/pull/29110)
is unmerged, based on 0.55.1 rather than the locked 0.55.2, and documents
parsing, display and date-range changes. A root Cargo patch or relabelled vendor
source would not establish a consumer-safe, advisory-free replacement.

The new public-dispatch controls use fixed Gregorian/RFC 3339 expectations for
expanded-year leap days and the existing maximum date, UTC day crossings,
fractional weekly boundaries, leap-second labels and naive/offset rejection.
An independent read-only review identified an over-restrictive test for the old
maximum and missing fractional-anchor coverage; the candidate now rejects an
impossible date instead and checks the microsecond boundary.

With Rust/Cargo 1.99.1 selected explicitly and a per-worktree target directory,
`cargo test --locked --offline -p innovate-core --test temporal_compatibility -j 2`
passed five tests. The existing native-boundary calendar selection passed three,
the offset-week selection passed one, and the internal calendar selection passed
two. `cargo fmt --check`, the local repository-policy check and Conductor registry
hygiene passed. An extra `cargo fmt --all -- --check` failed on unchanged vendored
third-party sources; it is not the repository CI formatting command. No source
or dependency has been migrated, the alert is not closed, and these checks do
not qualify all platforms, consumers or release readiness.

The later artifact-import guard passed with the five earlier controls: six tests
passed using Rust/Cargo 1.99.1 and `cargo test --locked --offline -p
innovate-core --test temporal_compatibility -j 2` in this worktree's isolated
target. The actual repository `cargo fmt --check` and repo-local Python policy
check also passed. The guard checks the decoded accepted timestamp string and
rejection of four malformed inputs; it does not itself migrate or qualify a
temporal library. An independent read-only reviewer found no blocking issues in
the test, while noting that the comparison is not original JSON wire-byte proof.
