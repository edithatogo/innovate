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
  `63158d51e40f180a34c68bdb614f318b13ffd2de765607871daa7db36a8de4fd`,
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
- Local execution used CPython 3.15.2 and Rust 1.99.1. The combined games suite
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

> **Dependencies:** `deny.toml` `[sources] allow-git = []` strict mode requires `rust_tabular_crates_io_20261115` and `rust_kairos_crates_io_20261115` to complete first (allow those two URLs temporarily in `deny.toml` if implementing earlier).

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
