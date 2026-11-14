# Implementation Plan

## Phase 1: Readiness and prerequisites

- [x] Inspect the current native repository and registry-specific prerequisites.
  Base: `276c2e505a0ce26910820fe8a94a6a498f505700`; native development version
  `0.7.0a1`, Apache-2.0, Rust/Polars with Python/Julia/R bindings. The release
  gate remains closed; metadata presence is not release or rights acceptance.
- [x] Reconcile GitHub ownership and hierarchy. Read-only REST inspection of
  `/repos/edithatogo/innovate/issues/398/sub_issues` confirmed native edges to
  open #399, #400 and #401. PR #402 is closed/unmerged. Concurrent PRs #590,
  #600, #602 and #603 own separate security/native/quality work.
- [x] Capture current repository-specific checks and their actual outcomes below.
  Earlier July completion markers are historical, not native-release evidence.

## Phase 2: Registry deliverables

- [x] Correct development citation metadata and the placeholder author ORCID;
  retain the existing author affiliation pending publication-specific confirmation.
- [x] Implement the current source-fingerprinted readiness record and offline
  integrity checker; move behavioural coverage into the active `tests/native`
  suite and replace the old string-only readiness test.
- [x] Prepare Software Heritage/RRID metadata, normalized read-only lookup
  receipts, official eligibility sources, and a JOSS assessment/content outline.
- [x] Replace the unsupported repository-ready claim with explicit preparation
  and external gates; expose the contract in the active documentation site.
- [ ] [Issue #399](https://github.com/edithatogo/innovate/issues/399): selected
  release, successful archival and resolving matching SWHID remain pending.
- [ ] [Issue #400](https://github.com/edithatogo/innovate/issues/400): duplicate
  check, submission, curator decision and resolving matching RRID remain pending.
- [ ] [Issue #401](https://github.com/edithatogo/innovate/issues/401): current
  qualified manuscript/software/research-use evidence and author declarations
  remain prerequisites before submission and editorial/publication evidence.

## Phase 3: Reconciliation and closeout

- [x] Run independent source review and apply the two identified fixes: reject
  query/fragment identifiers as resolver URLs, and reject negative SWH results
  while accepting the provider's structured successful resolver response.
- [x] Reconcile track metadata, current documentation and native issue hierarchy.
- [ ] Verify GitHub project placement with an accessible project-membership
  surface; this increment does not claim the project-visible acceptance criterion.
- [ ] Obtain final-head hosted CI evidence and the existing integrator's merge
  window. Do not race concurrent promotion or weaken required checks.
- [ ] Close registry deliverables only with authoritative provider evidence or
  an evidenced ineligibility decision; keep parent #398 open while those remain.
- [ ] Archive this track only after all automatable work and external gates have
  been reconciled. This increment does not complete the whole track.

## Verification receipt — 14 November 2026

Owned interpreter: `.venv-registry/bin/python`, observed CPython **3.15.2**.
Focused tools: pytest **9.2.0**, pytest-cov **7.1.0**, Ruff **0.17.4**,
coverage **7.17.3**, cffconvert **2.0.0**. This isolated environment does not
change the repository lockfile or another worker's environment.

| Command / check | Observed outcome |
| --- | --- |
| Initial `python -m pytest tests/native/test_registry_readiness.py --maxfail=1` before implementation | Failed because the new checker module did not yet exist. This is availability evidence, not a behavioural or mathematical regression. |
| `.venv-registry/bin/python -m pytest tests/native/test_registry_readiness.py` | **56 passed**. Independent reviewer repeated the same suite successfully and reproduced/rechecked both review findings. |
| Focused coverage with branch measurement, restricted to `scripts/check_registry_readiness.py` | **138/144 statements (95.8%)**, **24/30 branches (80.0%)**; 93.1% combined. Coverage belongs only to the metadata checker; it is not native/scientific coverage. The initial module-name selector collected no data and is not counted. |
| `.venv-registry/bin/ruff check scripts/check_registry_readiness.py tests/native/test_registry_readiness.py` | Passed after formatting. |
| `.venv-registry/bin/ruff format --check scripts/check_registry_readiness.py tests/native/test_registry_readiness.py` | Passed. |
| `.venv-registry/bin/cffconvert --validate -i CITATION.cff` | Valid against CFF **1.2.0**. |
| `.venv-registry/bin/python scripts/check_registry_readiness.py --json` | Exit **0**; record valid, recorded_complete false, online_verification_performed false; #399/#400/#401 pending. |
| Same checker with `--require-complete --json` | Exit **1**, intentionally preserving all three unresolved destinations. |
| `.venv-registry/bin/python scripts/conductor_registry_hygiene.py --json` | Passed; no missing links, unregistered/orphaned tracks, or wrong targets. |
| `.venv-registry/bin/python scripts/sync_versions.py --check` | Native development mapping valid; release remains gated. |
| `.venv-registry/bin/python scripts/check_migration.py` | Development policy valid; **38** scientific obligations remain. |
| Same migration checker with `--release` | Exit **1**, expected: scientific migration and acceptance are incomplete. |
| Local Markdown-link validation and `git diff --check` | Passed. |

The receipt and source revision for hosted validation belong in the PR. No full
native suite, installed-package acceptance, docs build, provider submission,
publication, author attestation, or external acceptance is inferred from these
local checks. Local Node is 24.22.1; the documentation build requires Node 26,
so the existing qualified hosted documentation workflow must supply that gate.
