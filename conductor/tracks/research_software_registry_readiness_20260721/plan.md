# Implementation Plan

## Executable preparation follow-up — 16 November 2026

The user's follow-up authorises implementing the remaining feasible preparation.
Starting published head: `3f7e39249de496dc28a59e9e94a3c01390d1cb4b`
(local tree-equivalent receipt `0c718cea3eb982717ce940560868004d71261e9b`).
Inspected main: `18c467e18b71d972060365c68865aa405a69949c`; its only advance
from the assessed source changes the Renovate workflow. #624 still owns shared
integration; #656/#659 have no overlap with the allowed files below.

- [x] JOSS-02a: implement an offline manuscript/source integrity check with
  negative tests for drift, incomplete evidence and false readiness; preserve
  the current unapproved edition and all scientific/author/provider gates.
  Fifty tests and independent review pass; 14 source hashes also match Git blobs
  at the recorded assessment revision. The checker itself does not claim Git
  authentication or submission readiness.
- [x] JOSS-02b: implement the reproducible pinned draft-render wrapper and its
  18 guard tests; execute actual bibliography parsing, official TeX/citeproc and
  draft-marker checks. The final `attempt-2` receipt matches the final script.
- [!] JOSS-02c: complete PDF compilation and visual inspection. Actual LuaLaTeX
  stopped at missing `luacode.sty`; Make exited 2, wrapper exited 1 and no PDF
  was produced. Continue in a complete TeX/Inara environment with a new receipt;
  do not infer visual acceptance from TeX output.
- [x] SWH-01a: resolve the historical release/tag inventory, including the
  annotated-tag target; preserve target selection, rights and archival gates.
- [x] Reconcile observed hosted results and remaining JOSS-01, RRID-01, SWH-01,
  author and external gates after execution; independently review the bounded
  implementation and final portable fixtures. The exact publication, issue
  readback and new-head hosted receipts belong in #662 and its #624 handoff.

Allowed files: this track, `scripts/check_joss_readiness.py`,
`tests/native/test_joss_readiness.py`, `tests/native/test_joss_render.py`, and
the separately scoped scientific Arrow qualification packet. Root workflows,
manifests, dependency pins, common registries, empirical protocols and other
owners' source remain reserved.
Preparation checks cannot supply research-use facts or author attestations.

## Scientific and journal reconciliation — 16 November 2026

Current work packet: `REG-16`, based on
`69b674814513e09a683279c679b1584af8bb4214`. This section and
[the current reconciliation](reconciliation-20261116.md) supersede operational
status in the dated receipts below; those receipts remain historical evidence.
Allowed changes are this track and the existing scientific ecosystem alignment
track. The shared integrator in #624 retains promotion, root registry, workflows,
pins and scientific acceptance. #656 owns the overlapping empirical plan and
#659 retains documentation work.

- [x] Reconcile the five requested issues, merged/closed PRs, current metadata,
  scientific dependencies and the actual native issue hierarchy.
- [x] Prepare a separate JOSS-format draft, checked bibliography and claim/source
  register; preserve the legacy manuscript and V05-owned scholarly corrections.
- [x] Refresh bounded read-only provider lookups and distinguish absent tested
  origins, inaccessible results, submission and verified external outcomes.
- [x] Verify the updated pending record, manuscript structure/citations/links,
  exact changed-file scope and independent review; retain real failing gates.
- [x] Publish the bounded PR and reconcile all five issue descriptions with its
  exact source/evidence. Published as draft [#662](https://github.com/edithatogo/innovate/pull/662),
  initial head `0b5751002c7fd6cfe6ebd60b16208c94597cfad2`; all five issue bodies
  were updated and read back on 16 November 2026. Integration and external
  completion remain separate.

The remaining executable packages and their completion evidence are in the
[current reconciliation](reconciliation-20261116.md#ordered-work-and-acceptance).
No new scientific implementation owner is assigned by this packet.

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
