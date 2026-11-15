# Voiage alignment acceptance reconciliation

Recorded 15 November 2026 in Australia/Brisbane; referenced executions and
inspection are dated 14 November UTC. This is the current acceptance record for
[issue #559](https://github.com/edithatogo/innovate/issues/559), within the existing
`scientific_ecosystem_alignment_20261029` programme.

[PR #605](https://github.com/edithatogo/innovate/pull/605) is merged. Its final
head `5bfafaede6bb1001463480cc16a429cee052eb33` and receiving commit
`eeac954a5fa335e00632f0aec4f3cbb94f75b656` have the same complete tree,
`2cedc46eaa7511c56b10f77288e0c60a6a13c5ef`. All seven required contexts passed
on that final head. The merged implementation has subsequently passed the
installed native consumer on main.

This record supersedes the execution-pending, signed-out Dependabot,
unobserved-bot-signature and earlier-validator status in the dated audit,
qualification and continuation. Those records remain historical evidence.
The record-restoration proposal still needs its own actual protected checks
and integration by the existing repository coordinator before archival.

## Issue acceptance

| Scope | Executed evidence and disposition |
| --- | --- |
| Public data boundary | Polars 2.1.0 is the base tabular dependency; optional Arrow interoperability uses PyArrow 26.0.1. The inspected active Python closure contains neither pandas nor xarray. No runtime exception was introduced. |
| Voiage interchange | The installed consumer passed all six IPC/Parquet cases, including multi-chunk columns and Parquet-normalized schemas. Original/restored digests and chunk layouts agree; the supported numerical projection produces `[1, 2, 3]`. |
| Toolchain | Executed Python 3.15.2, Rust 1.99.1 and uv 0.13.8 retain the later native migration targets. Ruff 0.17.4 and the existing type-checking gate are retained. The checker/scanner exceptions below are explicit. |
| Dependency alerts | Direct authenticated observation of the `is:open` Dependabot page found **0 open alerts**. No alert was dismissed or modified. The 159 closed alerts were not individually audited. |
| Signed bot updates | Actual Renovate PR #612 has a GitHub-verified signature on its bot commit. All seven required contexts passed on that exact head. |
| Installed artifact | The original complete raw receipt is restored unchanged. A newer main archive was independently downloaded, SHA-256 checked and parsed; a subsequent main archive was also independently checked after executor access returned. Its earlier API-only snapshot is preserved separately. |

### Retained toolchain exceptions

The original issue's Python 3.15.1 and Rust 1.92 targets predate the approved
native migration. Retain Python **3.15.2** and Rust **1.99.1**; the historical
required context `unit-tests-base (3.15.1)` is still the protected name, while
the platform jobs actually assert Python 3.15.2.

Retain **basedpyright 1.40.2** as the named checker exception for this bounded
consumer follow-up. Its active gate is preserved. **ty was not executed or
qualified**; no incompatibility is asserted. Any future pinned ty candidate,
public-binding diagnostics and deliberately invalid type-fixture qualification
belong to the existing shared toolchain owner before replacing that gate. This
disposes of the exception for #559 without representing ty as passed.

The existing Presidio scan uses isolated **Python 3.12**, because that scanner's
recorded release excludes Python 3.15. It is a scanner-only exception, separate
from the application's interpreter, and remains with the privacy workflow owner.

## Installed native consumer evidence

The restored [historical continuation](continuation-20261115.md) retains the
unmodified [qualification JSON](evidence/hosted-consumer-43603039393/qualification.json),
[test output](evidence/hosted-consumer-43603039393/tests.txt) and
[build/archive provenance](evidence/hosted-consumer-43603039393/provenance.json)
for [run 43603039393](https://github.com/edithatogo/innovate/actions/runs/43603039393).
Its receipt identifies the actual synthetic checkout `6d132d6`, not a later
head. The independently checked original archive SHA-256 is
`e9fce853715cba333207b884b13012600a0ca11d21b3f8c91f1ee17383fcc6c2`.

The [final-head/receiving-main receipt](evidence/acceptance-gates-20261115.json)
records subsequent successful native runs:

| Actual source | Consumer run | Result |
| --- | --- | --- |
| Final #605 head `5bfafaed` | [43626767375](https://github.com/edithatogo/innovate/actions/runs/43626767375) | Success |
| Receiving main `eeac954a` | [43629677420](https://github.com/edithatogo/innovate/actions/runs/43629677420) | Success |
| Later main `f6c7d058` | [43640667994](https://github.com/edithatogo/innovate/actions/runs/43640667994) | Success; archive independently inspected |
| Main after #611, `384ed3b0` | [43649444847](https://github.com/edithatogo/innovate/actions/runs/43649444847) | Success; archive subsequently downloaded, hashed and parsed |

For `f6c7d058`, artifact **14131728926** was downloaded and its independently
computed ZIP SHA-256 matched the published artifact digest:
`8cdd3629c30541ec99fff7c14c4ddb37ebc49a7d9c4184709a764c2b89dd36f0`.
The [inspection summary](evidence/consumer-43640667994-inspection.json) records
the actual source, logged invocation, outside-checkout working directory and
installed-package root, versions, wheel/native/RECORD hashes and numerical results. The full archive
contained the expected three files; qualification stderr was empty. Parsed
qualification status was passed with worker exit **0**, all six normalized
cases agreed, and the focused suite passed **19 tests in 2.37 seconds**.

The native result uses only the explicit `time`/`sample_id` projection.
Preserving timestamp/covariance metadata does not qualify native datetime or
covariance operations. Equal-weight and empirical 3:1 EVPI remained within the
unchanged `1e-12` tolerances: references **5.641720719927869** and
**3.7317144196428957**, observed **5.641720719927868** and
**3.731714419642895** respectively.

The later `384ed3b0` run passed **19 tests in 1.85 seconds** and its native
qualification step completed successfully. The initial
[API/log receipt](evidence/current-main-consumer-20261115.json) correctly
records that archive bytes had not yet been inspected at 23:08 UTC. Executor
access subsequently returned. Artifact **14135661886** was then downloaded,
independently SHA-256 checked and parsed; the computed ZIP digest matches
GitHub's reported
`6800e15218c51ee8181b3aea0cf73be4621ed8c28c14c4cfc555d73a7be1ebbc`.

The complete, unmodified
[latest qualification JSON](evidence/hosted-consumer-43649444847/qualification.json)
and [test output](evidence/hosted-consumer-43649444847/tests.txt) are retained with
[independent archive provenance](evidence/hosted-consumer-43649444847/provenance.json).
The actual receipt identifies `384ed3b0`, worker exit **0**, all six matching
digest/chunk/native-result cases, the expanded worker command, each imported
module path and all input/output hashes. Its working directory is
`/tmp/innovate-installed-consumer-pfrx4x2l`; imports are beneath the installed
`voiage-consumer.4HY0Ij` environment outside the source checkout. The earlier
API-only limitation remains accurate for its observation time and is
superseded by this later direct artifact verification.

The harness, focused tests and consumer workflow remain byte-identical to the
qualified source through the inspected `384ed3b0` revision. The consumer workflow
does not trigger on Conductor-only record changes. An absent new consumer run
for this restoration is not represented as a pass on its eventual new commit.

## Dependency and automation acceptance

### Authenticated alert inventory

The [direct UI receipt](evidence/dependabot-inventory-20261115.json) records an
authenticated [Dependabot page](https://github.com/edithatogo/innovate/security/dependabot)
with filter `is:open`, **Open (0)**, **Closed (159)** and the explicit message
that no open Dependabot alerts exist. The observation is bounded to the
recorded 14 November UTC inspection window.

No alert dismissal, remediation-setting change or unsupported API call was
performed. The 159 closed alerts and the auto-dismissed subset were not audited;
the attempted read of the latter timed out. The source inventory identifies
the isolated historical candidate Cargo lock, with no historical Python lock
in that inspected tree. There was no open alert requiring an alert-specific
historical-lock disposition at the observed inventory.

The earlier 99-distribution Python audit remains scoped to its actual audited
environment. It is distinct from the live alert inventory and from the
101 package records in the subsequently inspected uv lock. The
[current-source inventory](evidence/acceptance-security-20261115.json) is
explicitly labelled as a reconstructed source audit at `f6c7d058`, not a new
vulnerability scan. The separate #611 merge subsequently advances Astro from
7.4.7 to **7.4.11**, both above the issue's requested floor.

### Actual signed Renovate update

The dashboard request selected only the available Renovate validator patch
**44.239.0 to 44.239.1**. The resulting
[PR #612](https://github.com/edithatogo/innovate/pull/612) is authored by
`renovate[bot]`; its one-line change is confined to the CLI package version
in `.github/workflows/renovate-config.yml`.

Bot head `594380162002123533db322248363a6546301534` has GitHub
`verification.verified: true`, reason **valid**, verified at
`2026-11-14T22:44:33Z`, with GitHub's `web-flow` committer.
The [signature and check receipt](evidence/renovate-signing-acceptance-20261115.json)
retains actual commit/check evidence. All seven required contexts passed on that exact head.

The initial scoped draft conversion was followed by an observed
`edithatogo` ready-for-review and protected auto-merge action. That subsequent
coordination decision was preserved. PR #612 merged at `2026-11-14T23:18:20Z` as
`663002b03ccb92abee2ece934d69116cb4bf3415`. Its bot head and receiving merge
share tree `b070238cde2eaec58eca2f68457e83282328c282`; the receiving commit is
also GitHub-verified. The only change is the validator package-version line.
Full classic signing-rule details remain inaccessible to the integration API;
this evidence demonstrates a real cryptographically verified bot commit and
its actual protected checks without claiming a full settings audit.

### Current configuration validation

PR #606's current configuration retains `platformCommit: enabled`, the central
preset and all **three custom regex managers**. Its Git blob is
`30d1a0d6ae51704578b42da778a1ad43575926fe`; SHA-256 is
`92dba60b86e925c326e8e62b2046485abb063ed45c896549864a2ec47a4951ae`.
The historical no-custom-regex/local-RegExp-fallback result applies only to its
earlier configuration.

Actual hosted repository-mode validation with Node **24.22.1** and Renovate
**44.239.0** passed on both #606's final PR checkout and receiving main; no
RE2/RegExp fallback warning appears in either inspected log. PR #612's actual
**44.239.1** validator also passed
`renovate-config-validator --strict --no-global .github/renovate.json`
in [run 43649679071](https://github.com/edithatogo/innovate/actions/runs/43649679071).
The Monday schedule inherits **Australia/Sydney** from the central preset;
the original audit's UTC wording is historical. Repository-mode validation
does not establish acceptance of a fully evaluated remote preset; that
separate shared-preset scope remains with its existing owner.

## Protected checks, restoration and integration

The seven required contexts remain `lint`, `package`, `CodeQL`,
`Secret Scan`, `Workflow and policy lint`, `Python quality harness` and
`unit-tests-base (3.15.1)`. All passed on #605's final head. Push runs on the
receiving merge and `f6c7d058` have six same-name successes and five successful
CodeQL analyses; the PR-only aggregate CodeQL context is absent on those
pushes, not silently counted as an additional pass. Actual IDs and workflow
URLs are retained in the gate receipt.

The integration rebase retained the harness and tests but omitted the six
historical continuation/evidence files from
`4ddd0c71c8c723046dd81d7d0baba79145d3bf97`. This restoration reuses their
exact original Git blob IDs, recorded in the gate receipt, and adds the
superseding evidence above. It preserves the separately merged #606 and #611
work. The [main-advance receipt](evidence/acceptance-main-advance-20261115.json)
records all 27 #611 paths and confirms no collision with this track or its
harness/test/consumer-workflow/Renovate-config paths.

The documentation work remains with its existing #611 coordinator. Current
main's [documentation job](https://github.com/edithatogo/innovate/actions/runs/43649444910/job/132614927566)
failed on 14 November at 23:05 UTC: the aggregate documentation checker reports
`Inventory example source hash is invalid.` for the Julia and R receipts,
with exit **1**. This is retained as an actual separate documentation failure;
the merged #611 change is not represented as full documentation acceptance.
The current-main receipt contains the precise log lines.

The restoration changes this track's evidence and status records. Its own
protected checks, final integration and archive/registry reconciliation remain
with the existing repository coordinator. The plan and metadata therefore
remain in progress until that handoff completes. No prior source-run success
is attributed to an untested record-restoration commit.

## Scientific and release boundary

The inspected Release Readiness logs reproduce the same **38 incomplete
scientific obligations**, with **exit 1**, on final #605 head, its receiving
merge, `f6c7d058` and `384ed3b0`; the real Renovate run retains that failure.
The exact lists/log lines are in the gate, main-advance and signing receipts.
No obligation was removed, waived or marked complete by this work.

This is a bounded synthetic CPU installed-consumer and automation acceptance.
The broader `voiage_arrow_handoff`, HPC, clinical, release and external
submission obligations remain unchanged. The independent offline security
draft #590 also retains its existing unqualified status.
