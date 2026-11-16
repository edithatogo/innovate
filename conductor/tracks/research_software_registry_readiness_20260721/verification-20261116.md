# Verification receipt — scientific and journal reconciliation

Base software: `69b674814513e09a683279c679b1584af8bb4214`.
Scope: the two existing scientific-alignment and registry-readiness tracks;
no scientific source, shared registry, CI, dependency or release changes.

## Executed baseline checks

An isolated `.venv-registry-reconciliation` was created for this checkout using
actual CPython **3.15.2**, pytest **9.2.0** and Ruff **0.17.5**. No existing
owner environment or project dependency lock was changed. System Python
**3.12.14** performed preliminary dependency-free metadata checks only.

| Check | Outcome and scope |
| --- | --- |
| `python scripts/check_registry_readiness.py --json` | Exit 0: metadata fingerprints valid; all three destinations pending; no online verification or external completion |
| `.venv-registry-reconciliation/bin/python -m pytest tests/native/test_registry_readiness.py -o addopts='' -q` | **56 passed**; actual Python 3.15.2, pytest 9.2.0. Only output verbosity was overridden; no version minimum, assertion, selection or gate was weakened. |
| `.venv-registry-reconciliation/bin/python scripts/check_migration.py` | Exit 0: development policy valid; **38 obligations** remain |
| `.venv-registry-reconciliation/bin/python scripts/check_migration.py --release` | Exit 1: all 38 scientific obligations still block release; this expected refusal is preserved |
| `python scripts/check_repository_policy.py` | Exit 0: local repository policy passes; live enforcement and external evidence not checked |
| `python scripts/sync_versions.py --check` | Exit 0: native development version mapping valid; not a released version |
| `python scripts/conductor_registry_hygiene.py --json` | Exit 1: three existing completed tracks under active paths: ODE, NUTS, CI supply-chain. Shared correction remains with #624/#656. |
| Native GitHub subissue readback for #398 | Existing #399/#400/#401 edges confirmed; all open. Project membership not verified. |
| Fresh read-only Software Heritage/SciCrunch queries | Two exact origins returned 404; RRID search returned 403/non-JSON. [Normalized observations](evidence/provider-lookups-20261116.json) preserve the limited outcome. |

An earlier system-Python `check_migration.py --release` attempt failed at the
runtime guard before evaluating the ledger. The Python 3.15.2 executions above
supersede that execution limitation without changing the guard.

## Hosted baseline evidence

At the inspected main, [Release Readiness run 43844964697](https://github.com/edithatogo/innovate/actions/runs/43844964697)
failed in job [133222714028](https://github.com/edithatogo/innovate/actions/runs/43844964697/job/133222714028).
The retrieved log records Python 3.15.2 and the same 38 unmet scientific
obligations. Release-lane checks were skipped. This is an intentional retained
release gate, not a failure introduced by manuscript preparation.

At the initial check observation, CI, documentation, installed Voiage and coverage
were running, and offline security was queued. Privacy/secrets, RustSec,
workflow/policy lint and Scorecard had succeeded. These are statuses at that
observation, not a final-source aggregate qualification.

## Final source review

An independent automated reviewer read the complete 20-file packet and found
no blocking scoped defect. Its one bibliography correction, changing the Mesa
author literal from `rht` to the publisher's `Rht`, was applied and that reviewed
bibliography fingerprint refreshed. The reviewer independently repeated the
56 registry tests and verified the 14 unchanged source hashes, three draft
edition hashes, allowed-file boundary, pending provider states and preserved
historical scientific-plan bytes. This is automated source review, not human
author confirmation or independent scientific acceptance.

Final local checks after the reconciliation edits:

| Check | Result |
| --- | --- |
| Registry test suite under Python 3.15.2/pytest 9.2.0 | **56 passed** |
| Updated registry checker | Exit 0; all three destinations pending; source baseline updated after verifying unchanged metadata bytes |
| `--require-complete` | Exit 1 as required; no external state promoted |
| Ruff check and format check on the existing checker/tests | Passed; neither source file was changed |
| Repository policy and development version mapping | Passed |
| JSON, metadata, scope and relative links | Five changed/new JSON records parsed; all 20 paths stay in the two tracks; 107 relative links checked, excluding preserved historical plan text |
| Source and edition fingerprints | 14 inspected baseline inputs and three draft-edition files match their recorded SHA-256 values |
| JOSS structural checks | YAML front matter parsed; six mandatory sections present; five unique citations resolve to five bibliography entries; **1,074 prose words** excluding notice/headings/citation tokens |
| Historical scientific plan | Original section from the 14 November continuation to EOF is byte-for-byte preserved |
| Whitespace | `git diff --check` and changed/new text checks passed |
| Conductor hygiene and scientific release checks | Same pre-existing three registry/archive discrepancies and 38 unmet scientific obligations; no new waiver or acceptance claim |

The independent link check examined 79 newly added links; the root check also
examined relevant existing text in changed files and counted 107. These are
different scopes, not conflicting results. JOSS word count uses a local Unicode
word-token rule, not an editorial or official-renderer decision. A full BibTeX
parser and JOSS PDF rendering were not executed.

Before publishing, main was reread unchanged at the baseline. PR #656 had
advanced to `035f2c7b253a9eee58b97cbb22de5637a289ad88`; all 226 changed paths
were reread and none overlaps these two tracks. #659 remained at
`76f455fb568639cccb2370720eebcce399b2e158`. Shared files and their integration
remain owned by the existing coordinator. Publication and final-head hosted
results belong in the PR receipt; this file does not predict them.

Full native computation, installed artifacts, platform compatibility, benchmark
performance, scientific validity, author approval and external provider/editorial
acceptance are not established by these preparation checks.
