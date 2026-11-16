# M01 source qualification lane

- Owner: Codex chat `01a1e3d8-fb04-7010-a038-2a5478662a5d`.
- Worktree: `/Users/doughnut/.codex/worktrees/innovate-empirical-m01/innovate`.
- Branch: `codex/innovate-empirical-m01`; base: `18c467e18b71d972060365c68865aa405a69949c`.
- Scope: M01 and a bounded later-packet preflight. Allowed files: this lane handoff, `validation/data/pbs-sources.json`, `scripts/validate_pbs_source_manifest.py`, `tests/validation/test_pbs_manifest.py`, and one upstream metadata receipt template under `tests/validation/fixtures/`.
- Native object scope: none. Source-owner material is read-only. No full-object acquisition, publication or messaging is authorised by this lane.
- Shared track plans, manifests, locks and promotion remain with their existing integrator. Next operation: hand over this reviewed packet for shared CI selection and integration.

An initial read-only metadata exploration accidentally used the shell's Python 3.11.6 rather than the required repo-local interpreter. It wrote nothing in either repository; its relevant results will be independently rechecked using this lane's verified Python 3.15.2 interpreter before use.

## Evidence bound by this packet

The manifest freezes dataset `edithatogo/australian-pbs-utilisation-archive` at
revision `9f1d53cd50eea251b326da42a96f3ac72ebce97a` and the six exact paths,
SHA-256 values and byte counts already identified in the medicines research
plan. Their total is 191,047,619 bytes. Its upstream metadata is pinned to Atlas
commit `c6d289685ff5712c64e2ab62a6974e20ddaf4712` and the hashes in the
manifest. The checked-in fixture is a source-attributed Atlas replay receipt
template containing metadata, not source values or records. The test rebuilds
its deterministic idempotency digest from the partition and snapshot IDs, then
verifies the exact original receipt hash before use. See the
[integration review](integration-review-20261117.md) for the scan finding and
the distinction between the template and the original upstream bytes.

The earlier [role-readiness snapshot](https://github.com/edithatogo/global-medicines-atlas/blob/c6d289685ff5712c64e2ab62a6974e20ddaf4712/quality/qualifications/australian-m112-pbs-utilisation-role-readiness-20261028.json)
preceded the later [six-object raw-admission cohort](https://github.com/edithatogo/global-medicines-atlas/blob/c6d289685ff5712c64e2ab62a6974e20ddaf4712/quality/qualifications/australian-m112-pbs-v4-admission-readers-20261028/manifest.json).
All six raw admissions and bounded raw-reader canaries passed; their native
schemas and PBS domain semantics remained unqualified. Atlas's
[2026-11-15 reconciliation](https://github.com/edithatogo/global-medicines-atlas/blob/c6d289685ff5712c64e2ab62a6974e20ddaf4712/quality/qualifications/australian-m112-effective-raw-delivery-reconciliation-20261115-august.json)
retains these six within its later raw-delivery totals. No Innovate modelling
consumer canary or qualified temporal panel was evidenced by those raw checks.

The later maintainer attestation covers public release permission for these
exact six objects; its permission instrument was not independently reviewed.
The old lineage record's `unknown`/`unreviewed` fields remain historical. The
source-specific `non_sensitive` assessment is not exhaustive privacy clearance
or proof that personal data are absent. No new legal or privacy assessment is
claimed.

## Execution and review

- Local environment: explicitly selected `.venv/bin/python`, Python 3.15.2;
  pytest 9.2.0 and Ruff 0.17.5 installed under constraints exported from the
  unchanged repository lock. The initial read-only Python 3.11.6 exploration
  noted above was repeated with this interpreter, including SHA-256 checks of
  the four earlier qualification records and all six replay receipts.
- Initial test run failed at the not-yet-implemented validator. A further test
  run rejected the newly reconciled admission fields until the manifest
  validator was updated. These were expected red states, not passing evidence.
- `uv run --locked --no-sync --python .venv/bin/python pytest tests/validation/test_pbs_manifest.py -q`:
  **27 passed**. Negative cases cover revision, object identity, payload bytes,
  receipt corruption and substitution, duplicate JSON keys, rights chronology,
  unsupported sensitivity claims and false qualification/canary status.
- `ruff format --check` and `ruff check` on the two Python files: passed.
  `.venv/bin/python scripts/check_repository_policy.py`: passed for local policy;
  live settings were not checked.
- `.venv/bin/python scripts/validate_pbs_source_manifest.py --receipt-root /Users/doughnut/.codex/worktrees/gma-issue-reconciliation/global-medicines-atlas`:
  passed against the existing Atlas metadata. It verifies the six replay
  receipts, six later raw-admission receipts and their cohort, rights and
  sensitivity evidence hashes; it does not read the source objects.
- The six existing replay workflow runs `40677682866`, `40678419501`,
  `40678703243`, `40679077698`, `40679177671` and `40679271451` were separately
  logged upstream jobs at `8343d511179e71adea825a5f46e7c6cd60b16373`. The six
  later raw-reader runs `40691968215`, `40692208170`, `40692360821`,
  `40692757372`, `40692888934` and `40693427540` were run at
  `10b104417ccb37da6ec20b8e7151c6486a787eec`. `gh run view` on 2026-11-16
  read back all twelve as completed/success. These are the already-performed
  acquisition/read jobs, reused without another acquisition in Innovate.
- Review fixes: rejected a receipt paired with a caller-substituted digest;
  added source-payload verification bound to the validated revision and object;
  corrected the stale pre-admission disposition using the newer raw cohort;
  and fixed collection under the plan's `uv run pytest` command. The earlier
  command initially failed collection because its import path differed from
  `python -m pytest`; the exact command above passed after the fix.

Originally reviewed source-packet SHA-256 values before the integration
fixture change: manifest
`8ae7b0c185818f78bb0ba10512f6ec10a577d3b2f51e42c41ac8a53a678133ce`;
validator `aa3daea47bae7592da60ea36eb748f80a44561a22387f04ed1d68f8213107a87`;
tests `5b3fd131d9f86d6e9ed3f346bce448fb5838371760a93059ff687b940ac25134`;
original upstream fixture `18f0101937e34aaa10f19387cd1d5b1fcc67933bd9bf4928ca9fab561d89754a`.

## Handoff and remaining gates

The shared CI and Nox test selectors at this lane's base run `tests/native`, not this new
`tests/validation/test_pbs_manifest.py`. The shared-workflow integrator must add
and qualify the exact offline selector and corresponding lint coverage; no
hosted CI pass is claimed by this lane. The shared track plan remains with its
integrator and M01 remains unchecked there pending integration.

Main advanced to `57d2916586d107c8affe90355e453abe27f01c26` by merging PR
#656 while this lane was active. Its archived Rust plan now marks R02, R03 and
R08 complete and identifies `innovate-tabular` as the R03 implementation; those
labels were not requalified by this packet. The R08 record inspected at that
revision lists commands and a status but does not itself bind the five
taskbook qualification checks to results. M02 still needs accepted upstream
dependencies, qualified source semantics and an appropriate consumer canary.
No observed coverage, field meaning or modelling qualification follows from
these byte receipts. M03–M05, the broader validation programme, scientific
review and release/publication acceptance remain open. The empirical track
must not be archived on the strength of this packet.

## Later-packet source preflight ownership — 17 November 2026

The same chat and worktree are performing a bounded, read-only follow-up for
the requested later empirical packets. The only tracked file in this follow-up
is this existing lane receipt; downloaded public explanatory documents are
kept under the worktree-local ignored environment for hash verification.
The source-owner repository and hosted dataset remain read-only. This
follow-up binds observed CSV headers and current publisher documentation to
their exact sources, without claiming M02 or modelling readiness.

### Observed headers and source context

At 2026-11-16 approximately 15:52–15:56 UTC, bounded read-only Hugging Face
filesystem requests targeted the manifest's exact dataset revision
`9f1d53cd50eea251b326da42a96f3ac72ebce97a`. The connector reported the two
supplementary CSV object sizes as 35,384,567 and 36,381,946 bytes, and the
mapping CSV size as 1,063,694 bytes, matching the earlier manifest. Both
supplementary files returned this identical header:

```text
MONTH_OF_SUPPLY,ITEM_CODE,DRUG_TYPE,PATIENT_CAT,PHRMCY_TYPE,SCRIPT_TYPE,PRESCRIPTIONS,PATIENT_CONTRIB,GOVT_CONTRIB,TOTAL_COST,RETAIL_MARKUP,PATIENT_NET_CONTRIB
```

The mapping header was `ITEM_CODE,DRUG_NAME,FORM/STRENGTH,ATC5_Code`.
The SHA-256 of the UTF-8 header strings without newline is respectively
`26bd5464cd25523aa4cd3e962470cbedb59d13733c8f4ec4a6018adb6bee5a26` and
`92706b60d6d5b47b848a2bcf4f593a036efbf17b1a4a481fa4fe5b8c5d0c7bf2`.
Only bounded prefixes were requested; the complete objects were not downloaded
or rehashed in this follow-up. One initial response also exposed a few
public aggregate rows in the tool trace; no source payload was saved as a
project artifact. The earlier upstream full-byte receipts remain the source of
the object digests, not this header observation.

The current [PBS date-of-supply page](https://www.pbs.gov.au/statistics/dos-and-dop/dos-and-dop)
and its [explanatory notes](https://www.pbs.gov.au/statistics/dos-and-dop/files/dos-and-dop-explanatory-notes.pdf)
were read on 17 November Brisbane time. The downloaded HTML and five-page PDF
were retained only under `.venv/source-context/`, with SHA-256 values
`1529efd1ea3fb78e7ada9485cf090f40c121462f07341fca870fa2ce939fd9ec` and
`e21161b48a05df9c708c5eb48eafb83c598d6c97391f9fa9b9bd584b07b0437b`.
The current notes describe `MONTH_OF_SUPPLY` as a `YYYYMM` supply month and
`PRESCRIPTIONS` as aggregate prescription counts. They describe supplementary
`DRUG_TYPE` and `PHRMCY_TYPE` as coded categories, and the item-code mapping as
including historical items and an RPBS-only catch-all code. The current page
says the supplementary report excludes Section 100 supply and that historical
figures may differ as claims and adjustments are processed.

Separate read-only HTTP HEAD requests to the current publisher URLs returned
`Content-Length` 35,384,567 bytes for the 2024–25 supplementary CSV,
36,398,673 bytes for the 2025–26 CSV, and 1,070,507 bytes for the item map.
The first size matches the frozen object but does not prove identical bytes;
the other two differ from the frozen object sizes above. These are response
metadata only, not full-content digests or evidence of an original publication
date. They reinforce the need to retain the frozen revision and avoid silently
substituting current files.

These are current publisher descriptions and exact frozen-header observations,
not proof that today's notes were the operative edition for either frozen CSV.
Do not parse item codes as numbers, infer individual patients, treat the
catch-all code as a unique medicine, assume the mapping is valid at earlier
dates, combine processing and supply periods, or infer complete coverage from
file names. Field values, observed month coverage, duplicates, mapping history,
workbook sheets, units and semantic fitness still need an appropriately scoped
source-owner/consumer qualification. No modelling canary or M02 acceptance is
claimed.

### Release and integration preflight

The exact current main observed during this follow-up was
`cc2e4da4af9c848043b31f1191c9d96212048276`. The existing read-only integration
preflight compared that tree with M01 `51c87f30290016afef385e9133f67ee9ca143f60`,
Q04 `1f9e6ae6cd2c6d8264335326812f7fd83c83236b`, and the R04 candidate
`15221370b7cae83bb55cb015c088a5f1177a9bf8`. At the observation point it found
clean worktrees, no overlapping changed paths, staged deletions or missing
target files, and no patch-equivalent commit on the target. This snapshot does
not establish mergeability, current hosted checks or acceptance; repeat it at
integration if inputs change. The shared integrator retains promotion.

The migration checker and ledger in this M01 branch are byte-identical to the
observed main versions. `.venv/bin/python scripts/check_migration.py --release`
exited **1** and listed all **38** obligations as still unqualified. A separate
existing isolated regression,
`tests/native/test_migration_policy.py::test_release_source_digest_tracks_source_and_excludes_receipts`,
passed its own assertions, including one that makes a synthetic 38-obligation
ledger refer to the same arbitrary text file and receives exit 0 when its hash
and source digest match. This is a demonstration of the checker's current
scope, not real release evidence: hashes and a self-declared ledger alone do
not establish the taskbook's scientific and exact-artifact acceptance. The
source-bound Q04 receipt contract and Q02/Q03 gates remain open; no schema,
release decision or shared workflow was changed here.

At the later hosted readback, [CI run 43896559807](https://github.com/edithatogo/innovate/actions/runs/43896559807)
had completed successfully on main `cc2e4da4`: its native Python and Julia/R
jobs reported success on Linux, macOS and Windows, as did package and lint.
The branch-protection context retains a historical `3.15.1` name; the native
job explicitly selects and checks Python 3.15.2. This is a main-source CI
receipt, not evidence for the unmerged M01, Q04 or R04 candidate commits or
their full installed release matrices. The exact-main [release-readiness run
43883430394](https://github.com/edithatogo/innovate/actions/runs/43883430394)
failed at `check_migration.py --release` with the 38 outstanding obligations;
its release-lane job was skipped. Neither the green CI run nor the skipped job
establishes release acceptance.

## Review of merged empirical-track status

At main `57d2916586d107c8affe90355e453abe27f01c26`, the shared empirical plan
and registry mark V01–V04, Q01 and Q04 complete. The following findings show
that the V01–V04 and Q04 completion marks exceed their taskbook evidence. The
shared plan and registry remain with the existing integrator; they have not
been modified in this lane.

- V01: `validation/oracles/manifest.json` contains labels in the hash fields
  and no bound input hashes. Its validator accepted a changed Bass input without
  changing the expected value. The required independent and stale-input
  qualification is incomplete.
- V02: `validation/benchmarks/protocol.py` accepted negative training indices
  and an out-of-range holdout in `validate_split`; a NaN prediction returned
  status `ok` and NaN RMSE. The taskbook also requires frozen baseline and
  tuning budgets, probabilistic scoring, calibration and memory evidence that
  this implementation does not report.
- V03: `validation/cross-domain/protocol.json` and its test supply three cohort
  labels and observation kinds, but no source revisions, coverage, frozen
  native/reference comparison or calibration and failure results. These are
  useful protocol scaffolding, not completed cross-domain validation.
- V04: `tests/validation/test_performance_receipts.py` accepted a fabricated
  receipt with a positive elapsed time and a negative error. The available
  benchmark emits one timing and absolute error; it does not establish the
  taskbook's comparable cold/warm, hardware, memory and other workload results.
- Q04: `scripts/validate_capability_claims.py` accepted a fabricated capability
  with an arbitrary revision and `/dev/null` as its native receipt. Its default
  validation does not bind the supplied revision to the checked source, and a
  test-source path is not a native execution or scientific acceptance receipt.
  This does not satisfy the taskbook's stale/changed-algorithm and false-claim
  rejection requirements.

The five adversarial observations above were reproduced read-only using
`.venv/bin/python` 3.15.2 and exact source extracted from commit `57d2916`.
The Q01 mark also needs evidence against its named corpus, bounded fuzz and
sanitizer requirements; this review has not established whether those runs
exist elsewhere. The integrator should reopen the unsupported completion marks
in both the plan and registry, retain the implemented partial work, and
requalify the dependent packets rather than treating the merged labels as
scientific acceptance.
