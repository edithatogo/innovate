# Research Software Registry Readiness

Status: `repository_preparation_in_progress`

Issue [#398](https://github.com/edithatogo/innovate/issues/398) coordinates
Software Heritage [#399](https://github.com/edithatogo/innovate/issues/399),
RRID [#400](https://github.com/edithatogo/innovate/issues/400), and
JOSS [#401](https://github.com/edithatogo/innovate/issues/401). All three remain
open. Current source is the Rust/Polars **0.7 development line**; registry
preparation does not qualify a release or satisfy the scientific migration gate.

## Current evidence

The owning [Conductor track](../conductor/tracks/research_software_registry_readiness_20260721/index.md)
contains a [machine-readable record](../conductor/tracks/research_software_registry_readiness_20260721/readiness.json),
the [Software Heritage and RRID dossier](../conductor/tracks/research_software_registry_readiness_20260721/registry-evidence.md),
and the [JOSS assessment](../conductor/tracks/research_software_registry_readiness_20260721/joss-readiness.md).

| Destination | Observed position | Evidence still needed |
| --- | --- | --- |
| Software Heritage | The canonical origin and `.git` variant returned `404 NotFoundExc` in the recorded 14 November 2026 checks. | Selected release/tag and commit, successful archive visit/snapshot and matching resolving SWHID. |
| RRID | Software is a plausible resource type. The live duplicate search was unavailable. | Completed duplicate check, curator decision and matching resolving RRID. |
| JOSS | Existing manuscripts describe the historical Python implementation; no qualified native submission packet exists. | Current manuscript, research-use/history evidence, scientific and release qualification, author declarations, then editorial and publication evidence. |

PR [#402](https://github.com/edithatogo/innovate/pull/402) is closed and unmerged;
the July issue comments describing it as open are historical. The current record
supersedes pre-0.7 `ready` / `ready_for_review` assertions in the historical
`docs/source/_static` inventories. Those files are retained as history.

## Executable checks

Use the repository-selected Python 3.15.2 interpreter. No network access or
native library import is needed for this metadata check:

```sh
.venv-native/bin/python scripts/check_registry_readiness.py --json
.venv-native/bin/python scripts/check_registry_readiness.py --require-complete --json
.venv-native/bin/python -m pytest tests/native/test_registry_readiness.py
```

The first command exits **0** when the evidence record is internally consistent,
including honestly pending work. `--require-complete` exits **1** while any
destination remains pending or submitted. Either command exits **2** for an
invalid record, changed metadata fingerprint, missing receipt, mismatched
identity/version, untrusted provider URL, invalid identifier or changed evidence.

The checker is an **offline integrity check**. It does not contact providers,
authenticate a response, submit software, close issues or qualify a release.
Its report always states `online_verification_performed: false`.
`recorded_complete` describes retained dispositions, not independent proof of
provider acceptance. A maintainer must verify the original provider record
before admitting it through review. A fabricated response and matching hash are
not authoritative evidence.

## Updating the record

`assessed_revision` identifies the upstream reconciliation baseline. The four
`input_sha256` entries identify the metadata actually assessed, including the
citation corrections in this increment. Reassess changed inputs before updating
their hashes; do not regenerate hashes merely to silence a failure.

Each destination has an immutable issue mapping, a dossier, an explicit state
and outstanding work. `pending` has no acceptance identifier or event receipt;
negative/unavailable lookups remain observations in the dossiers. `submitted`
requires a retained provider receipt but remains incomplete. `verified` and
`ineligible` require a provider outcome and no outstanding work. Ineligibility
must have an evidenced decision, not an unsupported local label.

An event receipt is a JSON file referenced by path and SHA-256 in the manifest.
It records `provider`, canonical `repository`, `software_version`, `event`,
timezone-aware `checked_at`, authoritative `url`, successful integer `http_status`,
`summary`, `identifier`, and a retained `response_path` / `response_sha256`.
The record and receipt identifier must agree. Verification requires a supported
provider resolver path, with no query or fragment substituting for the identifier.
The supported paths are Software Heritage's `/api/1/resolve/<SWHID>/`,
SciCrunch's `/resolver/<RRID>`, and JOSS's `/papers/<DOI>` or `doi.org/<DOI>`.
For RRID and JOSS, the retained response must contain the identifier.
For Software Heritage, the
[resolver response](https://docs.softwareheritage.org/devel/swh-web/uri-scheme-api-swhids.html)
must instead confirm `namespace`, `scheme_version`, `object_type` and `object_id`;
a not-found/error response cannot establish archival completion. Release evidence
also identifies the selected `release.tag` and `release.commit`; a revision
SWHID must match that commit. This version accepts revision or release SWHIDs
for the release-archival deliverable, not a bare snapshot or unrelated directory.
Use `null` identifiers for submitted or ineligible events. All referenced files
must remain within the repository. Store only public, relevant provider evidence.

The corrected `CITATION.cff` describes development version `0.7.0a1`, not a
published release. The author ORCID is corroborated by the
[publisher's author record](https://www.nature.com/articles/s41431-023-01417-6).
No DOI, SWHID, RRID or JOSS acceptance has been invented or assigned.
