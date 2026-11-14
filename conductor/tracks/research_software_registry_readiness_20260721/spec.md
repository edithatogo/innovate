# Specification

## Objective

Make innovate's appropriate registry, archival, identifier, or research-software destinations explicit, executable, and evidence-gated.

## Requirements

- Maintain one GitHub parent issue and native registry-specific subissues.
- Record eligibility and prerequisite evidence before submission.
- Record authoritative submission, acceptance, publication, archival, or identifier evidence when available.
- Do not claim external completion from local preparation or a successful workflow alone.
- Preserve existing repository contracts and rights constraints.

## Executable preparation contract (14 November 2026 continuation)

- Bind the current metadata assessment to the inspected upstream revision,
  software version, and SHA-256 fingerprints of citation, licence, package and
  migration-ledger inputs. Changed inputs require reassessment.
- Keep exactly the three registry-specific issue mappings below. Each destination
  has a dossier, explicit outstanding work, and one of `pending`, `submitted`,
  `verified` or `ineligible`. Only evidenced final dispositions resolve a record.
- Validate retained provider events offline: matching resource/version/state,
  timezone-aware observation time, authoritative provider path, successful
  response status, immutable receipt/response digests, and identifier consistency.
  Software Heritage resolver fields must confirm the selected object, with the
  tag/commit relationship recorded separately.
- Reject stale or missing evidence, unsupported state transitions, placeholder
  identifiers, duplicate JSON fields, path escape and negative resolver results.
- A valid pending record exits zero. A completion check fails while unresolved
  destinations remain. No local validation result claims online verification,
  author approval, provider authenticity, publication, or scientific/release
  qualification. Provider records require maintainer verification before admission.
- Exercise the contract and the real repository record in `tests/native`, using
  synthetic positive/negative fixtures without making network requests.

## GitHub hierarchy

Parent: [#398](https://github.com/edithatogo/innovate/issues/398)

- [#399](https://github.com/edithatogo/innovate/issues/399)
- [#400](https://github.com/edithatogo/innovate/issues/400)
- [#401](https://github.com/edithatogo/innovate/issues/401)

## Acceptance

The planning structure is complete when all subissues are linked, project-visible, and traceable from this track. Individual registry deliverables close only with authoritative evidence or a documented ineligibility decision.
