# Software Heritage and RRID evidence dossier

## Executed target inventory — 16 November 2026 follow-up

The [read-only target inventory](evidence/archive-target-inventory-20261116.json)
resolves the available archival candidates without selecting or publishing one.
GitHub's complete returned release page contains six published releases; its tag
references identify the separate annotated `v0.6.0` tag.

| Candidate | Verified repository identity | Remaining decision |
| --- | --- | --- |
| Latest published GitHub release, `v0.5.0` | Lightweight tag at `6b673832c6024f5332c971ac35172a96615f75f9`; published 29 August 2025 | Historical archival target only; assess the selected source and third-party rights before requesting archival. |
| `v0.6.0` | Annotated tag `86eceddcfa17362d05afa34a6f224e7eac5d561b` targets `3459f4528c11ceba8d1e5840d3da852dfb8d58bd`; GitHub reports its signature verified | No matching published release appeared in the returned collection. A signed tag does not establish scientific, rights or archival acceptance. |
| Current native development line, `0.7.0a1` | No matching `0.7` tag in the returned references | No qualified native release can be selected from this inventory. Keep current native acceptance with #556/#624. |

This completes a bounded identity-preparation step of SWH-01. The selected target,
rights assessment, actual archive request, successful visit/snapshot and resolving
object linkage remain pending. Historical archival may be considered separately
without being represented as current native qualification. The existing RRID
draft is locally complete; its inaccessible duplicate search and actual curator
outcome still require external evidence.

**16 November update:** [fresh read-only observations](evidence/provider-lookups-20261116.json)
again returned 404 for the two exact Software Heritage origins and 403/non-JSON
for SciCrunch search. These are observations, not archival or curator receipts;
the duplicate check remains incomplete. The
[current reconciliation](reconciliation-20261116.md) clarifies parallel RRID
assessment, release/source citation identity and JOSS dependencies. The dated
14 November dossier and responses below remain intact.

Parent: [#398](https://github.com/edithatogo/innovate/issues/398). Deliverables:
[Software Heritage #399](https://github.com/edithatogo/innovate/issues/399) and
[RRID #400](https://github.com/edithatogo/innovate/issues/400).

Prepared on 14 November 2026 against repository commit
`276c2e505a0ce26910820fe8a94a6a498f505700`. This is a repository preparation and
lookup record. It is not a submission receipt, archival certificate, curator
approval, or release qualification. Neither deliverable has verified external
completion in this dossier.

## Repository identity and scope

The canonical resource is **Innovate**,
<https://github.com/edithatogo/innovate>. The
[README](../../../README.md), [contribution guide](../../../CONTRIBUTING.md), and
[product definition](../../product.md) describe a Rust scientific core, Polars
tables, and thin Python, Julia and R bindings. The breaking native migration
remains in progress. Scientific obligations and distribution/release gates remain
required; local metadata validation does not satisfy them.

Project-owned code uses [Apache-2.0](../../../LICENSE). Third-party licences and
notices remain applicable. Archive/publication preparation must retain those
rights constraints and must not treat a public repository URL as evidence that
every unrelated local artifact is approved for publication.

The July subissue comments refer to [PR #402](https://github.com/edithatogo/innovate/pull/402)
as open. The inspected PR is now closed, unmerged, and draft (closed on
8 November 2026; head `4b482a2611ccdf2f9f119c3d1a539b3b9d09a410`). Its historical
planning work is not an external provider receipt.

## Software Heritage: requirements and observed lookup

The [official archival guide](https://www.softwareheritage.org/how-to-archive-reference-code/)
and [FAQ](https://www.softwareheritage.org/software-heritage-faq/) distinguish a
request to save code from completed ingestion. GitHub hosting alone does not
establish that the selected repository state has been archived. Software
Heritage provides object identifiers and records provenance through origins,
visits and snapshots; see its [API and data-model overview](https://www.softwareheritage.org/software-heritage-features/).

Annotated Git tags can appear as Software Heritage release objects. A lightweight
tag can still be archived as a reference without appearing as a release object.
The absence of an entry in the archive's Releases tab is therefore insufficient
to establish that a release reference is absent; see FAQ section 2.7.

Two read-only origin lookups returned HTTP 404 JSON responses:

| Origin | Provider response date (UTC) | Observed result |
| --- | --- | --- |
| `https://github.com/edithatogo/innovate` | `2026-11-14T13:48:11Z` | `NotFoundExc`: tested origin not found |
| `https://github.com/edithatogo/innovate.git` | `2026-11-14T13:48:30Z` | `NotFoundExc`: tested origin not found |

Exact endpoints and response bodies are retained in the
[normalized lookup receipt](evidence/software-heritage-origin-lookups-20261114.json).
The provider response times are preserved separately from the receipt's capture
time. Only response bodies and selected non-sensitive metadata are retained.

**Interpretation:** these two origin URLs were not found when checked.
`not_found` is not permanent absence, a search across every possible origin, or
evidence that no equivalent source object exists in the archive. Later ingestion
or an alternate origin could change the result. No Save Code Now request was
sent, no release was selected for archival, and no SWHID was verified during this
work. A syntactically valid identifier derived from a Git hash would not prove
provider ingestion or resolution.

### Remaining archival gates

1. Select an actual release/tag and resolve it to its immutable commit, including
   annotated-tag dereferencing when applicable. Record the release URL, tag and
   commit; do not infer release readiness from this dossier.
2. Confirm the selected public source and relevant history are appropriate for
   archival under the repository's existing rights and publication constraints.
3. Recheck the canonical origin. If an archival request is made, retain its
   provider receipt and distinguish accepted/queued from successful ingestion.
4. Obtain completed visit/snapshot evidence linking the exact origin and selected
   tag or revision. Follow snapshot references rather than relying on an unrelated
   archived object or a moving default branch.
5. Resolve the cited SWHID through Software Heritage and verify its object kind
   and relationship to the selected source. Preserve the provider URL, response
   date and result.

The external gates are provider ingestion and resolution of the matching source
identifier. Keep #399 open until authoritative evidence supports the named
deliverable, or record an explicit disposition under the track specification.

## RRID: eligibility assessment and submission draft

The [SciCrunch registry curation guidance](https://scicrunch.org/scicrunch/about/Curation%20Guidelines)
includes research software and source code. It requests the resource's name,
URL and description, and generally registers an individual/root resource rather
than every version or binding. The [resource suggestion page](https://scicrunch.org/create/resource)
and [RRID resource page](https://rrid.site/about/resource) explicitly distinguish
suggesting a resource from curator approval and RRID assignment.

**Local assessment:** Innovate is a plausible candidate in the research software
category. Eligibility has not been accepted by a curator. The draft below
describes the repository's current architecture and its qualification limits;
it does not claim a released native product or complete scientific acceptance.

| Submission field | Draft value |
| --- | --- |
| Resource name | Innovate |
| Canonical URL | `https://github.com/edithatogo/innovate` |
| Resource type | Software library; research software |
| Description | Software library for modelling innovation adoption, technology substitution, competition and policy diffusion with a Rust scientific core and Polars tabular interfaces. The active development line provides thin Python, Julia and R bindings; scientific migration and release qualification remain in progress, and unsupported native capabilities return explicit errors. |
| Suggested keywords | Innovation diffusion; policy diffusion; technology adoption; competition; substitution; forecasting; Rust; Polars |
| Licence | Apache-2.0 for project-owned code; preserve third-party notices |
| Existing identifier | Unverified; do not supply a placeholder RRID |

This packet is ready for a duplicate check and maintainer review of its factual
description. It has not been submitted. A version-specific software citation and
an RRID identify different aspects of the resource; retain version information
when citing software rather than creating a new resource entry for each release.
See the [RRID FAQ](https://rrid.site/about/faqs).

### Duplicate check and remaining RRID gates

Bounded public web searches for the exact repository URL and the combination of
Innovate with diffusion did not yield a verifiable matching record. This is
incomplete discovery, not proof of absence. A direct read-only request to the
[SciCrunch tools search](https://scicrunch.org/resources/Tools/search?q=innovate&l=innovate)
returned HTTP 403 with a browser challenge at `2026-11-14T13:48:39Z`.
The [normalized lookup receipt](evidence/rrid-lookup-20261114.json) records that
limitation without retaining challenge tokens or HTML. No access challenge was
bypassed. Similarly named resources, or records for other repositories maintained
by the same author, cannot establish Innovate's identity.

1. Complete the registry duplicate check using resource name and canonical URL,
   inspecting potential matches. A blocked search must remain unverified.
2. If an entry exists, confirm the repository identity and pursue an update or
   ownership claim through the provider when appropriate; avoid duplicate entry.
3. If a new suggestion is submitted, retain a provider submission receipt and
   its status. A local packet or successful HTTP request alone is not approval.
4. Record curator approval, rejection, or questions. Only an authoritative
   decision establishes the provider's eligibility/acceptance outcome.
5. If assigned, verify the `RRID:SCR_...` resolver record matches Innovate and its
   canonical repository, then update citation guidance with the real identifier.

No submission, curator decision, RRID assignment, or matching resolver response
is evidenced here. Keep #400 open while those external gates remain, unless a
documented ineligibility decision satisfies the track's acceptance rule.

## Verification boundary

Local validation can check metadata consistency, required fields, source links,
identifier syntax and completeness of recorded evidence. It cannot establish
curator approval, archival ingestion, scientific validity or release acceptance.
Unavailable lookups must retain an unavailable state; missing evidence must never
be converted into a passing external-completion claim. This dossier does not
change JOSS acceptance, which remains separately tracked in #401.
