# Software Heritage and RRID evidence acquisition — 16 November 2026

## Result and scope

**Innovate's historical PyPI source distributions 0.5.0 and 0.6.0 have existing
Software Heritage archives with resolving release SWHIDs.** This pass found and
verified the provider's origin, full visit, snapshot references, synthetic release
objects, resolvers and original-artifact metadata. The recorded tarball URLs,
filenames and SHA-256 digests match the separately acquired
[PyPI inventory](../research-history/pypi-history.json).

The GitHub origin and Git release objects remain a separate evidence gap. RRID
duplicate discovery remains incomplete because its live search routes returned
403; indexed registry records found in this pass identify other resources.

This packet was prepared in checkout
`2748084292e5e57c181c1665a686ad3a291145bf`, tree
`787eb4cd175b7343026cdad88143e8879e0b3062`, incorporating main
`b29a1d27f54fdd08d77881ee999772fceab7c03a`. The initial live issue readback found
#398–#401 open and PR #662 draft/unmerged at
`d36c33f6b4fb480828ddb9aae0121f3379667015`. Only this evidence namespace was
written. No provider submission, registration, login, challenge bypass or curator
correspondence occurred.

## Verified historical distribution chain

The [chain summary](swh-chain-summary.json) is the compact machine-readable
entry point. Its raw responses and request receipts preserve exact URLs, HTTP
statuses, provider dates, response byte counts and SHA-256 hashes.

| Distribution | Resolving release identifier | Release target |
| --- | --- | --- |
| PyPI 0.5.0 | [swh:1:rel:104cbcae703902978703a64b129d0dd8dd2c31b8](https://archive.softwareheritage.org/api/1/resolve/swh:1:rel:104cbcae703902978703a64b129d0dd8dd2c31b8/) | Directory `4e1c8dddd8bb2554f7830123faff277aeea1f27b` |
| PyPI 0.6.0 | [swh:1:rel:bdb01a82e50284e59c4f09f670f5d94ddd91051d](https://archive.softwareheritage.org/api/1/resolve/swh:1:rel:bdb01a82e50284e59c4f09f670f5d94ddd91051d/) | Directory `03ce01ed9f43a580277665e01a806db8d2b5d061` |

The [PyPI origin](https://archive.softwareheritage.org/api/1/origin/https://pypi.org/project/innovate//get/)
returned 200. Its [visit collection](swh-pypi-visits.response.json) contains three
full visits, with no pagination link. The latest
[visit 3](swh-pypi-visit-3.response.json) is dated
`2026-11-01T15:40:38.107000+00:00` and links to
[snapshot 537bb104…](swh-pypi-snapshot.response.json).
The snapshot has six branches, no next branch, and named references
`releases/0.5.0` and `releases/0.6.0` targeting the release objects above.
`HEAD` aliases `releases/0.6.0`.

Both [0.5.0](swh-pypi-release-0.5.0.response.json) and
[0.6.0](swh-pypi-release-0.6.0.response.json) release responses explicitly mark
`synthetic: true` and `target_type: directory`. These are the archive's PyPI
loader objects. Their
[0.5.0](swh-pypi-resolve-0.5.0.response.json) and
[0.6.0](swh-pypi-resolve-0.6.0.response.json) resolver responses returned 200 on
16 November at provider Date 13:28:03 GMT, confirming the namespace, scheme,
release object kind and exact object IDs.

The archive's original-artifact records for
[0.5.0](swh-pypi-original-artifacts-0.5.0.response.json) and
[0.6.0](swh-pypi-original-artifacts-0.6.0.response.json) supply these source
distribution hashes:

| File | Provider-recorded SHA-256 |
| --- | --- |
| `innovate-0.5.0.tar.gz` | `c8687b0ce2d08a55549d3707a7120847e8433fe0d51ae577fadba1ec94cb7961` |
| `innovate-0.6.0.tar.gz` | `ea0ce110499320871e568d5107d468f00f93185005a1909248de12f3b1bf8be2` |

The corresponding metadata lists bind each record to its release, directory and
PyPI origin. The independent PyPI inventory confirms the same file URLs, names
and hashes, and its version-specific 0.5.0 and 0.6.0 metadata name the canonical
Innovate repository. This comparison checks provider metadata correspondence;
this lane did not download the tarballs or recompute the archived directory
objects.

These facts support historical distribution archival. They do not establish
equivalence to Git commit `6b673832…` or `3459f452…`, the annotated Git tag
`86eceddc…`, or the current native 0.7 source. PyPI reports no provenance for the
0.6.0 files on its [release page](https://pypi.org/project/innovate/0.6.0/).
No release target or rights assessment has been selected for the track, and this
packet does not alter the current-version readiness record.

## Other Software Heritage observations

[Initial lookup observations](public-lookup-observations.json) preserve nine
normalized direct-request outcomes. The prior two exact GitHub origin queries
were not blindly repeated.

- The v0.5.0 Git revision, v0.6.0 annotated Git tag object and v0.6.0 Git revision
  each returned 404 for the tested object at the recorded response time.
- A **GET** of the canonical Git origin's existing Save Code Now requests
  returned 404 with no requests found. No POST was made.
- Searches for `edithatogo` and `edithatogo/innovate` timed out. A timeout is
  unavailable discovery, not a zero-result search.

Those negative Git/object observations coexist with the positive PyPI origin.
They cannot support a general claim that Innovate is absent from Software
Heritage. The official
[origin-search](https://archive.softwareheritage.org/api/1/origin/search/doc/),
[visit](https://archive.softwareheritage.org/api/1/origin/visits/doc/),
[snapshot](https://archive.softwareheritage.org/api/1/snapshot/doc/) and
[save-request](https://archive.softwareheritage.org/api/1/origin/save/doc/)
documentation supplied the route semantics used here.

## RRID identity review

[RRID identity review](rrid-identity-review.json) records the exact bounded
queries, identity terms, live limitations and disambiguated results. Queries
covered the canonical repository URL, the PyPI identity, Innovate/innovate,
`edithatogo`, and the public author metadata. No returned primary registry
result identified this Innovate repository.

The alternate public RRID searches for the owner and resource name both returned
403/non-JSON at provider Date `Mon, 16 Nov 2026 13:23:53 GMT`. The suggestion
page was also unavailable through public retrieval. No challenge HTML, cookies
or tokens were retained.

Two indexed first-party records were useful for excluding false matches:

| Indexed record | Why it does not identify Innovate |
| --- | --- |
| [KairoECS, SCR_029085](https://rrid.site/data/record/nlx_144509-1/SCR_029085/resolver?i=rrid%3Ascr_029085&q=) | Its resource URL is `https://github.com/edithatogo/kairos`, a different repository and resource under the same owner. |
| [CHOP IDEA Lab, SCR_029073](https://rrid.site/data/record/nlx_144509-1/SCR_029073/resolver) | The matching word occurs in a core facility's name; its URL is `https://www.research.chop.edu/idea-lab`. |

Direct opening of those record pages returned 403, so the indexed information
is not represented as a successful live resolver check. No matching RRID,
curator eligibility outcome or completed duplicate check was obtained. A
pre-release record for another project does not establish Innovate's eligibility.

## Concrete next inputs and integration

[Prepared requests](prepared-requests.json) supplies a canonical origin-level
Git archive request and a factual RRID root-resource draft, without executing
either. For a new Git request, the unresolved decisions are the exact archival
target/scope and rights constraints. Existing PyPI ingestion requires no new
request merely to report these observed records.

The RRID packet includes name, canonical URL, package identity, description,
candidate resource types, keywords and licence scope. Remaining user facts are
the submitter/current contact permitted for this suggestion, confirmation of the
description, and any earlier private submission or known identifier/alias.
Accessible live duplicate discovery and the actual provider form remain
necessary before a new suggestion. The final form's required field schema could
not be inspected in this pass.

The integrator can now record **verified historical PyPI archival evidence** and
replace the blanket evidence-gap description with these scoped results, while
preserving the earlier dated Git-origin negatives. The current native target,
Git-source archival, rights and RRID curator gates remain open. No shared
readiness, plan, citation or issue state was edited by this lane.

[Verification](verification.json) records the local check of all 15 retained
raw-response receipts, exact response hashes/lengths, the complete linkage for
both release objects, and their correspondence with the PyPI inventory.
