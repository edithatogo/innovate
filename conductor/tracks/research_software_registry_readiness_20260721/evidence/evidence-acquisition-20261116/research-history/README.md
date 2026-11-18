# Research use and public development history — 16 November 2026

## Result and scope

**Public package-distribution history and a dated development chronology are now
assembled. Actual version-bound research use remains unverified in the inspected
public sources.** The new evidence includes ten PyPI artifacts across five
versions, a reproducible issue/PR census, and the source/data identities of an
executed public-data teaching example. The separate registry investigation also
verified existing Software Heritage archives of the 0.5.0 and 0.6.0 PyPI source
distributions. These are useful, distinct additions to the JOSS-01 evidence.

This bounded acquisition is complete. It does not make an editorial eligibility
decision or close JOSS-01, any of the 38 scientific release obligations, author
declarations or submission gates. The official
[JOSS submission criteria](https://joss.readthedocs.io/en/latest/submitting.html)
require public development over time and actual research use; an old timestamp
or an executable example alone does not settle those assessments.

The inspection used checkout `2748084292e5e57c181c1665a686ad3a291145bf`, tree
`787eb4cd175b7343026cdad88143e8879e0b3062`, incorporating main
`b29a1d27f54fdd08d77881ee999772fceab7c03a`. The observed PR #662 head was
`d36c33f6b4fb480828ddb9aae0121f3379667015`; PR #671 was inspected at
`070a04424dfb084758d1c0b201bf2980002ea81d`. Both were open at inspection.
Only public sources were read and this evidence directory written. No analysis
was rerun, private workspace accessed, person contacted or provider submission
made in this lane.

## Evidence inventory

| Record | What it preserves |
| --- | --- |
| [PyPI history](pypi-history.json) | Exact artifact URLs, provider upload timestamps, filenames, SHA-256 digests and version-specific project identity; HTTP response identities and retrieval dates. |
| [Source-distribution inspection](pypi-sdist-inspection.json) | Downloaded 0.5.0/0.6.0 artifact hashes, sizes and immutable embedded package metadata; no extraction or execution. |
| [GitHub history](github-history.json) | Seven representative PRs, source context, the 17-month issue/PR summary, native manifest history and interpretation limits. |
| [Activity ledger](github-activity-ledger.json) | The 671 normalized returned records needed to reproduce the creation-month census; this is not a commit census. |
| [Historical-event attempt](github-public-event-20250707.json) | The actual HTTP 403 from the one bounded GH Archive request; no public event was recovered. |
| [Research-use candidates](research-use-candidates.json) | Three inspected candidates, source links, findings, missing evidence and three bounded public-repository searches. |
| [Teaching-workflow identity](teaching-workflow-identity.json) | Successful byte comparisons for the lesson source, frozen raw data and extracted observations. |
| [Registry dossier](../registry/README.md) and [archive chain](../registry/swh-chain-summary.json) | Independently verified PyPI-origin Software Heritage records and their exact correspondence with the PyPI distribution metadata. |

The earlier [JOSS evidence record](../../../joss/README.md) and
[GitHub release inventory](../../archive-target-inventory-20261116.json) remain
the records of the preceding searches and release/tag inspection. This packet
adds evidence rather than relabeling those earlier checks.

## Public package availability

The public [PyPI API](https://pypi.org/pypi/innovate/json) returned five versions,
each with a wheel and source distribution. The table uses each source
distribution's upload timestamp; the inventory retains the separate wheel
timestamps, immutable file URLs and provider-recorded hashes. All ten entries
were non-yanked at retrieval.

| Version | Source distribution uploaded, UTC | Repository identity in currently served version-specific metadata |
| --- | --- | --- |
| [0.1.0](https://pypi.org/project/innovate/0.1.0/) | 7 July 2025, 03:43:42.054398 | No project URL or homepage field; retained as an earlier entry in the same PyPI project. |
| [0.1.1](https://pypi.org/project/innovate/0.1.1/) | 7 July 2025, 22:47:48.152726 | No project URL or homepage field. |
| [0.2.1](https://pypi.org/project/innovate/0.2.1/) | 29 August 2025, 09:46:39.561671 | No project URL or homepage field. |
| [0.5.0](https://pypi.org/project/innovate/0.5.0/) | 29 August 2025, 09:46:41.108181 | Homepage identifies `https://github.com/edithatogo/innovate`. |
| [0.6.0](https://pypi.org/project/innovate/0.6.0/) | 1 November 2026, 11:58:32.680487 | Homepage, source and issue URLs identify the canonical repository. |

The strongest repository-identified package anchor in this inventory is thus
0.5.0 on 29 August 2025. The July records provide earlier same-project package
history, with the metadata limitation shown above. These provider upload dates
are stronger distribution evidence than an author-supplied Git commit date.
The index metadata was read now; this lane did not reconstruct what every
project field displayed on each historical upload date. A subsequent bounded
artifact inspection closes the digest and embedded-metadata gap for the two
source distributions below; other inventory artifacts retain provider-recorded
digest evidence only.

Both named source distributions returned HTTP 200. Recomputed SHA-256 values
match the PyPI inventory and archived original-artifact metadata: 0.5.0 is
107,028 bytes and 0.6.0 is 185,910 bytes. Each immutable top-level `PKG-INFO`
names `innovate`, the corresponding version, the canonical GitHub repository,
`License-Expression: Apache-2.0` and `License-File: LICENSE`. Metadata hashes and
the exact fields are retained in the source-distribution inspection. Archives
were read directly without extraction or code execution and remain outside the
Git worktree. This establishes artifact identity and embedded declarations;
it does not prove Git equivalence, repository visibility history or research use.

Publication channels must remain separate. GitHub's v0.2.1 release is dated
12 July 2025 in the earlier inventory, whereas its PyPI files were uploaded on
29 August. The absence of a GitHub v0.6.0 published release in that inventory
does not negate the verified PyPI 0.6.0 upload.

The [Software Heritage chain](../registry/swh-chain-summary.json) provides an
additional archival observation: the PyPI origin's full visit on
1 November 2026 links to resolving synthetic release objects for 0.5.0 and
0.6.0, with original-artifact URLs and hashes matching the PyPI inventory.
Those archive objects are PyPI-loader releases targeting directories, not Git
tags. No package-to-Git-commit byte equivalence, current native qualification or
research adoption follows from this chain.

## Iterative development chronology

The [retained GitHub records](github-history.json) show work at multiple dates:

| Public record | Provider event | Evidence of the work recorded |
| --- | --- | --- |
| [PR #60](https://github.com/edithatogo/innovate/pull/60) | Merged 7 November 2025 | Backend changes, paper-development consolidation and dependency updates. |
| [PR #78](https://github.com/edithatogo/innovate/pull/78) | Merged 9 July 2026 | Batched network-diffusion prediction work. |
| [PR #554](https://github.com/edithatogo/innovate/pull/554) | Merged 1 November 2026 | Python 3.15 and Arrow dependency qualification. |
| [PR #577](https://github.com/edithatogo/innovate/pull/577) | Merged 8 November 2026 | Native-runtime and PR-history consolidation. |
| [PR #602](https://github.com/edithatogo/innovate/pull/602) | Merged 14 November 2026 | Rejection of incomplete scenario outcomes before Arrow export. |
| [PR #662](https://github.com/edithatogo/innovate/pull/662) | Created 16 November 2026; open at inspection | Scientific and journal-readiness qualification. |
| [PR #671](https://github.com/edithatogo/innovate/pull/671) | Created 16 November 2026; open at inspection | Native learning resources, including the public-data lesson below. |

Seven API pages yielded **671 unique issue/PR records: 414 PRs and 257 issues**.
The final page contained 71 records and no returned IDs were duplicated.
Pages were read during one investigation, not as a transactional snapshot.

| Creation month | PRs | Issues | Entries from classified automated actors |
| --- | ---: | ---: | ---: |
| 2025-07 | 50 | 0 | 4 |
| 2025-08 | 2 | 0 | 2 |
| 2025-09 | 3 | 0 | 3 |
| 2025-10 | 2 | 0 | 2 |
| 2025-11 | 7 | 0 | 6 |
| 2025-12 | 2 | 0 | 2 |
| 2026-01 | 0 | 0 | 0 |
| 2026-02 | 0 | 0 | 0 |
| 2026-03 | 0 | 0 | 0 |
| 2026-04 | 0 | 6 | 6 |
| 2026-05 | 1 | 0 | 1 |
| 2026-06 | 0 | 1 | 1 |
| 2026-07 | 162 | 178 | 214 |
| 2026-08 | 50 | 17 | 39 |
| 2026-09 | 23 | 12 | 35 |
| 2026-10 | 29 | 19 | 41 |
| 2026-11, through inspection | 83 | 24 | 50 |

The automated-actor count uses GitHub's `Bot` type, logins ending `[bot]`, or
the `Copilot` login. It neither detects all automation nor establishes human
authorship for other accounts. The concentration in July and November 2026
must remain visible when reviewers assess the distribution of work. A month
with no new issue/PR does not establish that no commits occurred.

Current public issue and PR objects expose provider event timestamps, but
their historical timestamps alone do not prove that the repository was public
at every corresponding date. One concrete attempt to recover a contemporaneous
public event from [GH Archive](https://www.gharchive.org/) used the hour of the
known 7 July 2025 GitHub release. It returned HTTP 403; the
[failure receipt](github-public-event-20250707.json) preserves that limit.
No archived public event or uninterrupted visibility is asserted.

### Historical project versus native line

The remote mainline history for `rust/crates/innovate-core/Cargo.toml`, anchored
to main `b29a1d27…`, returned 11 commits. Its oldest returned commit is the
8 November 2026 consolidation `95579544…`, corroborated by PR #577. This
identifies a mainline integration event, not the first private or branch
development of Rust code. The earlier Python project's distribution history
must not be presented as six months of qualification for the rewritten native
0.7 line. The separate installed/native receipts retain their own exact source,
artifact, compiler and platform scope.

## Actual research-use candidates

### Genomic testing budget-impact study

The [publisher](https://www.valueinhealthjournal.com/article/S1098-3015%2825%2905624-4/fulltext),
[PubMed](https://pubmed.ncbi.nlm.nih.gov/41106549/) and newly found
[Flinders institutional record](https://researchnow.flinders.edu.au/en/publications/genomic-testing-in-australia-a-budget-impact-analysis-using-diffu/)
identify *Genomic Testing in Australia: A Budget Impact Analysis Using Diffusion
Modeling From a Healthcare System Perspective*, DOI
[10.1016/j.jval.2025.09.3067](https://doi.org/10.1016/j.jval.2025.09.3067).
Accessible records describe a research workflow using historical Australian
MBS claims and diffusion/time-series projections. They do not establish the
Innovate revision, executable analysis environment or mapping from code to
published outputs.

The institutional record exposes a final-publication PDF, a new route beyond
the earlier DOI searches. That PDF returned HTTP 403 through both retrieval
routes attempted. Publisher full text and ScienceDirect were also inaccessible;
indexed extracts were not treated as full-text inspection. The result is
**unverified implementation lineage**, not a finding that the paper did not use
Innovate. The needed record remains the actual analysis archive with its
software identity, data provenance, commands and output-to-paper linkage.

### Tobac integration assessment

The public [Tobac progress record](https://github.com/edithatogo/tobac/blob/891f79e07292b7754f78d7193130623bd5f9e243/conductor/tracks/nz_au_empirical_20261106/phase4-progress.md)
at commit `891f79e0…` names Innovate source `b9998add…` and PyPI 0.6.0. It
explicitly records that Innovate imports, calls, fit recovery and numerical
parity were unexecuted. Its separate Voiage execution used synthetic fixtures.
This is evidence of a considered integration, not executed Innovate research
use. Its historical pandas conflict is not attributed to the current Rust
0.7 implementation.

### Executed WDI/ITU teaching workflow

[PR #671](https://github.com/edithatogo/innovate/pull/671) contributes a concrete
public-data example. Its
[dataset manifest](https://github.com/edithatogo/innovate/blob/070a04424dfb084758d1c0b201bf2980002ea81d/docs/datasets/wdi-aus-internet-20261116/manifest.json)
records Australian Internet-use observations from ITU via World Bank WDI,
retrieved on 16 November 2026, source update 11 November, and a CC-BY-4.0 license
claim. The indicator measures recent Internet use, not cumulative first
adoption. The frozen series preserves missing years 2002, 2003, 2004 and 2025,
uses elapsed calendar years, and divides 23 observed training rows through
2015 from nine holdout rows in 2016–2024.

The [retained execution receipt](https://github.com/edithatogo/innovate/blob/070a04424dfb084758d1c0b201bf2980002ea81d/conductor/tracks/documentation_learning_20261116/evidence-20261116/hosted-b2ed945/execution.json)
for implementation source `b2ed945f…` reports a successful
`examples/native_public_data.py` run. Its holdout mean absolute errors, in
percentage points, are **12.916 for Gompertz, 15.994 for logistic and 8.410 for
persistence**. The persistence comparator performed better in this example.
This investigation independently fetched the source, frozen raw JSON and CSV
at inspected head `070a044…`; all three SHA-256 identities match the producer's
records. It did not rerun the analysis or requery the live World Bank snapshot.

The inspected [lesson](https://github.com/edithatogo/innovate/blob/070a04424dfb084758d1c0b201bf2980002ea81d/docs/astro-site/src/content/docs/tutorials/public-data-fit.mdx)
is teaching and exploratory work. Its 2026 data vintage makes the split a
retrospective holdout, not an as-of-2015 forecast. One indicator and country do
not establish general forecasting, inference or causal performance. This is a
useful source-bound executable example; it is not reclassified here as
research-use acceptance.

## Search bounds and remaining evidence

The new public routes included the institutional publication/PDF record,
PyPI version metadata, related public consumer repositories, public scholarly
and data-repository searches, and the newly executed lesson. Indexed
default-branch searches for `innovate` returned no matches in
`edithatogo/reimbursement-atlas`, `edithatogo/microcosting_genomics` and
`edithatogo/microsim_oa`. These bounded negative results do not cover all
branches, private work or the wider research literature. No private mail,
Drive, workbooks or analysis folders were accessed.

The existing research owner can supply the missing factual use record when
available: the research question, actual Innovate revision or artifact,
analysis source and environment, input provenance and rights, executed
commands, outputs and their connection to the publication or analysis.
Historical Python use, if verified, should be recorded as such. The public
history packet is ready for human/editorial assessment with the visibility,
activity-distribution and implementation-line limits retained. This acquisition
does not generate an author attestation or turn intended integrations,
synthetic checks, package downloads or examples into research adoption.

## Local record validation

On 16 November 2026, all seven JSON documents parsed successfully. Recalculation
from the 671-entry ledger reproduced every one of the 17 monthly rows, the
414/257 PR/issue totals and zero duplicate IDs. All relative Markdown links
resolved. The three retained teaching-workflow checks each report HTTP 200 and
an exact expected/observed digest match. All three research-use candidates
remain explicitly unaccepted. The PyPI inventory remained unchanged at SHA-256
`047e25c4836a24ecddfa5506479501976e81277657bacb041c0c836b83f288c4`, matching
the input fingerprint used by the independent archive comparison. Both
downloaded source distributions match their preserved artifact digests and
contain the expected package identity fields. These are record and artifact
identity checks, not a rerun of scientific or release qualification.
