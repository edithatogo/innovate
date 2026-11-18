# Separate evidence acquisition and integration — 16 November 2026

This is the current operational record for #398–#401 and the scientific
dependency #556. It supersedes earlier pending-source/provider observations
within the bounds below and preserves all dated receipts. Shared integration
remains with the sole integrator in #624; documentation work has moved from
closed #659 to the same owner's active draft #671.

## Source and ownership

- Starting published scientific/journal head:
  `d36c33f6b4fb480828ddb9aae0121f3379667015` in draft #662.
- Consumed main: `b29a1d27f54fdd08d77881ee999772fceab7c03a`, including the #668
  Rust documentation/coverage repair and the separate #667/#666/#670 updates.
- Local integration: `2748084292e5e57c181c1665a686ad3a291145bf`, tree
  `787eb4cd175b7343026cdad88143e8879e0b3062`. Native inputs are separately
  verified against public main; local and later publication commit IDs differ.
- Allowed changes: the two owning scientific/journal track namespaces and the
  new builder guard tests. Main's source, manifests and pins are consumed
  unchanged. The new complete workflow is a track-local candidate, not an edit
  to the integrator's active workflow namespace.

## Evidence acquired

| Work | Acquired result | Limit retained |
| --- | --- | --- |
| Historical Software Heritage archival | Both PyPI 0.5.0 and 0.6.0 have verified origin → full visit → snapshot → release → resolving SWHID chains; provider original-artifact metadata matches PyPI | These are package-loader releases targeting directories; no Git-tag/commit equivalence or native 0.7 archival is inferred |
| Public distribution and development history | Five PyPI versions/ten artifacts, dated PR anchors, and a reproducible 671-record issue/PR census; the repository-identified package anchor is 29 August 2025 | Current visibility and old event dates do not prove uninterrupted historical public visibility; the newer native rewrite keeps its own scope |
| Actual research-use investigation | Three concrete candidates assessed; the public-data lesson has verified source/data identities and a retained successful run | No inspected candidate establishes actual version-bound research-use acceptance; published research, intended integration and teaching have different evidence |
| Installed native Arrow | Fresh current-main development wheel, 178 verified native inputs, **42 passes**, all four optional Arrow cases executed, zero skips | Linux development artifact; other platforms, release artifacts and full scientific validity are separate |
| Declared Rust minimum | Actual Rust **1.99.1** all-target workspace compile passes for all five workspace crates | Compile only, Linux/default features; games crate, test execution and linking excluded |
| Recurring integration | Implemented observed-build helper, 24 guard tests and a complete Arrow/MSRV workflow candidate | Hosted adoption and promotion remain with #624 |
| Hosted evidence | Formerly pending d36 platform/coverage checks now completed; current main's Rustdoc repair verified; #671's substantive source passes 164 browser checks and complete documentation contract | Exact checkout/head distinctions and skipped cases retained; later commits need their own gates |
| RRID | Additional identity-aware queries, false-match discrimination and concrete unsubmitted metadata | Live duplicate discovery remains inaccessible; no curator outcome or matching Innovate RRID verified |

### Existing historical archives

The [authoritative registry packet](evidence/evidence-acquisition-20261116/registry/README.md)
and [chain summary](evidence/evidence-acquisition-20261116/registry/swh-chain-summary.json)
retain 15 provider responses with hashes, statuses and timestamps. Both resolving
release identifiers were independently fetched again by the integrating agent
and matched the retained responses:

| Distribution | Verified Software Heritage release |
| --- | --- |
| PyPI 0.5.0 | `swh:1:rel:104cbcae703902978703a64b129d0dd8dd2c31b8` |
| PyPI 0.6.0 | `swh:1:rel:bdb01a82e50284e59c4f09f670f5d94ddd91051d` |

The earlier exact GitHub-origin 404 responses remain historical observations;
they are not a general absence claim. The selected target, rights scope and
current native/Git-source linkage remain open in the current-version readiness
record. No new archive request or registry submission was made.

### Public history and research-use distinction

The [research-history packet](evidence/evidence-acquisition-20261116/research-history/README.md)
preserves all ten immutable PyPI artifact records and distinguishes the July 2025
same-project entries from the August 2025 canonical-repository anchor. Two source
distributions were additionally downloaded without extraction or execution:
their recomputed SHA-256 values and embedded name/version/repository/licence
metadata agree with the PyPI/SWH evidence.

The 671-entry activity ledger reproduces 414 PRs, 257 issues and 17 monthly
rows, including periods of concentrated activity. Its records are not a commit
census or proof of past public visibility. The 8 November 2026 native
consolidation is not assigned the older Python line's maturity.

The genomic budget-impact publication is real, but its analysis-to-Innovate
version linkage remains unverified. Tobac explicitly records an unexecuted
Innovate integration. #671's WDI/ITU lesson has executed, source-bound teaching
evidence; its persistence comparator outperforms the two fitted curves on its
retrospective holdout. Those observations do not supply an author assertion of
research use, broad model performance or causal validity.

## Manuscript and render

The [reviewed source delta](evidence/evidence-acquisition-20261116/source-reassessment.md)
checks all 14 inputs against current main. Only the assessed tabular source
changed, adding 36 Rustdoc comments with unchanged executable text. The paper's
visible assessment and claim/source register were deliberately advanced; the
**1,076-word** prose, five references and withheld claims retain their scope.
Visual inspection also identified and corrected a BibTeX particle issue in
`ter Hoeven`, using the publisher's stated citation. The old PDF and bibliography
are retained and a fresh edition is rendered; no scientific prose was changed.
All 14 source and three edition fingerprints pass. Four registry-input hashes
also match current-main Git blobs and local bytes in the
[readiness source check](evidence/evidence-acquisition-20261116/readiness-source-verification.json).

The new [render packet](evidence/evidence-acquisition-20261116/render/README.md)
completes compilation with the unchanged pinned wrapper and official toolchain.
The final attempt-4 PDF has **four pages**, SHA-256
`14adfb6b79bd16c26fef5f3a704cae487fdc2d883e7c1e010a203b69c932c5b0`.
All four pages were visually inspected. The standard draft template retains
editorial placeholders and licence text crowding/overlapping sidebar line-number
positions 16–19; these are
recorded draft-layout limitations, not actual submission/editorial facts.
The retained final receipt and visual-review record are the authority for the
PDF artifact and its inspection; the two earlier missing-support-package
failures in this attempt and all prior-turn failures remain preserved.
Rendering does not approve authorship, establish research use or make the
draft submission-ready. Independent review also verified all 241 package
checksums, the copied fonts and owned tool binaries, and required stronger
setup guards for fresh directories, the pinned database and executable/font
identity. The executed original helpers and corrected reproduction validation
are retained separately.

## Concrete integration handoff

The [executed current-main native packet](../scientific_ecosystem_alignment_20261029/evidence/evidence-acquisition-20261116/README.md)
and [complete recurring workflow candidate](../scientific_ecosystem_alignment_20261029/integration/README.md)
make the former recurring-CI request executable. Independent review verified
the wheel, installed package bytes, all 178 native inputs, source/test identities,
42-case JUnit, real build logs and separate MSRV result. The review's one
test-environment portability recommendation was applied and independently
rechecked. No new provider or scientific implementation is copied into Innovate.

The [hosted intake](evidence/evidence-acquisition-20261116/hosted/README.md)
identifies actual checkout SHAs, platform skips, three-crate coverage scope and
the remaining documentation distinction. Main's strict Rustdoc gate now passes;
its browser suite has 141 passes/one failure. #671's substantive source b2ed945,
tested at synthetic merge `96a02ac`, has 164 browser passes and a complete
documentation report. Its later `070a044` evidence commit is not claimed to have
that same complete tree.

There are no changed-file overlaps between #671 and these two track namespaces.
However, #671 changes two **assessed manuscript source inputs**: the native
migration guide and tabular `src/lib.rs`. The preserved disposable drift check
correctly rejects the incoming guide. After the integrator receives both PRs,
review those changes against the claims, deliberately reassess the combined
source, rerun the paper check/render and obtain current native receipts. Do not
auto-update hashes or import old hosted results as new-head acceptance.

## Validation and review

The [final six preparation suites](evidence/evidence-acquisition-20261116/validation/final-focused-tests.json)
record **208 passed, zero skips**: 56 registry, 50 manuscript-integrity,
18 renderer, 45 installed-qualifier, 24 observed-build and 15 environment-guard
cases. The 193-case pre-guard run remains a separate earlier receipt. The actual
PDF compilation, original toolchain verification, corrected setup replay, fresh
42-case installed consumer and MSRV compile are distinct executions, not
inferred from unit tests.

[Independent native/tooling review](../scientific_ecosystem_alignment_20261029/evidence/evidence-acquisition-20261116/independent-review.md)
and [hosted/source review](evidence/evidence-acquisition-20261116/hosted/independent-review.md)
record their tested identities, findings and resolutions. Repository policy and
Conductor hygiene pass within their reported scope. Explicit release readiness
still exits 1 for the 38 scientific obligations; the preparation checkers do not
turn submission or registry completion into a pass.

## Remaining decisions and evidence

| Owner/input | Concrete missing evidence or action |
| --- | --- |
| Research owner | A real analysis archive or run: question, actual Innovate version/artifact, source/environment, data provenance, executed commands, outputs and connection to the paper/workflow; historical Python use should retain that identity |
| Author | Confirmed author list/order, affiliations, funding/conflicts, human review and truthful AI-use declaration; the draft currently supplies no attestation |
| Selected archive/release | Target and rights scope; matching current Git/native archival linkage if that is the intended target |
| RRID submitter/provider | Current contact, any earlier private submission/identifier, accessible duplicate discovery and actual curator/resolver evidence |
| Shared integrator #624 | Receive both owners' PRs in a safe order, adopt recurring CI, verify the actual receiving head and preserve conditional offline-security and scientific gates |
| Scientific/release owners | Close the 38 required migration obligations with their prescribed independent evidence; exact artifacts, provider distribution and release acceptance remain separate |

The issues remain open for those evidenced residuals. Completing this acquisition
does not imply external correspondence, registration, submission, editorial
acceptance, main promotion or a public release.
