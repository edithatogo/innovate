# JOSS manuscript preparation packet

Status: **unapproved working draft; not submission-ready**.

Owning issue: [#401](https://github.com/edithatogo/innovate/issues/401), under
[#398](https://github.com/edithatogo/innovate/issues/398). Read the parent
[JOSS readiness dossier](../joss-readiness.md) and
[track specification](../spec.md) with this packet.

- [paper.md](paper.md) is a separate software-paper draft for the native development
  line, not a replacement or requalification of the legacy `paper.qmd`.
- [paper.bib](paper.bib) contains five references checked against primary publisher
  or software-project sources on **16 November 2026**.
- Source baseline: `69b674814513e09a683279c679b1584af8bb4214`, development version
  `0.7.0a1`. Refresh the draft when its implementation evidence changes.

## Scope and evidence

The draft describes current implementation boundaries and documented rationale.
It contains no empirical effect estimate, benchmark result, comparative
superiority claim, confirmed research adoption, human author attestation or
external acceptance claim. All 38 migration obligations and existing
scientific/distribution/release gates remain required. A pending capability is
not retired or made optional by excluding it from a manuscript claim.

The author name, ORCID and affiliation reproduce
[CITATION.cff](../../../../CITATION.cff) provisionally. Authors must confirm the
complete author list and affiliations for this publication; existing metadata
does not supply their consent. The paper records this limitation visibly.

| Manuscript claim | Repository evidence at the baseline |
| --- | --- |
| Rust scientific computation; Python/Julia/R interfaces; sixteen bounded families; 38 outstanding obligations | [README](../../../../README.md), [migration ledger](../../../../migration-ledger.json) |
| Model conventions, time/observation semantics, synthetic versus empirical evidence | [Native scientific guide](../../../../docs/astro-site/src/content/docs/native-migration.md) |
| Polars/Arrow boundary; copies and LazyFrame collection; incomplete R/Julia IPC/bundles | [README](../../../../README.md), [tabular crate](../../../../rust/crates/innovate-tabular/Cargo.toml) |
| Reuse of Rust numerical dependencies and pinned Kairos crates | [Core crate manifest](../../../../rust/crates/innovate-core/Cargo.toml) |
| Kairos/Voiage responsibilities and architectural rationale | [Product context](../../../product.md), [README](../../../../README.md) |
| Scientific, platform, distribution and release conditions | [Release gates](../../../release-gates-20261107.md) |
| Outstanding research-use and external-publication evidence | [JOSS dossier](../joss-readiness.md), [readiness record](../readiness.json) |

This is source inspection, not newly executed native, installed-consumer or
empirical qualification. The proposed architecture and implementation boundaries
are distinct from their scientific adequacy in a particular study.

## Reference verification

These checks establish bibliographic identity and support the limited contextual
claims made in this draft. They do not reproduce the cited studies or benchmark
the current versions of their software. No paid full text was acquired.

| Key | Verified identity and primary source | Scope used |
| --- | --- | --- |
| `bass1969` | Frank M. Bass; *A New Product Growth for Model Consumer Durables*; *Management Science* 15(5), 215–227 (1969); [publisher](https://pubsonline.informs.org/doi/10.1287/mnsc.15.5.215); DOI `10.1287/mnsc.15.5.215` | First-purchase diffusion model and innovation/imitation framing. The unusual word order in the title follows the publisher record. |
| `fisher1971` | J. C. Fisher and R. H. Pry; *A simple substitution model of technological change*; *Technological Forecasting and Social Change* 3, 75–88 (1971); [publisher](https://www.sciencedirect.com/science/article/pii/S0040162571800057); DOI `10.1016/S0040-1625(71)80005-7` | Technological substitution context. Bibliographic details and abstract were available through indexed publisher content; direct page access returned HTTP 403. |
| `mesa2025` | Ewout ter Hoeven and colleagues; *Mesa 3: Agent-based modeling with Python in 2025*; *Journal of Open Source Software* 10(107), 7668; [publisher and citation](https://joss.theoj.org/papers/10.21105/joss.07668); DOI `10.21105/joss.07668` | General agent-based modelling, analysis and visualisation. The paper describes Mesa 3.1.5; it is not a benchmark of a later release. |
| `ndlib2018` | Giulio Rossetti and colleagues; *NDlib: a python library to model and analyze diffusion processes over complex networks*; *International Journal of Data Science and Analytics* 5, 61–79; [publisher](https://link.springer.com/article/10.1007/s41060-017-0086-6); DOI `10.1007/s41060-017-0086-6` | Network diffusion framework. The bibliography uses the volume year 2018; online publication was 20 December 2017. |
| `pymc2023` | Oriol Abril-Pla and colleagues; *PyMC: a modern, and comprehensive probabilistic programming framework in Python*; *PeerJ Computer Science* 9, e1516 (2023); [publisher](https://peerj.com/articles/cs-1516/), [project citation and description](https://www.pymc.io/index.html); DOI `10.7717/peerj-cs.1516` | Bayesian model specification and fitting. Publisher content was available through indexed results; direct publisher access returned HTTP 403. Author list/DOI were cross-checked against the project's citation. |

The old `mordaunt2020budget` citation is deliberately not reused. The parent
dossier records its bibliographic mismatch; a corrected background citation
would still require independent evidence before being represented as Innovate
research use.

## Bounded research-use and development-history inspection

A 16 November 2026 search of the exact repository URL, paper DOI, publisher and
public scholarly records did not establish an Innovate version/commit used in
a published or preprint analysis. This is a bounded search result, not proof
that no such use exists.

The author's [genomic diffusion paper](https://pubmed.ncbi.nlm.nih.gov/41106549/)
is identifiable as Mordaunt, Stark, Elshaug and Schilling, *Value in Health*
29(3), 375–382 (2026), DOI `10.1016/j.jval.2025.09.3067`. It is relevant research
context, but the accessible bibliographic record does not establish an Innovate
analysis lineage. Direct publisher/PDF access was unavailable; indexed
software-description leads were not promoted into verified implementation-use
claims. Obtain the actual analysis repository/archive, commit, environment and
mapping to published outputs. If these establish earlier Python-version use,
record it as historical research use and qualify the native implementation
separately. A shared research topic or a corrected citation is not that linkage.

Current first-party GitHub release metadata provide concrete chronology anchors:

| Release | Recorded publication time (UTC) |
| --- | --- |
| [v0.1.1](https://github.com/edithatogo/innovate/releases/tag/v0.1.1) | 2025-07-07T22:50:34Z |
| [v0.2.1](https://github.com/edithatogo/innovate/releases/tag/v0.2.1) | 2025-07-12T04:12:49Z |
| [v0.3.0](https://github.com/edithatogo/innovate/releases/tag/v0.3.0) | 2025-07-31T03:15:34Z |
| [v0.4.0](https://github.com/edithatogo/innovate/releases/tag/v0.4.0) | 2025-07-31T11:39:23Z |
| [v0.4.1](https://github.com/edithatogo/innovate/releases/tag/v0.4.1) | 2025-07-31T11:41:43Z |
| [v0.5.0](https://github.com/edithatogo/innovate/releases/tag/v0.5.0) | 2025-08-29T09:23:10Z |

All six records are currently non-draft and non-prerelease. The
[Bayesian tutorial PR #2](https://github.com/edithatogo/innovate/pull/2) was
created on 12 July 2025 and merged that day, with head
`b9e87ceeb3dc9093cd5e1a23b8e5d531e226fbac`. Later source/issue history is
preserved in the scientific track. These are useful iterative-development
anchors beyond repository creation, but do not independently prove continuous
public visibility, complete activity distribution, community engagement or
JOSS eligibility. Retain that assessment as pending until the submission
history packet is reviewed.

## Current JOSS requirements

The [paper guide](https://joss.readthedocs.io/en/latest/paper.html) specifies
750–1750 words and the six required sections included here. The
[submission guidance](https://joss.readthedocs.io/en/latest/submitting.html)
also requires demonstrated research use and public iterative development.
The [review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html)
and [policies](https://joss.readthedocs.io/en/latest/policies.html) require
independent attention before submission. These sources were checked on
16 November 2026; recheck them at the decision to submit.

For this drafting increment, ChatGPT with GPT-6 Astra Pro assisted with primary
source checks and drafting. The whole project's AI history remains unknown;
the draft requests a complete, truthful tool/version/task inventory and the
authors' own human-review and design-decision confirmation. It supplies no such
attestation. JOSS prohibits AI conversational interactions between authors and
editors/reviewers except translation support. The maintainer must conduct those
interactions; this packet does not authorise or automate journal correspondence.

## Remaining preparation and submission gates

1. Reconcile every claim against the selected software revision and qualified
   scope without removing existing scientific obligations or weakening gates.
2. Complete the actual research-use record and public-development evidence.
   Existing empirical-study owners retain their protocols and acceptance duties.
3. Confirm authorship, affiliations, funding, conflicts, rights and the full AI
   disclosure. Replace draft-specific pending statements only with verified facts.
4. Render the manuscript with the current JOSS-supported toolchain and inspect
   the PDF, citations and metadata. Record tool versions, source revision and
   output hash. Rendering was **not executed** by this drafting increment.
5. Re-run relevant installation, scientific, documentation and release gates on
   the selected source/artifacts. Structural manuscript checks do not establish
   submission readiness, scientific validity or a qualified software release.
6. Record an author decision to submit, followed by actual submission/editorial
   evidence. Complete the final software archive DOI and accepted publication
   record at the appropriate review stage. Local files are not provider receipts.

No external submission, author/editor/reviewer correspondence, publication,
release, archive deposit or registry registration was performed for this packet.

## Draft checks performed

An inline Python source check counted **1,074 prose words** from Summary through
Acknowledgements, excluding section headings and citation tokens. Including the
visible draft notice and headings, the body contains **1,136 words** under that
word-token rule. Both are within the current guidance; this is not the JOSS
renderer or an editorial word-count decision.

The YAML front matter parsed successfully with PyYAML. All six required section
headings were present; all five cited keys matched the five bibliography entries;
all 16 relative links resolved; and no trailing whitespace was found in the three
packet files. A full BibTeX parser and JOSS PDF rendering were not executed.
These are structural checks of the draft only, not acceptance checks for its
research claims or the software.
