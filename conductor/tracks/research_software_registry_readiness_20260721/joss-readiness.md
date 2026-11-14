# JOSS submission readiness

Status: `blocked_prerequisites`; no submission or acceptance claimed.

Owning issue: [#401](https://github.com/edithatogo/innovate/issues/401), under
[#398](https://github.com/edithatogo/innovate/issues/398).
Repository baseline inspected: `276c2e505a0ce26910820fe8a94a6a498f505700`.
Requirements and public bibliographic sources checked on **14 November 2026
(Australia/Brisbane)**. This is a preparation dossier, not a JOSS editorial
assessment or an author-approved manuscript.

## Current requirements

The authoritative sources are the JOSS
[submission guidance](https://joss.readthedocs.io/en/latest/submitting.html),
[paper format](https://joss.readthedocs.io/en/latest/paper.html),
[review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html), and
[review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html).
Recheck them before submission; this dated assessment is not a permanent policy
snapshot.

| Area | Requirement and evidence to prepare |
| --- | --- |
| Access and licensing | Publicly browseable and cloneable source, public issue tracking, a contribution route and an OSI-approved license. See the submission guidance. |
| Development | More than six months of public, iterative development and evidence of open-source practice. Repository creation date alone does not establish when the source became public. See the submission guidance. |
| Research use | Identify actual research use, at least by the developers, and its evidence. Aspirational applications alone do not satisfy the submission screening gate. See the submission guidance. |
| Software | Feature completeness, appropriate packaging, usable documentation, installation and functionality checks. Review remains a JOSS decision. See the submission guidance and review criteria. |
| Paper | Markdown with YAML metadata, normally 750–1750 words, covering the six sections listed below; author affiliations, references and funding acknowledgement also need attention. See the paper format. |
| Review evidence | Explain alternatives, design trade-offs and research significance; provide reproducible functionality checks. See the review criteria and checklist. |

The paper guidance permits an impact statement supported by credible near-term
significance, such as reproducible materials. That wording does not remove the
submission guidance's separate requirement to demonstrate research use. Both
must be addressed; this dossier makes no eligibility ruling.

JOSS requires an AI usage disclosure identifying the tools/models and versions,
where and how they were used, and the authors' review and validation of those
outputs. The authors must confirm their own decisions and responsibilities.
No such attestation is supplied here. The submission guidance also reserves
author interactions with editors and reviewers to humans, except AI-assisted
translation.

## Baseline evidence and gaps

All file findings below describe the inspected baseline. Later corrections must
be reviewed against their own source revision; the historical findings should
not be read as a claim that a corrected file still has the defect.

| Evidence | Observed state | Required next step |
| --- | --- | --- |
| [README](../../../README.md), [package manifest](../../../pyproject.toml), [product context](../../product.md) | The development version is `0.7.0a1`: Rust scientific computation, Polars tables, and Python/Julia/R bindings. README retains 38 scientific obligations and an intentionally failing release gate. | Choose the submission version and qualify its complete documented scope, installation and release artifacts. Existing scientific obligations remain required. |
| [paper.qmd](../../../paper.qmd) | Historical long-form manuscript; no JOSS `paper.md` was found. It includes API documentation and application results rather than the current six-section format. | Prepare a separate, source-qualified software paper; do not relabel this file as submission-ready. |
| [paper.qmd](../../../paper.qmd) implementation and performance sections | Claims Mesa ABM, NumPy/JAX numerical backends, scipy optimization, pandas tables, the earlier class API and historical backend benchmark numbers. | Reconcile claims with the current Rust/Polars implementation and ownership boundaries. Historical results do not qualify a different implementation. |
| [paper.qmd](../../../paper.qmd) applications and sustainability sections | Numerical genomic uptake forecasts and a general `>90%` coverage claim lack current native-version evidence in the proposed packet. | Retain as historical material until provenance, commands, inputs and exact-source results are qualified; omit unsupported claims from the new paper. |
| [manuscript_v2.md](../../../documents/manuscript_v2.md), [abstract_v2.md](../../../documents/abstract_v2.md) | Describe the earlier Python/Mesa design. The manuscript attributes examples to unspecified NIH/FDA/Gartner studies using organization homepages. | Establish actual source identity and rights before reusing those examples or representing them as research-use evidence. |
| [CITATION.cff](../../../CITATION.cff) | At baseline, Python-era description, version `0.5.0` and all-zero ORCID placeholder. | Review corrected metadata, distinguish a development version from a released version, and confirm author information. |
| [references.bib](../../../references.bib) | The `mordaunt2020budget` entry has a mismatched DOI, detailed below. | Correct bibliographic identity before reusing the entry and check remaining citations individually. |
| Current submission evidence | No JOSS review URL, editor acknowledgement, accepted paper DOI or source-bound archive receipt was identified in this track. | Keep submission and acceptance pending until authoritative evidence exists. Absence in this inspection is not proof that no external record exists. |

License presence, tests in the repository and selected CI passes are useful
preparation evidence. They do not establish complete scientific validity,
publication readiness or JOSS acceptance. See the repository's
[release gates](../../release-gates-20261107.md) and
[migration ledger](../../../migration-ledger.json).

## Historical readiness records

For the current native candidate, this dossier supersedes the following earlier
JOSS readiness assertions. Their files remain unchanged as historical records:

- `docs/source/_static/scientific_submission_packet.json`: JOSS
  `status: ready_for_review` and a packet pointing to the legacy manuscripts.
- `docs/source/_static/community_submission_readiness_matrix.json`: JOSS
  `readiness_status: ready` with an empty blockers list.
- `docs/source/_static/external_submission_target_inventory.json`: JOSS
  `status: ready_for_review`.

The existing Astro operations pages already label the earlier registry work as
pre-0.7 material. Historical `ready` flags are not evidence for the current
version, and should not be promoted into new submissions without requalification.

Issue #401's July comment calls PR #402 open. The live GitHub record inspected
for this review shows [PR #402](https://github.com/edithatogo/innovate/pull/402)
closed and unmerged on 8 November 2026. Some planning content is present on the
current baseline; neither that presence nor the old PR comment proves external
submission or acceptance.

## Bibliographic identity finding

The `mordaunt2020budget` entry in `references.bib` assigns DOI
`10.1016/j.jval.2020.06.001` to a genomic budget-impact paper. The
[publisher record for that DOI](https://www.valueinhealthjournal.com/article/S1098-3015%2820%2932125-2/fulltext)
identifies **Mayer and colleagues' 2020 study of valuation methods for general
practitioner consultation costs**. The
[PubMed record](https://pubmed.ncbi.nlm.nih.gov/32940231/) corroborates the mismatch.

The relevant genomic diffusion paper by **Mordaunt, Stark, Elshaug and Schilling**
has DOI **`10.1016/j.jval.2025.09.3067`**, *Value in Health* **29(3), 375–382
(March 2026)**, with online publication in October 2025. See its
[publisher record](https://www.valueinhealthjournal.com/article/S1098-3015%2825%2905624-4/fulltext)
and [PubMed record](https://pubmed.ncbi.nlm.nih.gov/41106549/).

These records support bibliographic correction and research context. This review
did not establish that the paper used Innovate, which revision it used, or that
its results validate the current native implementation. Such a use claim needs
an explicit source/version link and reproducible evidence. Legacy results and
manuscript text have not been rewritten by this dossier.

## Actionable outline for a separate paper.md

Create the new manuscript alongside its BibTeX and figure sources on a branch
containing the selected software revision. The following is a content plan,
not final prose or a human author attestation.

| Section | Content to prepare | Evidence boundary |
| --- | --- | --- |
| Summary | Explain adoption, substitution and policy-diffusion modelling for a nonspecialist reader. Identify the Rust core and supported language interfaces. | Describe only the selected version's documented and qualified capabilities. |
| Statement of need | State the research problems, intended users and the difficulties motivating a reusable implementation. | Separate documented motivation from demonstrated software use. |
| State of the field | Compare named, versioned alternatives on the specific modelling tasks; explain why extending existing tools was insufficient. | Cite primary software/paper sources. Do not reuse unqualified feature checkmarks from the old manuscript. |
| Software design | Explain the rationale and trade-offs for one Rust computation core, Polars/Arrow interchange and thin bindings; describe Innovate/Kairos/Voiage boundaries. | Tie architecture statements to source and contracts, including copies, unsupported operations and distribution limits. |
| Research impact statement | Identify actual research workflows, publications or integrations and the software revisions used; give reproducible supporting material. | A planned integration, citation to background research or internal test pass does not itself demonstrate research adoption. |
| AI usage disclosure | Compile the actual tools, versions when known, tasks and scope of assistance across software, documentation and paper preparation. | Authors must verify the record and supply their own review, validation and design-decision confirmation. Do not invent a no-AI or fully-human-reviewed statement. |
| Acknowledgements and references | Confirm funding, contributors, affiliations and citations; use a checked bibliography. | Do not infer current institutional affiliation or funding from an old manuscript. |

Before requesting author review, check YAML metadata, bibliography resolution,
local links, required sections and word count, and render a PDF using the current
JOSS-supported route. Preserve the command, tool versions, source revision and
output hash. Formatting success is a manuscript-build result only.

## Gates and completion evidence

Keep #401 open until its agreed deliverable has authoritative evidence or a
documented eligibility decision. The parent track must distinguish preparation,
submission, editorial acceptance and archive/identifier resolution.

1. **Repository preparation:** current manuscript and metadata; checked sources;
   explicit supported scope; executable installation, functionality and release
   evidence for the chosen revision; publicly inspectable development history.
2. **Author confirmation:** authorship, affiliations, funding, research-use
   linkage, rights, AI disclosure, human review and decision to submit. A
   generated dossier cannot supply these attestations.
3. **Submission:** actual JOSS submission and a review issue URL or editor
   acknowledgement, bound to the submitted source/version and manuscript.
4. **Review and acceptance:** the JOSS editor/reviewers' decision and resulting
   authoritative record. Repository tests do not replace that decision.
5. **Archive and publication:** the final accepted software release, its archive
   DOI and resolving archive record, and the published paper record when issued.

Do not trigger a release, deposit, external submission, account action or author
message from a readiness check. Record newly obtained evidence and its scope
before changing the corresponding state.
