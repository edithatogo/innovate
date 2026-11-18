# Scientific, registry and journal reconciliation — 16 November 2026

**Latest continuation:** [separate evidence acquisition and concrete integration](evidence-acquisition-20261116.md).
It records current main b29a1d27, new historical archives, public history, actual
current-main installed qualification, the completed PDF build and the #671
documentation handoff. Dated observations below retain their original scope.

**Later main advance:** [the integrated-source reassessment](integration-20261116.md)
records the external merge of #656 at `57d2916586d107c8affe90355e453abe27f01c26`,
the reviewed manuscript update and fresh affected execution. It supersedes
the initial source snapshot below; earlier receipts keep their original scope.

**Executed follow-up:** [implementation and evidence](implementation-20261116.md)
now record the manuscript integrity command, real BibTeX/official TeX processing,
blocked PDF compilation, archival target inventory and bounded installed native
qualification. Those results supersede preparation-only wording below within
their recorded scope. External, author, scientific and integration gates remain.

Status: repository preparation in progress; all five issues remain open.
Initially inspected software revision: `69b674814513e09a683279c679b1584af8bb4214`.
Development version: `0.7.0a1`; no submission version or qualified release is
selected by this record. The related
[scientific alignment reconciliation](../scientific_ecosystem_alignment_20261029/reconciliation-20261116.md)
owns the current native qualification summary.

## Issue and track contract

| Issue | Responsibility | Completion evidence |
| --- | --- | --- |
| [#556](https://github.com/edithatogo/innovate/issues/556) | Existing scientific ecosystem alignment: native implementation, installed consumers, exact provider/dependency qualification and integration | Source/artifact-bound native, optional Arrow, platform and provider receipts accepted by the existing integrator; remaining scientific and release obligations explicitly reconciled |
| [#398](https://github.com/edithatogo/innovate/issues/398) | This registry track: coordinate metadata, archival, root-resource identity and JOSS preparation | Child dispositions supported by authoritative evidence, current metadata/links and verified project placement; a plan or green metadata check is insufficient |
| [#399](https://github.com/edithatogo/innovate/issues/399) | Archive selected source/release with Software Heritage | Exact origin, tag/commit, successful visit/snapshot and matching resolving SWHID; request acceptance alone is insufficient |
| [#400](https://github.com/edithatogo/innovate/issues/400) | Assess and register the root Innovate resource for an RRID | Identity-aware duplicate assessment, actual curator disposition and correct resolving identifier; eligibility remains provisional until externally decided |
| [#401](https://github.com/edithatogo/innovate/issues/401) | Prepare and progress the JOSS software-paper deliverable | Qualified software/manuscript, research-use and development-history evidence, author declarations, actual submission/review and final archive/publication records as applicable |

The native parent-child edges `398 -> 399/400/401` were read back through
GitHub's sub-issues API. #556 is a dependency, not a duplicate registry subissue.
Project membership has not been verified through an available project surface.

## Reconciled source and historical records

- [PR #402](https://github.com/edithatogo/innovate/pull/402) is **closed and
  unmerged**; July comments calling it open remain historical. The substantial
  14 November registry dossier, checker and corrected citation metadata already
  exist on current main and are retained.
- [PR #554](https://github.com/edithatogo/innovate/pull/554) is merged as
  `e0b764896704c27562f66c8891e74b5f8582b9ed`. Its Python-era qualification is
  historical; current source uses a Rust core and Polars with Python/Julia/R
  interfaces.
- [PR #602](https://github.com/edithatogo/innovate/pull/602) merged the Arrow
  outcome correction as `2c9ce0fda410161961209ddbf19d99515732e3c9`.
  The [38-test Arrow receipt](https://github.com/edithatogo/innovate/issues/556#issuecomment-6907498117)
  belongs to `44cb9635`, not the current source/provider combination.
- The four input fingerprints in [readiness.json](readiness.json) still match
  the current citation, licence, package manifest and migration ledger. Their
  bytes were inspected and rechecked before refreshing `assessed_revision`;
  the unchanged hashes are retained. This metadata assessment is not a native
  runtime or complete dependency assessment.
- [JOSS draft](joss/paper.md), [bibliography](joss/paper.bib),
  [preparation notes](joss/README.md) and [claim register](joss/claim-evidence.md)
  form a new, unapproved preparation edition. They do not replace the historical
  `paper.qmd` or imply readiness to submit.
- Legacy `ready` / `ready_for_review` inventories under `docs/source/_static`
  are pre-0.7 evidence. The current pending record takes precedence. The active
  documentation owner decides the serial retirement of legacy content; this
  packet neither imports legacy claims nor deletes historical evidence.

## Ordered work and acceptance

These are work packages within the existing tracks, not new tracks or owners.

| Package | Dependencies and owner | Concrete next action | Evidence needed to finish |
| --- | --- | --- | --- |
| REG-16: current reconciliation | This track and #556; current bounded writer | Update track records, draft and five issues, then hand the scoped PR to #624 | Exact diff, metadata checks, source/citation checks, independent review and hosted-check disposition |
| JOSS-01: research-use and public-history evidence | Can start now; maintainer supplies factual use evidence | Identify actual analyses/publications/workflows using Innovate and their revisions; assemble public release/issue/PR chronology spanning active iterative development | At least one evidenced actual research workflow, with version/commit and relevant outputs; public availability/history evidence, not just repository creation date or aspirations |
| JOSS-02: current software-paper edition | Draft exists; depends on JOSS-01 for impact claims and #556/Q01–Q05 for qualified scope | Resolve each claim, update manuscript and bibliography against selected accepted source, retain negative/unsupported results outside permitted claims | Versioned claim register, checked references, 750–1750-word six-section paper, source/metadata hashes, current JOSS rendering receipt |
| RRID-01: root-resource identity | Independent of scientific release; registry owner | Complete live duplicate check using canonical URL and name; reconcile aliases and prepare curator packet from registry-evidence.md | Search/identity receipt; approved resource description; no version or binding-specific duplicate entries |
| SWH-01: origin and target preparation | Origin checks may run now; qualified target and rights assessment before release-archival completion | Select actual release/tag/commit; record history and third-party-rights scope; recheck origin and prepare archival request | Immutable target identity and rights disposition; later a real request receipt, successful visit/snapshot and object resolution |
| QUAL-01: scientific and distribution qualification | Existing #556/#624 and empirical owners | Obtain exact-source optional Arrow, Kairos, installed-platform, scientific, security and release evidence; resolve each existing obligation | Accepted native/artifact receipts; no transfer of older passes or generic CI into scientific acceptance |
| DOC-01: reproducible teaching and installation | Existing documentation owner #659; selected native artifacts | Qualify current examples, public-data provenance and install instructions; reconcile pending Astro/marimo plans | Executed examples and browser/build receipts, actual dataset admission and accurate documented scope; tutorial existence alone is insufficient |
| JOSS-03: author and submission decision | JOSS-01/02, QUAL-01 and applicable DOC-01 completed | Authors confirm authorship, affiliations, funding/COI, rights, full AI history and human validation, then decide whether to submit | Explicit author-confirmed record; no generated attestation or inferred submission authority |
| EXT-01: external outcomes and closeout | Separate authority for the relevant external action | Record actual provider/editorial outcomes and final accepted source/manuscript versions; update identifiers only after verification | Provider receipts, final archive DOI where required, resolving identifiers and publication record; then close corresponding issues and reconcile parent |

RRID eligibility/identity assessment, source archival preparation, research-use
collection and native qualification can progress in parallel. Neither an RRID nor
a Software Heritage identifier is a stated prerequisite to initial JOSS
submission. JOSS's final software archive DOI belongs to successful review/final
acceptance; this repository additionally requires its own scientific/release
gates before claiming a qualified submission scope. Those are different rules.

## Publication boundaries and improvements

The existing [empirical publication track](../empirical_validation_publication_20261107/index.md)
and [V01–V05 taskbook](../../innovation-taskbook-20261107.md#validation-and-publication--empirical_validation_publication_20261107)
remain authoritative for research studies. Their existing dependencies and
ownership are not reassigned here.

| Output | Contribution to demonstrate | Existing evidence owner |
| --- | --- | --- |
| JOSS software paper | Reusable research software, supported design and actual use | #401, consuming accepted #556 and Q01–Q05 evidence |
| Forecasting comparison | Out-of-sample performance, calibration and failures under frozen time/vintage splits | V01–V03; keep failures in denominators and avoid future saturation or revised-data leakage |
| Innovation, competition and hype study | Falsifiable mechanism or comparative empirical contribution | Advanced/hype tracks and V03; descriptive fits do not establish causal effects |
| Medicines application | Correct units, temporal mappings, observation model and application validity | M01–M05; prescriptions, expenditure and adoption counts remain distinct |
| Decision-method paper or reusable dataset | Decision usefulness or reusable, lawfully distributable data with reproducible provenance | V05 and the appropriate Voiage/application/data owner; separate claims and author approval |

This preserves distinct research questions and prevents the software paper from
claiming unexecuted applications, benchmark leadership or empirical validity.
The JOSS comparator discussion uses primary publications; it is not a measured
feature/performance comparison. The erroneous legacy `mordaunt2020budget`
bibliography entry stays flagged for its V05 owner; it is not reused in the new
draft and no claim is made that the correctly identified genomic paper used
Innovate.

## Current official requirements and evidence limits

Official JOSS [submission guidance](https://joss.readthedocs.io/en/latest/submitting.html),
[paper guide](https://joss.readthedocs.io/en/latest/paper.html), and
[AI policy](https://joss.readthedocs.io/en/latest/policies.html) were rechecked.
The 14 November dossier already captured the current requirements correctly:
six sections, normally 750–1750 words, more than six months of public iterative
development, demonstrated research use and good open-source practices. Community
engagement beyond the author is positive evidence, not a mandatory extra human
approval. The author must supply truthful AI tools/version/scope and human-review
declarations. JOSS author conversations with editors/reviewers remain human-led
under its policy, except permitted translation assistance.

[Software Heritage guidance](https://www.softwareheritage.org/how-to-archive-reference-code/)
and [FAQ](https://www.softwareheritage.org/software-heritage-faq/) distinguish
archival objects and contextual citation identifiers. The existing checker's
revision/release SWHID rule is deliberately scoped to #399's release-archival
proof. A contextual directory SWHID may additionally support source citation;
it does not replace target-release linkage. Lightweight tags need not create
release objects.

[SciCrunch curation guidance](https://scicrunch.org/scicrunch/about/Curation%20Guidelines)
supports research software and generally registers the root resource, without
version numbers in its name. The canonical URL and a curator's identity decision
matter more than a name-only match. Submission or a provisional identifier does
not establish final approval and resolving identity.

Fresh read-only [provider observations](evidence/provider-lookups-20261116.json)
returned HTTP 404 for the canonical Software Heritage origin and `.git` variant,
and HTTP 403/non-JSON for SciCrunch search. These observations establish neither
permanent archival absence nor an RRID duplicate-search result. No Save Code Now
request, registration, journal correspondence, deposit or submission occurred.

## Integration and review boundary

The sole integration coordinator remains
[issue #624](https://github.com/edithatogo/innovate/issues/624).
[PR #656](https://github.com/edithatogo/innovate/pull/656) modifies the empirical
plan, root registry and scientific implementation; it proposes track moves.
[PR #659](https://github.com/edithatogo/innovate/pull/659) and its existing owner
retain documentation work. At inspection neither PR modifies the two tracks
changed here. Reconcile relative links if track moves land before this packet.

The baseline registry-hygiene check reports the existing three completed tracks
still under active paths: ODE, NUTS and CI supply-chain hardening. Their shared
registry/archive correction belongs to #624/#656. No status or scientific
acceptance is changed to make the checker pass. This packet does not modify root
CI, manifests, locks, public schemas, shared registry, scientific code or legacy
manuscripts.

## Verification

Executed source checks and independent review are recorded in
[verification-20261116.md](verification-20261116.md). Provider/author/editorial
gates remain pending even if every preparation check passes.
