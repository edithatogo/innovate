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
- The current draft describes development version `0.7.0a1` at
  `34b4868c094742495fb7f89f5410676917e7a05b`. The
  [source reassessment](claim-evidence.md#security-integration-source-reassessment-on-26-november-2026)
  records the dependency and ledger changes. The earlier
  [adopter receipts](../../empirical_validation_publication_20261107/evidence/adopter-classification-20261120/README.md)
  and render receipts retain their original revisions.

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
| Rust scientific computation, Python/Julia/R interfaces, sixteen bounded families and 38 required obligations | [README](../../../../README.md), [migration ledger](../../../../migration-ledger.json) |
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

Actual PDF inspection exposed one name-formatting issue: Pandoc treated `ter`
in the Mesa author's surname as a dropping particle. The publisher's citation
explicitly uses `ter Hoeven et al.`. BibTeX brace protection now preserves that
form as a non-dropping particle, verified by a fresh parse, citeproc and PDF
inspection. The reference identity and scientific claim are unchanged; the
[prior inputs](../evidence/evidence-acquisition-20261116/render/attempt-3/inputs/)
and [visual finding](../evidence/evidence-acquisition-20261116/render/attempt-3/visual-review.json)
remain available with the render evidence.

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
4. Render and inspect the 20 November edition before proceeding to submission.
   The completed BibTeX, TeX/PDF and four-page inspection below apply to the
   16 November edition, with the recorded template layout and metadata limits.
5. Re-run relevant installation, scientific, documentation and release gates on
   the selected source/artifacts. Structural manuscript checks do not establish
   submission readiness, scientific validity or a qualified software release.
6. Record an author decision to submit, followed by actual submission/editorial
   evidence. Complete the final software archive DOI and accepted publication
   record at the appropriate review stage. Local files are not provider receipts.

No external submission, author/editor/reviewer correspondence, publication,
release, archive deposit or registry registration was performed for this packet.

## Draft checks performed

For the initial `69b6748` edition, an inline Python source check counted
**1,074 prose words** from Summary through Acknowledgements, excluding section
headings and citation tokens. Including the visible draft notice and headings,
that edition contained **1,136 words** under the same word-token rule. The
post-merge edition's current count is recorded by the integrity check and render
receipt below; neither count is an editorial word-count decision.

The original drafting increment parsed the YAML front matter with PyYAML,
checked the six required section headings, matched all five cited keys to the
bibliography, resolved its then-current 16 relative links and checked whitespace.
The follow-up below adds an actual bibliography parse and a bounded official
render attempt. These checks concern the draft's structure and build, not
acceptance of its research claims or the software.

## Repeatable integrity and render checks

Run the dependency-free integrity check from the repository root:

```sh
python scripts/check_joss_readiness.py --json
```

This checks the source snapshot's 14 implementation/documentation hashes and
three manuscript-edition hashes, the software version/revision identifiers, the
visible draft notice, the six required sections and the recorded prose-count
rule. A changed source hash requires reassessing the relevant manuscript claims
and their evidence before deliberately updating the snapshot. Do not regenerate
hashes merely to clear a failure. A changed manuscript edition likewise requires
reviewing the claims and citation mapping before refreshing its hashes.

The checker does not parse YAML or BibTeX, render a PDF, qualify the science, or
supply author approval. Its explicit readiness gate is expected to fail for this
draft:

```sh
python scripts/check_joss_readiness.py --require-submission-ready
```

### Official Inara build

[render.py](render.py) uses the local Makefile route documented by the
[JOSS paper guide](https://joss.readthedocs.io/en/latest/paper.html) and
[Inara](https://github.com/openjournals/inara). It requires an unchanged Inara
checkout at `88ee5d656f147d801448ce77ad8375619d383df6` and Pandoc **3.6.4**, matching
the Pandoc version in that revision's
[Dockerfile](https://github.com/openjournals/inara/blob/88ee5d656f147d801448ce77ad8375619d383df6/Dockerfile).
These primary-source rendering instructions were checked on **16 November 2026**.

Install or provide Git, Python, GNU Make, Pandoc 3.6.4, `latexmk` and a working
LuaLaTeX installation with the upstream Dockerfile's packages and fonts. This
includes the LuaTeX support packages, New Computer Modern and Hack; the upstream
image also bundles Libre Franklin. TeX formats, font discovery and a writable
font cache must be configured. The wrapper installs nothing and makes no
system-wide environment changes. Paths passed to the upstream Make recipes must
exclude spaces and shell metacharacters. This local wrapper targets a POSIX
environment, including Linux under WSL; it rejects native Windows drive and UNC
paths. The retained execution was on Linux. Portable orchestration tests do not
qualify native Windows rendering.

For an already provisioned local toolchain, use a new isolated checkout and
output directory. Replace the Pandoc path with the installed 3.6.4 executable:

```sh
joss_tools="$(mktemp -d /tmp/innovate-joss.XXXXXX)"
git init "$joss_tools/inara"
git -C "$joss_tools/inara" fetch --depth 1 https://github.com/openjournals/inara.git 88ee5d656f147d801448ce77ad8375619d383df6
git -C "$joss_tools/inara" checkout --detach FETCH_HEAD
python conductor/tracks/research_software_registry_readiness_20260721/joss/render.py \
  --inara "$joss_tools/inara" \
  --pandoc /path/to/pandoc-3.6.4/bin/pandoc \
  --output-dir "$joss_tools/render" \
  --source-date-epoch 1794787200
```

The epoch above is 16 November 2026 at 00:00 UTC and reproduces this attempt's
draft date. Record the selected epoch for each later attempt. Matching only the
Inara and Pandoc revisions does not promise byte-identical PDFs across different
TeX distributions, fonts or platforms.

The wrapper first runs the offline integrity check and stops if it fails. It then
parses `paper.bib` through Pandoc's BibTeX reader into CSL-JSON and runs upstream
`make tex` and `make pdf` with citeproc. It explicitly passes
`--metadata=draft:true` and `--variable=draft:true`, the draft settings used by
Inara's entrypoint, because the local Makefile does not itself supply them.
It checks the generated TeX for the draft watermark, line numbering and visible
working-draft notice. It uses the unmodified official template and does not
enable production mode.

Outputs include a receipt with commands, input/output hashes, software revision,
tool versions and separate stage outcomes. An existing output directory is
rejected to protect earlier evidence. Failure returns a non-zero exit code and
preserves the completed stages and log; PDF build success still leaves visual
review pending. Inspect every resulting PDF page for clipping, missing glyphs,
citations and the draft notice before recording visual acceptance. The renderer's
editor/reviewer/DOI placeholders are template defaults, not editorial records.

The official Docker route remains available on a Docker-capable host. Building
the pinned checkout's unmodified Dockerfile provides its intended package
environment; record the resulting image identity and use draft mode. No Docker
build or run was executed in this workspace, where Docker is unavailable.

### Completed draft render: 16 November 2026, after #668

The [render evidence packet](../evidence/evidence-acquisition-20261116/render/README.md)
contains the [four-page PDF from that edition](../evidence/evidence-acquisition-20261116/render/attempt-4/paper.pdf),
its [unchanged generated receipt](../evidence/evidence-acquisition-20261116/render/attempt-4/receipt.json),
the [separate visual review](../evidence/evidence-acquisition-20261116/render/attempt-4/visual-review.json)
and a [reproduction recipe](../evidence/evidence-acquisition-20261116/render/reproduce.md).
The PDF is 207,798 bytes, SHA-256
`14adfb6b79bd16c26fef5f3a704cae487fdc2d883e7c1e010a203b69c932c5b0`.

Source/edition integrity passed with fourteen source inputs, three edition
fingerprints, six required sections and **1,076 prose words**. Actual bibliography
parsing, official TeX and PDF compilation passed, including the corrected
`ter Hoeven` citation. All four pages were visually inspected after rasterizing
at 150 dpi. The draft/source notices, five reference DOI links, `Sîrbu` diacritic
and literal `Rht` author are present and readable.

The official template's line numbers crowd/overlap the license sidebar around
lines 16–19, and the last reference continues alone onto page four. Dummy DOI,
submitted date, editorial and footer fields remain placeholders; the visible
author metadata remain provisional and the PDF Info Author field is empty.
These observations are retained explicitly. This is a readable working draft,
not a zero-overlap or submission-layout pass. The generated receipt remains
`rendered_draft_visual_review_pending`; the later visual record supplies its
separate result without rewriting the renderer's generated facts.

The successful environment uses the pinned Inara/Pandoc versions above, a
complete selected TeX Live 2024 base/recommended LaTeX environment, LuaHBTeX
1.18.0 and latexmk 4.86a. All 241 official package archives were checksum checked.
The original helper versions and environment are preserved; subsequently guarded
reproduction helpers passed fifteen focused guard tests and an actual fresh
setup replay. That replay is separate from the PDF's original build identity.
No scientific, release, author or editorial gate changes follow from these results.

### Earlier blocked attempt: 16 November 2026, after #656

The [render receipt](../evidence/joss-render-20261116/attempt-3/receipt.json) binds
the reassessed `57d2916586d107c8affe90355e453abe27f01c26` edition. It records Pandoc
3.6.4, GNU Make 4.4.1, latexmk 4.86 and LuaHBTeX 1.18.0 (TeX Live
2025/dev/Debian), with the pinned Inara source above. The complete
[command log](../evidence/joss-render-20261116/attempt-3/render.log) preserves the failure.
The [prior edition's attempt](../evidence/joss-render-20261116/attempt-2/receipt.json)
and [initial wrapper attempt](../evidence/joss-render-20261116/receipt.json) are
preserved unchanged with their actual script and input identities. Their checks
do not establish the changed edition's identity or build result.

| Stage | Executed result |
| --- | --- |
| Offline integrity prerequisite | Passed with all 14 source and three edition hashes matching, six required sections and 1,076 prose words. Submission readiness remains false. |
| Actual BibTeX parse | Passed with five records and no Pandoc warnings; retained [CSL-JSON](../evidence/joss-render-20261116/attempt-3/bibliography.json). The parser preserved the `Sîrbu` diacritic and the literal author `Rht`. |
| Official template and citeproc | Passed; retained [draft TeX](../evidence/joss-render-20261116/attempt-3/paper.tex) contains all five resolved references, the working-draft notice and the draft watermark/line-numbering commands. |
| PDF compilation | **Blocked**: LuaLaTeX reported `File luacode.sty not found`; upstream Make returned 2 and the wrapper returned 1. No PDF was produced. |
| PDF visual inspection | **Not performed** because there is no PDF. Source-level draft markers are not visual inspection. |

During the earlier setup, system Pandoc 3.1.11.1 lacked an Inara template partial;
the isolated official Pandoc 3.6.4 binary resolved that mismatch. The workspace's
TeX image also lacked its file database, format, Lua support packages and some
fonts. A bounded, isolated setup used the installed distribution's files,
packages from the historic TeX Live 2024 repository named in Inara's Dockerfile,
and Hack v3.003. A process-local `TEXMFCACHE=.` allowed font-cache creation in the
isolated Inara checkout. Setup stopped at the remaining `luacode.sty` dependency;
no global TeX configuration or other worker's environment was changed. The
post-merge attempt reused that environment without installing more dependencies.
The later acquisition pass above used a complete isolated LaTeX base, retained
new receipts and completed the draft's page inspection. This earlier failure
remains an accurate record of its own environment and edition.

The wrapper's 18 focused tests in
[test_joss_render.py](../../../../tests/native/test_joss_render.py) passed using
explicit synthetic subprocess fixtures. They exercise version/checkout guards,
unsafe and reused output paths, the integrity prerequisite, mandatory draft
markers, failed PDF receipts and source changes or deletion during a build.
Host-independent path-text cases test the actual production restriction with
POSIX paths and rejected Windows drive/UNC paths. Mocked orchestration uses host
temporary files to model the POSIX Make environment without requiring a real
renderer on each CI platform. A
synthetic successful process still leaves visual review and submission pending;
these fixtures are not real renders or scientific acceptance evidence. Ruff's
`E,F,I,UP,B` checks and formatting passed for the wrapper and tests.

This follow-up used ChatGPT with GPT-6 Astra Pro for the wrapper, source checks
and build diagnostics. It does not complete the whole-project AI inventory or
the authors' review declaration. A parser or build pass does not establish
research use, scientific validity, a qualified release, author consent,
submission readiness or journal acceptance.
