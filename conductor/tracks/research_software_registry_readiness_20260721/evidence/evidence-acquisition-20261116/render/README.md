# Executed JOSS draft render and visual inspection

The [current draft PDF](attempt-4/paper.pdf) compiled successfully through the
unchanged official Inara template and was inspected on all four pages.
It is a **readable, unapproved working draft with documented template layout
limitations**, not a submission-ready paper or accepted software release.

The [generated render receipt](attempt-4/receipt.json) records real source,
bibliography, TeX and PDF results. Its original `visual_review: not_performed`
field remains unchanged: the subsequent inspection is recorded separately in
the [visual-review receipt](attempt-4/visual-review.json). The
[PDF content checks](attempt-4/pdf-content-check.json) supplement that page
inspection with exact DOI links, Unicode text and page bounds.
The [independent review](independent-review.md) separately verified the artifact,
setup guards, replay records and bounded visual conclusions.

## Current artifact and scope

| Item | Recorded identity/result |
| --- | --- |
| Assessed software | `b29a1d27f54fdd08d77881ee999772fceab7c03a`, development version `0.7.0a1` |
| Manuscript integrity | Fourteen source and three edition fingerprints, six required sections and 1,076 prose words passed |
| Actual bibliography parse | Five records; all five DOI link targets present in the PDF; no unresolved citation keys |
| Official TeX and PDF | Passed, using the unmodified `joss/render.py` and forced draft metadata/template flags |
| PDF size | Four A4 pages, 207,798 bytes |
| PDF SHA-256 | `14adfb6b79bd16c26fef5f3a704cae487fdc2d883e7c1e010a203b69c932c5b0` |
| Generated receipt SHA-256 | `950353aec45dcab855e53ca0501eff7bfb172b3fc54f7c25a43eaa40b517967b` |
| Visual-review SHA-256 | `a223a0ed5179b816b10cc92093b0ee10510cb00dc4633c00288175b3d94c4665` |

The [source reassessment](../source-reassessment.md) explains the one changed
implementation input: 36 added Rustdoc comments, with the remaining tabular
source bytes unchanged. The manuscript's visible commit changed accordingly.
No scientific claim or required capability was removed. All 38 migration
obligations, research-use requirements and author/editorial gates remain in force.

### What the visual inspection establishes

All four pages were rasterized with Poppler `pdftoppm` 26.05.0 at 150 dpi and
visually inspected. Body text, headings, footers and references are readable;
the unapproved notice and exact source revision appear on page one, and the
draft watermark appears on every page. All five references, their DOI links,
the `Sîrbu` diacritic and literal author `Rht` are present. The corrected Mesa
citation retains `ter Hoeven` in the text and bibliography.

The official draft template has visible limitations. Line numbers around 16–19
crowd/overlap the page-one license sidebar; page four contains only the final
reference and substantial whitespace. Its dummy DOI, 1970 submitted date,
editor/reviewer names and volume/issue/page fields are template placeholders,
not editorial events. The visible author/affiliation remain provisional and the
PDF Info Author field is empty. The template was preserved unchanged. This is
not a zero-overlap, complete-metadata or publication-layout pass. No PDF/A,
tagged-accessibility or other-platform qualification was performed.

## Reproducible environment

The [environment receipt](attempt-3/toolchain/environment-receipt.json) identifies
the environment used for both successful renders: Inara
`88ee5d656f147d801448ce77ad8375619d383df6`, Pandoc 3.6.4, LuaHBTeX 1.18.0 from
TeX Live 2024, latexmk 4.86a and GNU Make 4.4.1. The complete selected base and
recommended LaTeX collections plus the template's additions comprise 241
official archives, 47,572,856 compressed bytes. Every archive's size and SHA-512
was checked against the pinned historic TeX Live database. Font and executable
identities and actual process-local configuration are retained.

The [reproduction recipe](reproduce.md) documents fresh acquisition and the
unchanged wrapper invocation. No system TeX configuration, other worker's
environment or official template was changed. This used the supported local
Makefile route; it did not build/run the official container or exercise its
ConTeXt/JATS outputs.

### Guarded reproduction follow-up

The exact originally executed helper sources are preserved in
[executed-scripts](attempt-3/toolchain/executed-scripts/), matching the hashes
in the original environment receipt. The current reproduction helpers add
checks that reject existing directories and an unpinned database before writing,
compare all copied font and license files with their Git blobs, reject extra
files, and require the configuration's TeX tools inside the owned tree.
The configuration commands use those explicit owned executable paths.

All 15 focused [environment guard tests](../../../../../../tests/native/test_joss_environment.py)
passed on Python 3.15.2, including synthetic negative fixtures for these failure
cases; Ruff checks and formatting passed. These tests do not simulate scientific
acceptance or substitute for a real render. The fixed helpers were then
[executed once in fresh owned directories](guarded-setup/receipt.json), using
the verified archive cache. Package extraction, format creation and font-cache
creation passed; package, font, Pandoc and checked TeX-tool identities match the
original environment. This replay qualifies the guarded setup commands. It did
not create another PDF or retroactively replace the final PDF's environment
receipt.

## Preserved attempts and correction

| Attempt in this directory | Actual result |
| --- | --- |
| [1](attempt-1/receipt.json) | Bibliography and TeX passed; PDF stopped at an undeclared `ifmtarg.sty` dependency. |
| [2](attempt-2/receipt.json) | Bibliography and TeX passed; PDF stopped at the missing base graphics package `epstopdf-base.sty`. |
| [3](attempt-3/receipt.json) | Full base environment produced a PDF. Inspection found a dropped author-name particle; the original PDF, inputs and bibliography are preserved with the [finding](attempt-3/visual-review.json) and [layout clarification](attempt-3/visual-review-followup.json). |
| [4](attempt-4/receipt.json) | Fresh render after the narrow bibliography correction; current PDF and separate four-page inspection above. |

The [publisher's Mesa citation](https://joss.theoj.org/papers/10.21105/joss.07668)
explicitly retains `ter Hoeven`. Protecting that name in BibTeX changed Pandoc's
classification from a dropping to a non-dropping particle and corrected the
rendered citation. The bibliography's DOI, authors, title and scientific use
were unchanged. The preceding source snapshot is retained byte for byte with
the other [attempt-three inputs](attempt-3/inputs/).
These are exact historical input bytes. Relative links inside the retained
`claim-evidence.md` use its original
`conductor/tracks/research_software_registry_readiness_20260721/joss/` location
as their base, not the relocated `inputs/` directory. Preserve that link base
when inspecting the snapshot; its content has not been rewritten.

The [earlier post-#656 attempt](../../joss-render-20261116/attempt-3/receipt.json)
and all earlier receipts retain their original blocked result and source identity.
This preparation work supplies no author consent, research-use record, scientific
or release acceptance, journal submission, editorial correspondence or approval.
