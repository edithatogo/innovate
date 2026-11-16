# Independent PDF preparation and reproduction review

Reviewed on 16 November 2026 by the independent review agent. This is an
assistant review, not author approval or a human editorial review. The worktree
HEAD observed during review was
`2748084292e5e57c181c1665a686ad3a291145bf`; the manuscript assesses published
source revision `b29a1d27f54fdd08d77881ee999772fceab7c03a`. The reviewed pending
files are identified individually in [the structured review](independent-review.json).

**No unresolved blocking finding remains in the scoped setup-helper changes.**
The final four-page PDF is a readable draft with two recorded template-layout
observations. It does not receive publication-layout, author, scientific,
release or submission approval from this review.

## Findings and corrections

Three preparation defects were reported to the helper author before publication.
The author made the changes; this reviewer did not edit the helpers or their
tests.

| Finding | Correction and independently checked evidence |
| --- | --- |
| The package builder could overwrite a retained database before rejecting an invalid input. A temporary-directory reproduction observed this behavior. | The builder now rejects an existing setup root, including a symlink, and checks the pinned database SHA-256 and package selection before writing. Two actual CLI regressions now reject an existing root and an unpinned database with exit 1; retained bytes remain unchanged and a fresh parent is not created. The shell recipe also refuses an existing tools path before an atomic `mkdir`. |
| A pinned checkout revision did not itself verify the copied font/license bytes or exclude additional font files. | Configuration now compares the complete copied-file inventory and bytes against Git blobs and rejects unsafe, missing, changed or extra files. Tests cover changed font/license bytes, missing fonts and untracked/ignored extras. All 24 original configured fonts independently matched their pinned Git blobs; the guarded replay produces matching font hashes. |
| A missing owned TeX executable could allow a setup command to fall back to the host PATH. | The helper now requires executable paths to resolve inside its supplied TeX tree and invokes those owned paths directly. Tests reject missing, outside-tree and nonexecutable tools. The replay records actual owned command paths and successful format/font-cache execution. |

The inspected [builder](build_pdf_environment.py) verifies exact archive sizes
and SHA-512 values before extracting with the `tarfile` data filter. The
[reproduction commands](reproduce.md) pin the database and Pandoc archive
hashes, Inara revision and Hack revision, then use fresh installation,
configuration and render-output directories. The documented Linux x86_64
prerequisites and process-local TeX environment are consistent with the
observed local execution. Full network reacquisition and other operating
systems were not executed by this review.

The independent focused command was:

```sh
/workspace/scratch/e4108ca2b049/innovate-scientific/.venv-registry-reconciliation/bin/python \
  -B -m pytest -p no:cacheprovider tests/native/test_joss_environment.py
```

It exited 0 with **15 passed in 0.10 seconds** on Python 3.15.2. These are guard
tests using synthetic fixtures; the positive setup is established separately
by the actual replay and its retained records.

## Original execution and guarded replay

The [original environment receipt](attempt-3/toolchain/environment-receipt.json)
and the two [original executed scripts](attempt-3/toolchain/executed-scripts/)
remain unchanged. Both script copies match their original recorded hashes.
Independent inspection also verified all 241 original package archive hashes
and sizes, all 24 configured fonts against their recorded hashes and pinned
Git blobs, and all six recorded tool binary hashes.

The subsequent [guarded setup replay](guarded-setup/receipt.json) used the fixed
helpers in fresh owned directories with the verified archive cache. Its three
top-level commands exited 0; the nested format-build and font-cache commands
also exited 0. All five retained log hashes match. The independently checked
replay contains **241 archives, totaling 47,572,856 compressed bytes**, with
every size and SHA-512 matching the manifest. The package manifest, 24 font
hashes, Pandoc identity and four recorded TeX executable hashes equal the
original environment. The new format bytes and environment copy match the
configuration record.

The final PDF was produced using the original environment. The guarded replay
ran afterwards and did not render another PDF. The preserved scripts and
separate receipts maintain that distinction; the PDF is not attributed to a
helper revision that had not yet executed.

## Final draft PDF

[Attempt 4](attempt-4/paper.pdf) has **four pages and 207,798 bytes**, with SHA-256
`14adfb6b79bd16c26fef5f3a704cae487fdc2d883e7c1e010a203b69c932c5b0`.
The three current manuscript inputs, four retained outputs and unchanged
renderer match every corresponding hash in the
[original render receipt](attempt-4/receipt.json). Its actual integrity,
five-entry BibTeX parse, official TeX, draft-marker and PDF stages passed.

This reviewer independently rasterized all four pages with Poppler
`pdftoppm 26.05.0` at 150 dpi and visually inspected each page. All four PNG
hashes match the author's separate [visual record](attempt-4/visual-review.json).
The body, headings, source/draft notice and five references are readable. No
clipped body text, missing glyphs or unresolved citation keys were observed.
The corrected `ter Hoeven` appears in both the in-text citation and reference.
An independent PDF annotation check found all five expected reference DOI
targets.

Two layout observations remain explicit:

- On page one, the licence sidebar crowds/overlaps the draft line-number
  positions around 16–19. The contents remain readable, but this is not a
  zero-overlap or publication-layout pass.
- Only the fifth reference continues onto page four, leaving substantial
  whitespace. No reference content is missing.

The official draft DOI, submitted-date, editor/reviewer and volume/issue/page
placeholders remain visible. They are template defaults and do not establish
submission, publication or editorial events. The renderer's original
`visual_review: not_performed` receipt is preserved; the subsequent visual
inspection is recorded separately. The PDF does not establish author consent,
confirmed affiliations, research use, rights clearance, scientific validity,
release acceptance, PDF/A conformance or tagged accessibility.

## Reviewed identities

| File | SHA-256 |
| --- | --- |
| `build_pdf_environment.py` | `a985be16f46d2a8171daccc3a4ad687d1d31a4a2e05016c6720ece9812ed6231` |
| `configure_pdf_environment.py` | `c0ca120d679c91f4c10fd3ffb3ed012924ee1dde531f911fda1cb985436b3e8c` |
| `reproduce.md` | `32dc311b89399150cbbf6393e186c0d6735e7198d29eeb176b27d9732ba018b3` |
| `tests/native/test_joss_environment.py` | `fa7421cff76891d9d7fd5cf784083fade845c3a22996b8f972830dd914b3feac` |
| `guarded-setup/receipt.json` | `6dd4a1c9baec381e9b7dd053a8bfe0596413478a1a653a7ba78df28825eb2a2e` |
| `attempt-4/receipt.json` | `950353aec45dcab855e53ca0501eff7bfb172b3fc54f7c25a43eaa40b517967b` |
| `attempt-4/visual-review.json` | `a223a0ed5179b816b10cc92093b0ee10510cb00dc4633c00288175b3d94c4665` |

The [structured review](independent-review.json) retains the independent CLI
observations, replay checks, page identities and all reviewed file hashes.
