# JOSS source reassessment after the Rust documentation repair

Assessed on 16 November 2026. This record moves the working draft's source
baseline from `57d2916586d107c8affe90355e453abe27f01c26` to
[`b29a1d27f54fdd08d77881ee999772fceab7c03a`](https://github.com/edithatogo/innovate/commit/b29a1d27f54fdd08d77881ee999772fceab7c03a),
which contains the Rust documentation/test repair in
[#668](https://github.com/edithatogo/innovate/pull/668). It does not accept a
scientific result, provider integration, release or submission.

The inspected integration commit is
`2748084292e5e57c181c1665a686ad3a291145bf`, tree
`787eb4cd175b7343026cdad88143e8879e0b3062`, combining PR #662 head
`d36c33f6b4fb480828ddb9aae0121f3379667015` and that main revision.
Every one of the fourteen manuscript source inputs equals its Git blob at the
assessed main revision. The [machine-readable record](source-reassessment.json)
contains the comparisons, command results and fingerprints. The
[previous snapshot](source-reassessment.previous-snapshot.json) preserves the
preceding assessment byte for byte.

## Change and claim decision

Only `rust/crates/innovate-tabular/src/lib.rs` changed among the fourteen tracked
inputs. Its [retained diff](source-reassessment.diff) contains 36 added Rustdoc
field comments, with no deleted lines. Removing Rustdoc lines from both Git
blobs leaves identical bytes. The receipt removes each whole Rustdoc line while
preserving the remaining bytes and original line endings (`splitlines(keepends=True)`
followed by an empty-byte join); that result hashes to
`2853ec9a0b7f9bfc49065f6649b02e642ba39512b66302b13acb17200ec69639`.
The documented fields belong to column bindings,
panel rows, joint draws and cohort occupancy; the executable implementation is
unchanged by this diff.

| Claim | Assessed implication |
| --- | --- |
| C05: table and binding boundaries | The copying IPC, LazyFrame collection and outstanding Julia/R adapters and bundles remain as described. Field documentation does not complete those capabilities. |
| C07: validation and conventions | The existing statement about documented model time grids remains appropriately bounded. Added panel-row documentation does not make all tables obey a universal ordering contract. |
| C01–C04, C06 and C08 | No relevant claim needs a substantive wording change. Their existing evidence classes and limitations remain in force. |
| P01–P06 | Research impact, complete scientific/release qualification, superiority, accepted development-history assessment, author declarations and external outcomes remain withheld. |

The manuscript's only text change is its visible assessed commit identifier.
The source snapshot deliberately changes the one reviewed source fingerprint
and the affected edition fingerprints after this assessment. The bibliography,
all 38 migration obligations, model-family count and pending submission state
are unchanged. The scientific publication owner retains empirical validation
and the documentation owner retains native/site work.

The broader main advance also contains Rust field/variant documentation, enum
layout changes, native tests, Renovate updates and a Paignia pin update to
`5103675004d80afc845bbdd067e2fea4e75f6425` in the excluded games crate. Those
changes were inspected to avoid overlooking a relevant architecture change.
The manuscript does not claim accepted Paignia integration or derive scientific
acceptance from the newly added tests. Native execution and hosted integration
evidence have their own owners and receipts.

## Executed identity checks

```sh
git diff 57d2916586d107c8affe90355e453abe27f01c26 b29a1d27f54fdd08d77881ee999772fceab7c03a -- rust/crates/innovate-tabular/src/lib.rs
python scripts/check_joss_readiness.py --json
```

The actual checker command used the recorded Python 3.15.2 executable. Before
the deliberate reassessment, it failed on the changed tabular source digest.
After the update it passed all fourteen source and three edition fingerprints,
six required sections and **1,076 prose words**. The separate Git-blob comparison
above establishes the selected source identity; the checker itself does not
verify Git revision authenticity. Neither check supplies author approval or
scientific, research-use, release or editorial acceptance.

Earlier source assessments and failed render receipts remain unchanged. The
new render packet must identify this edition and preserve its own actual
toolchain, outputs and visual-inspection result.

## Bibliography display correction found during PDF inspection

The first successful PDF exposed a name-particle issue in `mesa2025`: Pandoc
interpreted `ter Hoeven, Ewout` as a dropping particle and printed `Hoeven et al.`.
The [publisher's citation](https://joss.theoj.org/papers/10.21105/joss.07668),
checked again on 16 November 2026, explicitly retains `ter Hoeven et al.`.
Protecting the family name with BibTeX braces makes Pandoc classify `ter` as a
non-dropping particle; a real parser/citeproc check then produces the publisher's
surname form. Only those braces and the corresponding edition fingerprint change.
No prose, source baseline, DOI, reference identity or scientific claim changes.

The [pre-correction inputs](render/attempt-3/inputs/), PDF, parsed bibliography
and [visual finding](render/attempt-3/visual-review.json) remain intact. The
machine-readable reassessment records both bibliography hashes and the new
integrity result. The corrected edition subsequently completed a
[fresh render](render/attempt-4/receipt.json) and a
[separate four-page inspection](render/attempt-4/visual-review.json), including
the official template's recorded layout and metadata limitations. The earlier
visual inspection was not transferred to the changed bibliography.
