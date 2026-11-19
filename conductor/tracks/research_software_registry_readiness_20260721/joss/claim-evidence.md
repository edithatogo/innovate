# JOSS working-draft claim register

Edition dated 20 November 2026 for development version `0.7.0a1` at
`c1399484aa7efa224304b0a84c559ecffc10726e`.
Applies to [paper.md](paper.md) and [paper.bib](paper.bib), not the legacy
manuscript or an eventual submitted version. Relevant input bytes and this
manuscript edition are identified in [source-snapshot.json](source-snapshot.json).
Hash equality preserves identity, not truth or scientific acceptance.

## Permitted source-supported wording

| ID / section | Claim and evidence class | Supporting source | Limit and revision trigger |
| --- | --- | --- | --- |
| C01 / Summary | Rust computation with Python, Julia and R interfaces; source architecture | [README](../../../../README.md), [product](../../../product.md), [core manifest](../../../../rust/crates/innovate-core/Cargo.toml) | Does not establish equivalent packaging, complete interfaces, correctness or supported platforms. Reassess when source/manifest changes. |
| C02 / Summary | Sixteen bounded native model families and 38 retained obligations; source inventory | [README](../../../../README.md), [migration ledger](../../../../migration-ledger.json) | No complete-family, stable-release or empirical-validity claim. Update counts only from the new ledger and accepted inventory. |
| C03 / Statement of need | Bass first-purchase diffusion and Fisher–Pry substitution as methodological context | Verified primary entries `bass1969` and `fisher1971` in [paper.bib](paper.bib) | These sources do not validate Innovate's implementation or applications. Do not infer adopters from service-use or stock measures. |
| C04 / State of the field | Mesa, NDlib and PyMC address simulation, network diffusion and Bayesian modelling | Verified primary entries `mesa2025`, `ndlib2018`, `pymc2023`; [reference checks](README.md#reference-verification) | Contextual comparison only. No current-version coverage, speed, accuracy or superiority ranking without an equivalent-workload experiment. |
| C05 / Software design | Polars tables, copying IPC, LazyFrame collection, IPC decoder validation limits and current Julia/R transport limitations; source contract | [README](../../../../README.md), [native guide](../../../../docs/astro-site/src/content/docs/native-migration.md), [tabular source](../../../../rust/crates/innovate-tabular/src/lib.rs) | Do not advertise zero-copy or equivalent R/Julia tabular distribution. Recheck when the implementation or distribution changes. |
| C06 / Software design | Reuse of Rust numerical packages, including vendored paths and the core publication hold, with explicit Kairos/Voiage ownership | [core manifest](../../../../rust/crates/innovate-core/Cargo.toml), [product](../../../product.md), [README](../../../../README.md) | A dependency declaration or data boundary is not accepted provider integration. Current Kairos pin must have its own receipts. |
| C07 / Software design | Unsupported operations fail explicitly; model-specific validation is not universal untrusted IPC validation | [native guide](../../../../docs/astro-site/src/content/docs/native-migration.md), [release gates](../../../release-gates-20261107.md) | The time-ordering example refers to documented model time grids, not every table or IPC row. This documents the intended/implemented boundary without claiming universal negative-test or scientific coverage. Relevant accepted tests remain necessary. |
| C08 / AI disclosure | ChatGPT with GPT-6 Astra Pro assisted this draft's source checks and preparation | This preparation work and [README](README.md) | Limited to this increment. Full software/paper history and human verification remain unknown; only authors can confirm their activities. |

## Claims withheld pending evidence

| ID | Withheld assertion | Required evidence / owner |
| --- | --- | --- |
| P01 | Innovate has demonstrated version-bound research impact | Actual research workflow/publication, source version, inputs and outputs. Maintainer and existing research owners supply the factual linkage; synthetic tests and future plans do not suffice. |
| P02 | Current native software is feature-complete, scientifically qualified or ready for release | Existing #556, migration ledger, Q01–Q05 and integrator acceptance at the selected source/artifact identities. All 38 obligations remain required. |
| P03 | Predictive, causal, uncertainty, scalability or performance superiority | Corresponding existing V01–V04/M01–M05/inference protocols, frozen comparisons, independent references, failure denominators and accepted results. |
| P04 | More than six months of publicly available iterative development has been established for submission | Public availability and development chronology with releases/issues/PRs and dates; a repository creation timestamp alone is insufficient. |
| P05 | Author list, affiliation, funding, conflicts, rights and human AI review are confirmed | Authors' own factual declarations. The current metadata are provisional; this draft cannot attest on their behalf. |
| P06 | SWH archival, RRID approval, JOSS submission/acceptance or archive DOI is complete | Actual provider/editorial records and identifier resolution under [readiness.json](../readiness.json). Local validation and structurally plausible identifiers are insufficient. |

## Adopter classifier dependency reassessment

The integrator compared all fourteen source inputs with
`60c0593988f522495bbc867d91db3949c8086f4c` on 20 November 2026. The core
manifest and lock add a direct edge to the already resolved `num-bigint` 0.4.8
for exact weighted moments. C01 and C06 still describe the Rust computation,
dependency reuse and publication limits. C05's table and interchange inputs
match the prior revision byte for byte. The scientific prose and bibliography
carry forward, and P01–P06 remain withheld. The prior render receipts identify
their respective editions.

## Calendar dependency reassessment: 19 November 2026

The integrator compared all fourteen source inputs with the 18 November
edition. The core manifest now selects `time` for private calendar
operations. The lock removes the core crate's direct Chrono dependency,
adds the `time` closure and still resolves Chrono through published Polars.
C01 and C06 retain the same Rust ownership, vendored dependencies and crate
publication limits. C05's Polars and Arrow interface inputs are byte-identical.
The remaining source inputs, bibliography and scientific prose are unchanged.
P01–P06 remain withheld. Earlier render receipts identify their own editions.

## Source-to-claim reassessment: 18 November 2026

The [full independent input comparison](../evidence/evidence-acquisition-20261116/source-reassessment-20261118.md)
reviews the seven changed source inputs since `b29a1d27f54fdd08d77881ee999772fceab7c03a` and the seven
unchanged inputs, selecting main `20fe0b41b815bc926fc3d61b9672c3d4b2feb289`. Both the assessed
main and the receiving PR head have identical bytes for all fourteen inputs.
The affected claims C05–C07 are now bounded for the vendor/package publication
hold, model-time-grid versus IPC decoder-validation limits, and supplied
unit/provenance semantics. C01–C04 and C08 remain scoped as before, and P01–P06
remain withheld, including research use, scientific release, external
registration, editorial review and human attestations.

The official four-page Inara PDF generated for `b29a1d27f54fdd08d77881ee999772fceab7c03a` is retained
unchanged as **historical**. A newly rendered official-template PDF and a fresh
page-by-page visual inspection are still required for this edition. No historical
PDF hash or prior hosted pass is reattributed to these changed bytes.

## Edition acceptance

The [current source reassessment](../evidence/evidence-acquisition-20261116/source-reassessment.md)
compares the previous `57d2916` assessment with main commit
[`b29a1d27f54fdd08d77881ee999772fceab7c03a`](https://github.com/edithatogo/innovate/commit/b29a1d27f54fdd08d77881ee999772fceab7c03a).
Only one of the fourteen tracked source inputs changed: the tabular crate gained
36 field-documentation comments, with the remaining source text unchanged.
C05's copying IPC, collection and binding/distribution limitations and C07's
model-time-grid qualification therefore remain unchanged. The manuscript's only
text change in this reassessment is its visible assessed commit identifier.

The broader main advance includes Rust documentation and tests, Renovate updates
and a Paignia pin update in the excluded games crate. These changes do not
establish accepted provider integration, empirical validation, research impact
or a qualified release. C01–C08 retain their bounded wording and P01–P06 remain
withheld. All 38 migration obligations remain required. The
[preceding snapshot](../evidence/evidence-acquisition-20261116/source-reassessment.previous-snapshot.json)
and earlier render attempts retain their original identities.

The [preceding source reassessment](../evidence/source-reassessment-20261116.md) records the
comparison with main commit
[`57d2916586d107c8affe90355e453abe27f01c26`](https://github.com/edithatogo/innovate/commit/57d2916586d107c8affe90355e453abe27f01c26),
which merged #656. Four of the fourteen tracked source inputs changed. The
workspace/lock additions and benchmark target preserve C01/C06's limited
architecture and dependency claims. The appended Rust panel, joint-draw and
cohort IPC APIs preserve C05's existing binding/distribution limitations; their
presence does not complete the Julia/R adapters. C07's time-grid sentence was
made more precise because panel ingestion preserves row order.

C02's sixteen-family inventory and all 38 required migration obligations remain
unchanged. P01–P06 remain withheld. Merged source, task checkboxes and registry
entries are not a substitute for accepted empirical, installed-consumer,
release, author or editorial evidence. The
[previous snapshot](../evidence/source-reassessment-20261116.previous-snapshot.json)
and earlier render receipts retain their original source identities; the new
edition is preparation evidence only.

For a later edition, inspect changed supporting files, update each affected claim
and preserve this edition's receipt. Do not refresh hashes solely to pass a
check. Check references and metadata, render through the current JOSS-supported
route, and retain the output identity. An automated source review or successful
render is preparation evidence; scientific review, author approval and editorial
acceptance remain separate.
