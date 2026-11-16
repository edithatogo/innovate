# JOSS working-draft claim register

Edition: 16 November 2026; source baseline
`69b674814513e09a683279c679b1584af8bb4214`; development version `0.7.0a1`.
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
| C05 / Software design | Polars tables, copying IPC, LazyFrame collection and current Julia/R transport limitations; source contract | [README](../../../../README.md), [native guide](../../../../docs/astro-site/src/content/docs/native-migration.md), [tabular source](../../../../rust/crates/innovate-tabular/src/lib.rs) | Do not advertise zero-copy or equivalent R/Julia tabular distribution. Recheck when the implementation or distribution changes. |
| C06 / Software design | Reuse of Rust numerical packages and explicit Kairos/Voiage ownership | [core manifest](../../../../rust/crates/innovate-core/Cargo.toml), [product](../../../product.md), [README](../../../../README.md) | A dependency declaration or data boundary is not accepted provider integration. Current Kairos pin must have its own receipts. |
| C07 / Software design | Unsupported operations fail explicitly; validation and scientific conventions are documented | [native guide](../../../../docs/astro-site/src/content/docs/native-migration.md), [release gates](../../../release-gates-20261107.md) | This documents the intended/implemented boundary without claiming universal negative-test or scientific coverage. Relevant accepted tests remain necessary. |
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

## Edition acceptance

For a later edition, inspect changed supporting files, update each affected claim
and preserve this edition's receipt. Do not refresh hashes solely to pass a
check. Check references and metadata, render through the current JOSS-supported
route, and retain the output identity. An automated source review or successful
render is preparation evidence; scientific review, author approval and editorial
acceptance remain separate.
