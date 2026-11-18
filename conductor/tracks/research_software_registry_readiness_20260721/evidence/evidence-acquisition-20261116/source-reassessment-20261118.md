# Source-to-claim reassessment — 18 November 2026

**Scope:** the unapproved JOSS working draft of Innovate 0.7.0a1; candidate
source identity [`20fe0b41b815bc926fc3d61b9672c3d4b2feb289`](https://github.com/edithatogo/innovate/commit/20fe0b41b815bc926fc3d61b9672c3d4b2feb289).
This deliberately reviews changes from the prior `b29a1d27f54fdd08d77881ee999772fceab7c03a` edition;
it is not an automatic hash refresh, research-use record, author attestation,
security waiver, scientific release or journal decision.

At PR #662 head `884f9ddc124bf2f43ce63b94ce99f0aef4d05e4b`, all 14 tracked source Git blobs were checked
against the selected main source. Their bytes match. The CI failure
[44263647474](https://github.com/edithatogo/innovate/actions/runs/44263647474)
is confirmed RED: stale `pyproject.toml` blocked 26 downstream JOSS tests
before their intended assertions; a changed `Cargo.toml` also failed in
separate negative cases. Fail-closed source integrity and all negative tests
are unchanged.

## Fourteen assessed inputs

Seven unchanged since the previously assessed source: `README.md`,
`CITATION.cff`, `LICENSE`, `migration-ledger.json`,
`conductor/product.md`, `conductor/release-gates-20261107.md`,
and `rust/crates/innovate-tabular/Cargo.toml`. Their source hashes
are retained unchanged.

The seven changed inputs are:

| Input | Change verified | Manuscript / claim disposition |
| --- | --- | --- |
| `pyproject.toml` | Added `THIRD-PARTY-LICENSES` and vendored source to sdist scope | Source packaging does not certify licence clearance or publication readiness |
| `Cargo.toml` | Added 16 explicit vendor root exclusions; five member crates unchanged | Root-workspace documentation coverage is not coverage of excluded crates |
| `Cargo.lock` | Updated resolved dependency graph with vendored/path changes | Lock resolution is not algorithm, security or scientific equivalence |
| `docs/astro-site/src/content/docs/native-migration.md` | Revised change-point description: PELT, 100,000 maximum observations, short-grid oracle, 5,000-point regression, 1–2000 jump | C07 remains bounded; neither universal tie/floating-point equivalence nor performance is claimed |
| `rust/crates/innovate-core/Cargo.toml` | Vendored nalgebra, levenberg-marquardt, ode_solvers, nuts-rs and faer; `publish=false` | C06 now distinguishes source reuse, upstream validation and registry publishing |
| `rust/crates/innovate-tabular/src/lib.rs` | More precise Rustdoc for panel, joint-draw and cohort IPC; decoder does not repeat all encoder finite/label checks | C05 and C07 now explicitly reject complete untrusted IPC validation and automatic unit/provenance semantics |
| `.github/workflows/ci.yml` | Source/capability evidence gates and third-party licence Docker smoke check; workspace cargo formatting | CI declarations alone cannot stand in for receiving-head passes |

The new manuscript clarifies the packaging hold and IPC decoder limitations.
C01–C04/C08 retain previous limits. All P01–P06 remain withheld; the
16-model/22-capability scientific release gate is unqualified.
The five existing bibliography entries remain unchanged.

**Rendering:** the source-bound official Inara PDF and four-page visual
inspection dated 16 November are preserved as historical evidence.
A new official-template render on the new paper/claims and a new all-page
inspection, with actual toolchain and PDF hashes, are required. None has
been fabricated or copied forward. Reader trials, external registry/curator
outcomes, a resolving current-release Software Heritage SWHID, author
declarations and JOSS submission remain pending.
