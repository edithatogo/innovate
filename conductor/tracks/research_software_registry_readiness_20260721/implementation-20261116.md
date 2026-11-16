# Executed scientific and journal preparation — 16 November 2026

**Preserved first execution packet.** Main advanced through #656 after this
packet was prepared. The [integrated-source reassessment](integration-20261116.md)
records the new source, reviewed draft edition and affected reruns; its later
scope takes precedence over the main/PR status and current-result wording below.

This follow-up implements the ready parts of the existing tracks after the
user requested further execution. It extends draft [#662](https://github.com/edithatogo/innovate/pull/662)
from published head `3f7e39249de496dc28a59e9e94a3c01390d1cb4b`, whose tree
equals local receipt `0c718cea3eb982717ce940560868004d71261e9b`:
`852b4687230505c1ee03cdbd657db0d2f9a24116`.

The inspected main advance to `18c467e18b71d972060365c68865aa405a69949c`
changes only the Renovate workflow. Scientific source, both dependency locks,
and the manuscript's 14 assessed inputs remain at the previously reviewed
bytes. The main/PR identities and final hosted results must be rechecked at
publication. Shared integration and promotion remain with #624; #656 and #659
retain empirical and documentation ownership.

The final ownership readback found #656 at
`ca9739d59d7467180ddce2eb5c411021ddb706f9` (226 files) and #659 at
`178779bd21d7fdde2fe4f6b8179444b72a4bc6d1` (393 files), with no overlap
against this packet's two tracks or five new script/test paths. #656 does change
four manuscript source inputs: `Cargo.toml`, `Cargo.lock`, the core crate
manifest and tabular source. If it integrates first, its changed claims and
native artifacts require reassessment and fresh affected qualification; the
new draft-integrity test must not be cleared by blind rehashing.

## Implemented preparation

### JOSS-02a: executable draft integrity

[`scripts/check_joss_readiness.py`](../../../scripts/check_joss_readiness.py)
validates the complete 14-source/three-edition fingerprint sets, project version,
revision/version tokens in the paper and claim register, visible unapproved
notice, required sections and prose budget. It detects stale or missing inputs,
duplicate JSON fields, path escape, misleading hidden Markdown and invalid
version/record shapes. Windows Git text normalization is explicit in its report.

The current draft passes with **1,074 prose words** and all six required sections.
The checker does not refresh hashes or infer research use, Git authenticity,
scientific acceptance, author consent or journal readiness. The separate
`--require-submission-ready` mode continues to fail for this unapproved edition.
Independent review found and reproduced hidden-comment, malformed-project and
version-prefix defects; regression tests now cover their corrected behaviour.

The existing native test collection includes
[`test_joss_readiness.py`](../../../tests/native/test_joss_readiness.py), so a
change to any of the 14 supporting files requires a reviewed manuscript/claim
reassessment before the real-record integrity test can pass. This is deliberate
drift detection. Coordinate upcoming #656/#659 changes through #624 instead of
refreshing fingerprints solely to obtain a pass.

```sh
python scripts/check_joss_readiness.py --json
python scripts/check_joss_readiness.py --require-submission-ready --json
python -m pytest tests/native/test_joss_readiness.py -q
```

### JOSS-02b: actual bibliographic processing and draft rendering

The current five-reference bibliography has been read by Pandoc's real BibTeX
parser. Pandoc 3.6.4 and official Inara revision
`88ee5d656f147d801448ce77ad8375619d383df6` generated draft TeX and resolved
all five citations. The [render wrapper](joss/render.py) enforces the offline
integrity prerequisite, pinned tool versions, fresh output directory and explicit
draft flags. Eighteen synthetic guard tests exercise its failure behaviour.

The [final actual rendering receipt](evidence/joss-render-20261116/attempt-2/receipt.json)
matches the final wrapper and records passed integrity, bibliography, TeX and
draft-marker checks. Its [official TeX](evidence/joss-render-20261116/attempt-2/paper.tex)
and [parsed bibliography](evidence/joss-render-20261116/attempt-2/bibliography.json)
are retained. PDF compilation stopped at **`luacode.sty` not found**: upstream
Make exited 2 and the wrapper exited 1. No PDF exists and no visual inspection
passed. The complete [build log](evidence/joss-render-20261116/attempt-2/render.log)
retains that failure. The earlier attempt is preserved and explicitly superseded;
successful TeX generation does not establish PDF or submission readiness.

### SWH-01: archival candidate identity

The [archival target inventory](evidence/archive-target-inventory-20261116.json)
resolves `v0.5.0` to its lightweight-tag commit and `v0.6.0` to its annotated tag
and target commit. It distinguishes published GitHub releases from tag-only
records and records that no `0.7` tag was returned. It selects no release and
makes no rights, ingestion or scientific acceptance claim. The
[registry dossier](registry-evidence.md#executed-target-inventory--16-november-2026-follow-up)
sets out those distinct next steps.

### ALIGN-02: installed optional Arrow consumer

The [scientific execution packet](../scientific_ecosystem_alignment_20261029/evidence/arrow-installed-20261116/README.md)
implements an installed-artifact qualifier and records a fresh Linux development
wheel. All **42 named cases pass with zero skips**: 38 scenario-export cases,
including all four optional PyArrow cases, and four native Kairos/paused-producer
cases. The command checks the wheel RECORD and imported package bytes, declared
runtime versions, exact test selection and source/build/test identities.

This closes the skipped-optional-case gap for that identified artifact and
platform. Recurring hosted execution remains an integrator-owned CI change;
other platform, scientific, release and publication requirements remain separate.
The separate [minimum-compiler receipt](../scientific_ecosystem_alignment_20261029/evidence/arrow-installed-20261116/msrv-check.json)
records actual Rust 1.99.1 passing
`cargo check --locked --workspace --all-targets -j 1` for all four workspace
crates. That establishes compile checking for Linux/default features, not tests,
artifact/linking or other-platform acceptance. The focused native-export tests
also passed eight cases; their exact command and receipt are in the scientific
packet. None of these checks completes the 38 scientific obligations.

The existing provider/network targets also passed **64 tests** on Rust 1.100.0:
17 Kairos, 37 network, five network-diffusion and five petgraph cases, with no
failed, ignored or filtered cases. The [provider receipt](../scientific_ecosystem_alignment_20261029/evidence/arrow-installed-20261116/provider-rust.json)
binds them to the same 122 native input hashes and Kairos revision. A fresh Git
provider retrieval and compilation were observed; crates.io publication and
broader provider/scientific acceptance remain separate.

## Remaining gates

Actual version-bound research use and the complete public-development history
assessment remain JOSS-01 work. Provisional author/affiliation metadata, funding,
conflicts, rights and the complete AI-use/human-validation account require the
authors' own factual confirmation. The local preparation packet cannot provide
those facts.

RRID duplicate checking remains incomplete after the recorded HTTP 403. Its
resource description is prepared, but neither submission nor curator acceptance
exists in this packet. Software Heritage still needs a selected target, rights
assessment, request and successful visit/snapshot/object linkage. GitHub Projects
membership is not exposed by the available connector; no project-visible claim
is made.

The 38 scientific obligations, four historical manual scenario checks, provider
publishability, scheduled consumers, offline-security qualification and final
release decision retain their existing owners and evidence requirements. No
external registration, archive deposit, journal correspondence, submission,
release or merge is performed by these preparation steps.

## Verification and publication

ChatGPT with GPT-6 Astra Pro also assisted this follow-up's qualification/preflight
implementation, regression tests, source review and evidence preparation. This
records the current increment's assistance for the eventual full AI-use inventory;
it supplies no human author-review or design-decision attestation. The retained
manuscript edition still describes its own earlier drafting scope explicitly.

The [preparation verification record](evidence/preparation-checks-20261116.json)
binds the final six new tool/test inputs to their executed local checks.

| Check | Actual outcome and scope |
| --- | --- |
| Combined registry and new tooling suites, Python 3.15.2 / pytest 9.2.0 | **169 passed, zero skips**: 56 registry, 50 JOSS integrity, 18 renderer and 45 Arrow-qualifier guards. Synthetic controls are not scientific or rendering acceptance. |
| JOSS integrity checker coverage | **135/140 statements (96.43%)**; **37/42 branches (88.10%)**. |
| Renderer-wrapper coverage | **119/126 statements (94.44%)**; **33/40 branches (82.50%)**. |
| Arrow-qualifier coverage | **196/215 statements (91.16%)**; **84/100 branches (84%)**, combining guards and the actual isolated worker. Its negative malformed-record run correctly fails. See the scientific packet. |
| New-code Ruff lint and format | Passed for all six new tool/test files; no lint threshold or source input was changed to clear a check. |
| Pending registry record / explicit completion mode | Record valid, all three destinations pending; `--require-complete` exits 1. |
| Draft integrity / explicit submission mode | Integrity valid, 1,074 words; `--require-submission-ready` exits 1. |
| Local repository policy and version mapping | Passed; live GitHub settings are not checked by that command. |
| Migration and release checks | Development check passes; release check exits 1 with the same **38 obligations**. |
| Conductor registry hygiene | Same three pre-existing archive/status discrepancies (ODE, NUTS, CI supply-chain); shared correction remains with #624/#656. |
| Source identity | All 14 manuscript source hashes match recorded Git blobs; all 122 native inputs match the build source. The manuscript, bibliography, claim register and scientific source are unchanged. |
| Independent review | Material guard/identity findings were fixed and independently rechecked. Portable fixture corrections preserve the production restrictions. No blocking residual within this scope; no human/scientific approval inferred. |

The original scientific plan from the 14 November continuation to its end
remains byte-identical. Source implementation, allowed-file boundaries, links,
JSON records and whitespace were checked before publication. The exact published
head/tree, issue readback and current hosted results are recorded in #662's PR
receipt and #624 handoff; earlier PR-head passes do not qualify later source.
