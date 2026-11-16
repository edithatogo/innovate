# Qualification receipts — 16 November 2026

Current hosted qualification is recorded in
[hosted-b2ed945](hosted-b2ed945/README.md): the full documentation workflow and
Windows/macOS/Linux native CI pass. Fresh local qualification is grouped under
[integrated-5821d8a](integrated-5821d8a/integrated-qualification.json), for the frozen
working tree based on `5821d8a38fbc6cbf3822a63474b79c07b18b5bc0`.
[qualification-summary.json](qualification-summary.json) indexes current and
historical evidence separately. Older main69/code-snapshot and hosted5ee/178
receipts retain their actual revisions and outcomes; they are not current-head
runtime or browser passes. Use source-file hashes, not the `HEAD` field alone,
to identify tested working-tree inputs.

The [implementation record](../implementation-20261116.md) explains decisions,
completed work, limitations and remaining DOC-01–08 acceptance.

## Final integrated records

| Record | Evidence and limits |
| --- | --- |
| [integrated-qualification.json](integrated-5821d8a/integrated-qualification.json) | Base/source/artifact identities and current scoped results; overall acceptance remains incomplete |
| [copy-manifest.json](integrated-5821d8a/copy-manifest.json) | 60 raw files, 407,966 bytes, verified against originals before selected archive envelopes |
| [final source summary](integrated-5821d8a/final-source/qualification-summary.json) | 286 source tests and all source gate outcomes; source-only strict content remains incomplete by design |
| [final runtime qualification](integrated-5821d8a/final-runtime/qualification.json) | Fresh final native installation; 11 examples, 19 public doctests, 50 focused tests, ten documented callables and optional notebook execution |
| [final content execution](integrated-5821d8a/final-runtime/execution.json) | Python passes; absent current Julia/R receipts keep the all-language aggregate at exit 1 and `complete: false` |
| [Rust completion](integrated-5821d8a/rust/rust-documentation-completion-20261116.json) and [frozen strict-doc log](integrated-5821d8a/rust/innovate-workspace-rust-docs-frozen-20261116.log) | Five root-workspace crates pass strict documentation checks; games is excluded |
| [fresh Rust doctests](integrated-5821d8a/final-runtime/rust-doctests.log) | Four tests on final frozen source; supersedes the earlier doctest timing retained in the Rust completion record |
| [Rust preservation evidence](integrated-5821d8a/rust/innovate-rust-doc-token-evidence-20261116.json) | 17 files; 16 exact stripped-doc matches plus three authorised formatting commas in likelihood; negative controls retained |
| [standalone games result](integrated-5821d8a/games/games-documentation-20261116.json) | Dependency resolution fails before rustdoc; doctests not run; no pins or locks changed |
| [Node tests](integrated-5821d8a/frontend/innovate-integrated-node-tests.log), [Astro build](integrated-5821d8a/frontend/innovate-integrated-astro-build.log) and [type check](integrated-5821d8a/frontend/innovate-integrated-astro-check.log) | 55 Node tests, 211 HTML pages and zero type errors/warnings/hints |
| [browser discovery](integrated-5821d8a/frontend/innovate-integrated-browser-list.log) | All 164 cases selected; discovery is not execution qualification |
| [final artifact](integrated-5821d8a/final-source/artifact.json) | 11,361 local links, zero errors, exact source/artifact fingerprints; independent build attestation remains false |
| [native notebook receipt](integrated-5821d8a/final-runtime/notebooks/qualification.json) | Two headless cutoff states, exact parity and successful export; retained HTML is evidence only, not site publication |

The test counts overlap across invocations and are not a unique combined total.
The fresh notebook HTML remains under the evidence directory and is not promoted
to the Astro site. Interactive notebook controls, WASM, human reader evaluation, broader scientific
acceptance and live publication remain unqualified. The later hosted-b2ed945
receipts qualify the site browser repair and actual Windows execution.

Raw copies were verified byte-for-byte before selected checksum-only scanner
matches received reversible [archive envelopes](archive-format/README.md).
The copy manifest and cross-record hashes identify the original decoded bytes.
Stored archive-file hashes are separate identities in the root evidence index.
Original absolute execution paths remain provenance; the copy manifest maps them
to their retained paths. Decode an envelope before using its original receipt
schema; never feed the envelope itself to a production verifier.

## Historical hosted follow-up and repair

[hosted-5ee03ddc.json](hosted-5ee03ddc.json) records the first hosted implementation
run: all maintained language examples passed, while both aggregate jobs correctly
failed a stale content-policy baseline binding. Browser steps were not reached.
This later hosted evidence does not rewrite the local unavailable-language record.

[ratchet-repair.json](ratchet-repair.json) records the exact failing and passing
source-only commands after explicit review of the zero-debt inventory. The
baseline still accepts no violations. Its [65 regression results](ratchet-regressions.log)
pass. [artifact-after-baseline.json](artifact-after-baseline.json) records identical
HTML/artifact hashes with only the baseline's source fingerprint changed. It is
separate from the earlier artifact receipt.

[hosted-178779bd.json](hosted-178779bd.json) records the next exact-head run. The
native aggregate now passes with fresh installed Python/Julia/R receipts and no
stale baseline. Its four sibling `hosted-178779bd-*` receipts are the original
downloaded execution records. The complete managed Chromium run has 163 passes
and one failure across 164 cases. A nonempty homepage prefetch exposes an empty
renderer body to the comparator. Its failure, full case-outcome inventory,
qualification-report hash, trace hash and downloadable artifact IDs are retained;
browser acceptance and deployment remain false.

[portability-repair.json](portability-repair.json) records the narrow Windows
newline fixes and source identities. The actual Git checkout regression fails
without the frozen-data attributes and passes with them. Its
[before log](checkout-regression-before.log) and
[44-case passing log](portability-tests.log) preserve the result. The generated
catalogue now writes explicit UTF-8 bytes; frozen dataset bytes and exact hashes
are unchanged. This local regression is not an actual Windows runtime receipt.

## Historical main69 records

| Record | Evidence and limits |
| --- | --- |
| [historical-summary-main69.json](historical-summary-main69.json) | Preserved original aggregate and its historical input identities |
| [69b6748-summary.json](69b6748-summary.json) | Historical main69 Python/Rust qualification; Julia and R explicitly not run locally |
| [69b6748-python-execution.json](69b6748-python-execution.json) | All 11 ordinary examples, exact source/stdout and observed assertion lines; public Python doctests |
| [69b6748-native-documentation.json](69b6748-native-documentation.json) | Installed callable inventory, source equality and native binary identity; independent source attestation remains false |
| [69b6748-cargo-doc.log](69b6748-cargo-doc.log) and [69b6748-cargo-doctest.log](69b6748-cargo-doctest.log) | Strict Rust documentation compilation and three executed doctests |
| [polyglot-source.json](polyglot-source.json) | Julia/R export, documentation and example structure; no runtime qualification |
| [notebook-qualification.json](notebook-qualification.json) | Exact native environment, two cutoff states, example parity and successful export hash |
| [notebook-structure.json](notebook-structure.json) | Separate source-only notebook admission result |
| [source-tests.json](source-tests.json) | 272 tests across 11 source suites, with sibling logs; not a sum of unique tests across every invocation |
| [astro-build.log](astro-build.log) and [astro-check.log](astro-check.log) | Historical 211-page build and type checking |
| [artifact.json](artifact.json) | Contemporaneous artifact/source fingerprints, 11,360 local links and zero errors; hashes alone are not build attestation |
| [content-structure.json](content-structure.json) | Current source inventory and zero structural debt; execution flags intentionally remain false here |
| [retirement.json](retirement.json) | Actual Git history plus all 141 rendered notices checked against 263 removed sources |
| [scientific-review.json](scientific-review.json) | 61 source hashes and internal review-record consistency; no human or exhaustive scientific certification |
| [browser-environment-blocker.json](browser-environment-blocker.json) | Browser attempts, exact counts/unknowns, hashes and the environment restriction |
| [browser-first-launch-error.md](browser-first-launch-error.md) and [browser-launch-failure.log](browser-launch-failure.log) | Actual Chromium socket failure before browser connection; 64 failed setup results, not 64 page quality failures |
| [managed-browser-install.log](managed-browser-install.log) | Pinned Playwright browser download failed with an invalid ZIP response |
| [browser-launch-failure-artifacts.json](browser-launch-failure-artifacts.json) | Hash inventory for 64 original traces and error contexts; the repeated binary traces are local diagnostic outputs, not committed source |
| [registry-hygiene.json](registry-hygiene.json) | Three inherited completed-track/archive inconsistencies, no missing links; shared registry reconciliation remains integrator-owned |

The historical local browser diagnostic retains original `.reports` locations as provenance.
The bounded copies in this directory have the filenames linked above. It
explicitly distinguishes the first zero-case preview setup receipt from the
interrupted second run. There is **no finalized second-run browser qualification
receipt and no successful browser connection**. One saved error context, the full
text failure log and the trace hash inventory retain the substantive diagnosis.

## Reproduction commands

Use the supported toolchains and installation instructions in CONTRIBUTING.md.
Commands below are entry points for reproducing the checks, not claims that
unexecuted language/browser gates passed. From the repository root:

```sh
python scripts/check_api_documentation.py --mode strict
python scripts/check_native_documentation.py
python scripts/check_polyglot_api_documentation.py --check
python scripts/check_documentation_content.py --mode strict
python scripts/generate_learning_catalogue.py --check
python scripts/check_community_documentation.py
python scripts/check_learning_notebooks.py --json
python scripts/check_retired_documentation.py --history --dist docs/astro-site/dist --json
python scripts/check_scientific_review_contract.py
python scripts/verify_production_docs.py --json
python scripts/check_repository_policy.py
python scripts/conductor_registry_hygiene.py --json
```

The earlier registry receipt records three unrelated archive inconsistencies.
Upstream #656 resolved them; [registry-integrated.json](registry-integrated.json)
records the successor's zero-finding pass. The content source-only
command does not execute the examples. The existing execution gate is:

```sh
python scripts/check_documentation_content.py --mode strict --execute --python .venv/bin/python --language-receipt-dir docs/astro-site/.reports
```

That full command also requires current Julia/R receipts; a Python-only result
cannot satisfy its complete language scope. The saved Python receipt records
the existing runner's actual per-example execution and public doctests separately.

Optional notebook execution uses its actual locked environment:

```sh
uv sync --project notebooks --locked
uv run --project notebooks --locked python scripts/check_learning_notebooks.py --execute --json
uv run --project notebooks --locked marimo edit notebooks/public_data_fit.py
```

Rust commands used:

```sh
RUSTDOCFLAGS='-D missing_docs -D rustdoc::broken_intra_doc_links -D rustdoc::invalid_codeblock_attributes' cargo doc --workspace --no-deps --locked
cargo test --workspace --doc --locked
```

In `docs/astro-site`, configure `INNOVATE_DOCS_PYTHON` to the absolute supported
Python executable as the existing workflow does, then run:

```sh
pnpm install --frozen-lockfile
pnpm test
pnpm check
pnpm build
pnpm exec playwright install chromium
pnpm test:browser
```

The native regression selection is `tests/native/test_public_data_tutorial.py`,
`tests/native/test_learning_catalogue.py` and
`tests/native/test_documentation_conventions.py` (43 initial cases; 44 after the
checkout regression; 50 in the final integrated selection after six workspace-scope
cases). Notebook structural
regressions are `tests/test_learning_notebooks.py` (18 unittest cases). The pnpm
entry point also runs five explicit Python suites with `-I -S` and rejects empty
discovery. Repeated invocations do not represent additional unique coverage.

## Review and evidence limits

Independent source review verified unchanged executable binding/FFI bodies and
unchanged shared runtime manifests/locks against main69. Scoped scientific/reader
review resolved time-unit, vintage and repeated-evaluation interpretation issues.
The actual review scope and primary references are maintained in
[the scientific review contract](../../../../docs/contracts/scientific-pedagogical-review.json).

Python runtime execution, Rust compilation, source correctness, artifact checks,
browser acceptance, scientific validity, human reader outcomes and publication
remain separate. Fresh integrated records qualify the local source, Python/Rust
and static-artifact scopes. Historical hosted evidence adds installed Julia/R
and aggregate execution success only at its earlier head, while preserving the
one full-suite browser failure. Current hosted-language/browser/Windows gates,
human evaluation, broader scientific acceptance and publication remain open.

## Publication and merge follow-up — pending hosted result

The current draft implementation is [PR #671](https://github.com/edithatogo/innovate/pull/671);
PR #659 is closed. The
[integration follow-up](integration-followup/integration-followup.json) records the
ordinary merge `b2ed945f9ffede3bdd00c29437636c0bf1eba03d` with upstream `b29a1d`,
its five preserved upstream paths, and the 16 audited Rust source blobs retained
exactly. All 16 have exact non-documentation byte equality with upstream.

| Retained record | Evidence and scope |
| --- | --- |
| [Integration receipt](integration-followup/integration-followup.json) | Merge identities, source comparisons, compact unchanged artifact fingerprints and raw-copy manifest |
| [Candidate scan verification](integration-followup/candidate-scan/verification.json) | Both scanner versions return zero findings for ordinary candidate tree `dc4a4ec`; this is not a scan of the later merge |
| [New upstream tests](integration-followup/upstream-b29-targeted-tests.log) | Nine tests pass across the three newly merged core targets; not the full workspace suite |
| [Scientific source check](integration-followup/merge-scientific.json) | 66 declared source files remain fresh; scientific quality is not certified |
| [Notebook source check](integration-followup/merge-notebook-source.json) | Source contract passes without notebook execution or export publication |
| [Hosted release hold](integration-followup/hosted-b2ed945-release.log) | Existing 16-model and 22-capability scientific migration/acceptance hold, exit 1 |

The fresh artifact check retains source `cf3ea071…` and artifact `adfc2087…`,
211 pages and 11,361 local links with zero errors. Exact digests and the original
fresh `/tmp` producer path/size/SHA256 are in the compact integration receipt;
the full fresh artifact JSON is not copied. The earlier full producer receipt
remains [retained](integrated-5821d8a/final-source/artifact.json) in its verified
reversible envelope. These are distinct records with equal declared source and
artifact fingerprints, not interchangeable producer files.

All 13 copied files (86,698 raw bytes) were verified before any subsequent reviewed
archival envelope. Their raw hashes refer to decoded originals, with stored
archive identities indexed separately. Hosted documentation run
[43866570433](https://github.com/edithatogo/innovate/actions/runs/43866570433) is
pending in this pre-result snapshot; a later exact-head receipt must record the
outcome separately. Standalone games, release acceptance, offline runner and
remaining programme/publication holds remain in place.
