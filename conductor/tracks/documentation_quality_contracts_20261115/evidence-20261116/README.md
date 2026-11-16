# Qualification receipts — 16 November 2026

These records qualify the delivered documentation increment, with native work
refreshed onto main `69b674814513e09a683279c679b1584af8bb4214` and code snapshot
`7bd5aa89b0798abaab4132df8096bbc683cbb553`. Some receipts cover the subsequently
reviewed working tree: use their source-file hashes, not the `HEAD` field alone,
to identify inputs. No receipt claims independent build attestation, combined
language completion, full scientific acceptance or live deployment.

The [implementation record](../implementation-20261116.md) explains decisions,
completed work, limitations and remaining DOC-01–08 acceptance.

## Results and primary records

| Record | Evidence and limits |
| --- | --- |
| [qualification-summary.json](qualification-summary.json) | Toolchains, observed checks, source identities and SHA-256 of the retained records |
| [69b6748-summary.json](69b6748-summary.json) | Current-base Python/Rust qualification summary; Julia and R explicitly not run |
| [69b6748-python-execution.json](69b6748-python-execution.json) | All 11 ordinary examples, exact source/stdout and observed assertion lines; public Python doctests |
| [69b6748-native-documentation.json](69b6748-native-documentation.json) | Installed callable inventory, source equality and native binary identity; independent source attestation remains false |
| [69b6748-cargo-doc.log](69b6748-cargo-doc.log) and [69b6748-cargo-doctest.log](69b6748-cargo-doctest.log) | Strict Rust documentation compilation and three executed doctests |
| [polyglot-source.json](polyglot-source.json) | Julia/R export, documentation and example structure; no runtime qualification |
| [notebook-qualification.json](notebook-qualification.json) | Exact native environment, two cutoff states, example parity and successful export hash |
| [notebook-structure.json](notebook-structure.json) | Separate source-only notebook admission result |
| [source-tests.json](source-tests.json) | 272 tests across 11 source suites, with sibling logs; not a sum of unique tests across every invocation |
| [astro-build.log](astro-build.log) and [astro-check.log](astro-check.log) | Integrated 211-page build and type checking |
| [artifact.json](artifact.json) | Contemporaneous artifact/source fingerprints, 11,360 local links and zero errors; hashes alone are not build attestation |
| [content-structure.json](content-structure.json) | Current source inventory and zero structural debt; execution flags intentionally remain false here |
| [retirement.json](retirement.json) | Actual Git history plus all 141 rendered notices checked against 263 removed sources |
| [scientific-review.json](scientific-review.json) | 61 source hashes and internal review-record consistency; no human or exhaustive scientific certification |
| [browser-environment-blocker.json](browser-environment-blocker.json) | Browser attempts, exact counts/unknowns, hashes and the environment restriction |
| [browser-first-launch-error.md](browser-first-launch-error.md) and [browser-launch-failure.log](browser-launch-failure.log) | Actual Chromium socket failure before browser connection; 64 failed setup results, not 64 page quality failures |
| [managed-browser-install.log](managed-browser-install.log) | Pinned Playwright browser download failed with an invalid ZIP response |
| [browser-launch-failure-artifacts.json](browser-launch-failure-artifacts.json) | Hash inventory for 64 original traces and error contexts; the repeated binary traces are local diagnostic outputs, not committed source |
| [registry-hygiene.json](registry-hygiene.json) | Three inherited completed-track/archive inconsistencies, no missing links; shared registry reconciliation remains integrator-owned |

The browser diagnostic retains original `.reports` locations as provenance.
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
python scripts/check_polyglot_api_documentation.py
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

The last command is expected to remain nonzero until the coordinator resolves
its three recorded, unrelated archive inconsistencies. The content source-only
command does not execute the examples. The existing execution gate is:

```sh
python scripts/check_documentation_content.py --mode strict --execute
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
`tests/native/test_documentation_conventions.py` (43 cases). Notebook structural
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
remain separate. This evidence set records the first four positively, retains the
browser environment failure, and leaves the remaining acceptance explicitly open.
