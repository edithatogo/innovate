---
title: Documentation quality contracts
description: Measurable API, tutorial, example and rendered-site requirements, with explicit remaining acceptance gaps.
---

Astro/Starlight is the active documentation stack. Quality is assessed against
the current source, executable examples and rendered artifact. Historical
receipts, generated headings and a successful build do not establish completeness.

## Contract layers

| Layer | Enforced evidence | What remains outside that check |
| --- | --- | --- |
| Python source | Every maintained module, class, function, method and property is inventoried; public arguments, returns and direct errors match docstrings | Accuracy of every prose claim and every implicit native failure |
| Rust API | Compiler/rustdoc missing-documentation and broken-link checks are required for full completion | Private PyO3 exports need separate runtime documentation checks |
| Julia and R API | Explicit public exports have attached source documentation, matching signatures and argument contracts; R help files match source defaults | The source checker does not compile the bindings or run their examples |
| Generated reference | Current Python, Julia and R reference pages match the verified source inventory and generated manifest | Rendering signatures and examples does not execute them or prove their scientific claims |
| Content | Every page and example is discovered and explicitly classified; current tutorials have outcomes, prerequisites, steps, expected results, interpretation and limitations | Pedagogical quality and scientific correctness cannot be inferred from headings |
| Execution | Python runs the exact displayed sources with assertions and expected output; Julia/R runners execute every inventoried native example and record observed assertion counts | Receipts establish source and outcome consistency, not authenticated provenance or scientific validity; Python isolation is not a security sandbox |
| Artifact | Rendered local links and fragments, canonical paths, sitemap, search index, semantics and size are inspected from the build output | Browser interaction, assistive-technology testing and live deployment |

The versioned machine-readable contracts live in 
[docs/contracts](https://github.com/edithatogo/innovate/tree/main/docs/contracts).
The Conductor track records implementation and remaining acceptance work.

## Three different results

**Audit** enumerates the full discovered surface and its gaps. A successful audit
command means the inventory was produced; its report can still say incomplete.

**Ratchet** prevents new or changed debt against a reviewed baseline. Entries are
bound to specific source hashes and violations. Fixing an entry requires removing
its stale exception; duplicate and resolved exceptions fail the check. A passing
ratchet never establishes complete documentation.

**Strict completion** rejects unresolved current obligations and cannot substitute
classification for execution. Python documentation completion, current-content
structure, executable outcomes and whole-site completion are reported separately.
Historical pages remain visible in the denominator and are not silently counted
as qualified current material.

The API policy explicitly declares `generated_reference_languages` as Python,
Julia and R. Content reports keep structural coverage and execution separate:

| Report field | Meaning |
| --- | --- |
| `current_structure_complete` | Current pages, source bindings and example declarations have no structural gaps |
| `full_scope_structure_complete` | Structural obligations also cover retained historical material |
| `python_execution_status` | Python examples and source-matched API doctests actually passed, failed, or were not run |
| `current_python_content_complete` | Current structure and Python execution both pass |
| `language_receipts` | Separate validation results for every declared Julia/R runner receipt |
| `current_content_complete` | Current structure and execution across all declared languages pass |
| `full_scope_complete` and `complete` | Execution passes and the whole discovered scope, including historical material, is qualified |

A static audit or ratchet leaves executable completion false and records
`execution_status: not_run`. Supplying a receipt directory alone does not claim
execution; `--execute` must also run the Python checks. Python success remains
visible when Julia or R evidence is absent, but overall execution fails.
These content results complement the separate Rust compiler, native-extension
documentation and rendered-artifact checks required for wider acceptance.

Only visible source bindings and tutorial guidance qualify. MDX and HTML
comments cannot supply an example component, required heading or historical
notice. Literal comment markers inside displayed code remain part of that code.

## Contributor commands

From the repository root, use the Python 3.15.2 tooling environment from the
[site workspace README](https://github.com/edithatogo/innovate/blob/main/docs/astro-site/README.md).
Generate current reference pages before auditing content:

```sh
.venv-docs/bin/python scripts/check_api_documentation.py --mode strict --output docs/astro-site/.reports/api.json
.venv-docs/bin/python scripts/check_polyglot_api_documentation.py --check
.venv-docs/bin/python scripts/generate_api_reference.py
.venv-docs/bin/python scripts/check_documentation_content.py --mode audit --output docs/astro-site/.reports/content-audit.json
.venv-docs/bin/python scripts/check_documentation_content.py --mode ratchet --output docs/astro-site/.reports/content.json
```

Next install the current native Python package, run the native-extension
documentation check and Rust compiler/doctest gates, build the release C ABI,
and stage that same library for Julia and R. Install the Julia project and the
R package with `jsonlite`. The workspace README provides the complete Linux
setup matching CI; Julia uses the existing 1.13 project, and R must provide
`tools::sha256sum`. With those prerequisites ready, run the language runners
before aggregation:

```sh
export DOCS_RECEIPT_DIR="$PWD/docs/astro-site/.reports"
julia --project=bindings/julia scripts/check_julia_documentation.jl
Rscript scripts/check_r_documentation.R
.venv-native/bin/python scripts/check_documentation_content.py \
  --mode ratchet --execute --python .venv-native/bin/python \
  --language-receipt-dir "$DOCS_RECEIPT_DIR" \
  --output "$DOCS_RECEIPT_DIR/content-execution.json"
```

The native runners write `julia-documentation.json` and `r-documentation.json`.
Their default directory, when `DOCS_RECEIPT_DIR` is unset, is
`artifacts/docs-qualification/` under the repository root. Always pass the chosen
directory to `--language-receipt-dir`; the final checker does not read that
environment variable. It executes the Python examples and API doctests, then
requires native receipts matching the current normalized inventory and all
source hashes, complete example IDs and code hashes, positive observed assertion
counts, runtime identities and the staged release binary.

Missing, failed, malformed, partial or stale Julia/R receipts fail `--execute`.
A Python-only execution command therefore cannot qualify the declared three-language
scope. Rerun the native runners after changing their inventoried inputs or native
library; a previous receipt is not reusable solely because it once passed.
Keep the staged binary in `target/release/` while aggregation rechecks its hash.

For whole-content completion, replace `--mode ratchet` with `--mode strict` and
retain both `--execute` and `--language-receipt-dir`. This stricter command
continues to fail while content or historical-qualification obligations remain
open, even after every current language example passes.

For the site, use Node 26, the declared pnpm version and the isolated Python
3.15.2 tooling environment. From the repository root:

```sh
cd docs/astro-site
pnpm install --frozen-lockfile
pnpm test
pnpm build
pnpm check
../../.venv-docs/bin/python ../../scripts/verify_production_docs.py
```

Generated reports belong to this run. They carry source fingerprints and execution
results; the artifact receipt fingerprints both source and output. The CI sequence
builds from a clean checkout and retains its reports. A source/output hash pair
alone is not a cryptographic proof that a particular compiler produced the output.

## Current acceptance boundary

The current Python entry path, scenario-workflow tutorial and eight native
reference examples are bound to ten executable source files. The content baseline
has no accepted current-page exceptions. A new or changed unbound snippet fails
the regression gate. Most retained pages describe the earlier implementation and
remain explicitly outside current API qualification.

Full acceptance additionally requires compiler-verified Rust docs, all supported
language examples, scientific reference review, keyboard/mobile/search behavior,
accessibility checks, performance budgets and reader task evaluation. These
requirements stay open until actual evidence is available. No automated score
declares this documentation scientifically validated or beyond the frontier.

## Publication and recovery

Pull requests build and qualify the site without deploying. Main deployment still
requires the repository's explicit Pages activation setting and the existing
integration process. Confirm the published generator and a representative current
API/tutorial route after deployment. A local artifact pass is not live-site proof.

Preserve the versioned historical reference and source history. Roll back a faulty
publication to a previously verified artifact; do not rewrite historical evidence
to make a new run appear to pass.
