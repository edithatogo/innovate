# Astro/Starlight documentation workspace

This is Innovate's active documentation site. Current API reference pages are
generated from maintained binding source; historical pages remain visibly
classified. The published site must be verified separately from a local build.

## Reproducible local build

Use Node 26 and the package manager declared in `package.json` (pnpm 12.18.0).
The checked-in lockfile and release-age policy are mandatory. From the repository
root, create the isolated Python 3.15.2 tooling environment for a new checkout;
reuse it when it already exists:

```sh
uv venv .venv-docs --python 3.15.2
cd docs/astro-site
pnpm install --frozen-lockfile
pnpm test
pnpm build
pnpm check
../../.venv-docs/bin/python ../../scripts/check_api_documentation.py --mode strict --output .reports/api.json
../../.venv-docs/bin/python ../../scripts/check_polyglot_api_documentation.py --check
../../.venv-docs/bin/python ../../scripts/check_documentation_content.py --mode ratchet --output .reports/content.json
../../.venv-docs/bin/python ../../scripts/verify_production_docs.py
```

The AST-based generator and contract checks need only the Python standard library.
`INNOVATE_DOCS_PYTHON` selects another pinned interpreter when needed; the older
`STARLIGHT_POLYGLOT_PYTHON` name remains a compatible alias. The site requires
a nonempty description for every content page.

`pnpm build` generates the Python, Julia and R API into the ignored `api/generated/` directory
before Astro loads the collection. `pnpm check` verifies those files still match
their sources and then type-checks the site. Do not edit generated reference
pages; modify the binding docstrings. The generator refuses to overwrite
unowned or altered output.

The old vendored polyglot dependency remains retained for historical traceability,
but its incomplete output integration is not the active reference generator.
`astro.config.mjs` is the active site configuration; `starlight.config.mjs`
is a retained historical configuration, not a second source of runtime settings.

## Execute documentation across Python, Julia and R

First install the current native Python package in `.venv-native` using the
[main README's development setup](../../README.md#development). Use the declared
Python 3.15.2 and Rust/Cargo 1.99.1 toolchains. The following sequence matches the
Linux documentation job; it also needs Julia 1.13 and an R runtime that provides
`tools::sha256sum`, plus the existing `jsonlite` dependency. Run it from the
repository root, after the source and site checks above have passed:

```sh
.venv-native/bin/python scripts/generate_api_reference.py
.venv-native/bin/python scripts/check_native_documentation.py --output docs/astro-site/.reports/native-api.json
RUSTDOCFLAGS='-D missing_docs -D rustdoc::broken_intra_doc_links -D rustdoc::invalid_codeblock_attributes' cargo doc --workspace --no-deps --locked -j 1
cargo test --workspace --doc --locked -j 1

cargo build --release --locked -p innovate-ffi -j 1
mkdir -p bindings/julia/deps bindings/r/inst/lib
cp target/release/libinnovate_ffi.so bindings/julia/deps/
cp target/release/libinnovate_ffi.so bindings/r/inst/lib/

export DOCS_RECEIPT_DIR="$PWD/docs/astro-site/.reports"
unset INNOVATE_NATIVE_LIBRARY
julia --project=bindings/julia -e 'using Pkg; Pkg.instantiate()'
julia --project=bindings/julia scripts/check_julia_documentation.jl

export R_LIBS_USER="$DOCS_RECEIPT_DIR/r-library"
mkdir -p "$R_LIBS_USER"
Rscript -e 'install.packages("jsonlite", repos="https://cloud.r-project.org", lib=Sys.getenv("R_LIBS_USER"))'
R CMD INSTALL --library="$R_LIBS_USER" bindings/r
Rscript scripts/check_r_documentation.R

.venv-native/bin/python scripts/check_documentation_content.py \
  --mode ratchet --execute --python .venv-native/bin/python \
  --language-receipt-dir "$DOCS_RECEIPT_DIR" \
  --output "$DOCS_RECEIPT_DIR/execution.json"
```

On another platform, use its release-library filename and staging convention
from the binding-conformance workflow. Keep that same release binary in
`target/release/` until aggregation finishes: receipt validation rechecks its hash.

The Julia and R runners write `julia-documentation.json` and
`r-documentation.json` into `DOCS_RECEIPT_DIR`. If this variable is unset, their
default is `artifacts/docs-qualification/` under the repository root. Pass the
same directory through `--language-receipt-dir`; the content checker does not
infer it from the environment. Rerun the native runners when their inventoried
sources, runner scripts or release binary change.

The final command executes the Python examples and source-matched API doctests,
then verifies both native-language receipts against the current inventory,
source hashes, exact example IDs, code hashes and observed assertion counts.
Running `--execute` without the required Julia/R receipts fails, even when every
Python example passes. Missing, stale, partial and failed receipts also fail the
gate. The validator checks evidence consistency; it does not authenticate the
receipt's origin or establish scientific validity.

The Python content contract bounds execution, checks assertions and exact
expected stdout, and distinguishes executed outcomes from static inventory.
The native extension check compares the complete runtime export inventory with
its stub and validates runtime docstrings. Ten checked Python files cover the
two tutorials and eight native reference examples; displayed fences or raw
imports must match their declared executable source exactly. The declared Julia
and R inventories add seven and six native-language example blocks respectively.

## Quality and publication

Contracts are in `../contracts/`. Audit, ratchet and strict completion have
different meanings: a ratchet pass does not erase the documented backlog or
qualify historical reference pages. See the
[maintainer guide](src/content/docs/maintainers/documentation-quality.md) and
[Conductor track](../../conductor/tracks/documentation_quality_contracts_20261115/index.md).

A source-only run can report `current_structure_complete: true` while execution
remains `not_run`. A successful Python run sets `python_execution_status` and can
set `current_python_content_complete`; `current_content_complete` also requires
successful execution evidence for every language declared in
`generated_reference_languages`. `full_scope_complete` and `complete` additionally
include retained historical material, whose qualification remains open.
Consequently, replacing `--mode ratchet` with `--mode strict` in the final command
continues to fail while historical obligations remain. Rust compiler checks,
native-extension documentation, rendered artifacts and reader evaluation remain
separate requirements for wider documentation acceptance.

The artifact verifier inspects emitted HTML, all local links/fragments, canonical
URLs, sitemap coverage, a populated Pagefind index, basic semantics and HTML size.
It fingerprints current inputs and output; it never copies a historical `passed`
claim into a new receipt. Browser accessibility and live deployment remain distinct.

GitHub Actions builds PRs without deploying. Main deployment requires both the
site and native-documentation gates plus the existing Pages activation setting.
PR build concurrency is separate from main publication. All reports are retained
as workflow artifacts under the run that produced them.
