---
title: Language documentation conventions
description: Native Python, Rust, Julia and R documentation styles, enforceable source contracts and the evidence needed to qualify examples.
---

Document an interface where its users look for help: Python docstrings, Rust
rustdoc, Julia's `Base.Docs`, and R's source comments and installed help. Generated
Astro reference pages reuse those sources. Change the source contract and its
examples together, then refresh the generated reference.

The policy in
[api-documentation.json](https://github.com/edithatogo/innovate/blob/main/docs/contracts/api-documentation.json)
names all four language conventions. The Python source gate checks that its
policy agrees with the implemented checkers. The generated
[polyglot inventory](https://github.com/edithatogo/innovate/blob/main/docs/contracts/polyglot-api-documentation.json)
records the Julia/R structural rules, source identities and asserted examples.
Its runtime status remains separate from structural coverage.

## Shared scientific and transport contracts

A useful parameter description answers what the value means, its type and
shape, its unit, its allowed domain, and what an omitted or null value does.
Include the details applicable to that interface rather than repeating every
possible heading. In particular:

| Contract | Information a caller needs |
| --- | --- |
| Observation quantity | Whether the model expects cumulative adoption, an instantaneous level or rate, interval counts, shares, or another quantity. Explain any actual conversion. |
| Time and parameters | The numeric time unit and origin chosen by the caller; corresponding parameter units; grid ordering and domain requirements. A derivative has output units per time unit, including on an irregular grid. |
| Shape and nulls | Row/series orientation, order preservation, singleton arrays, supported missing values, and whether a helper returns an envelope, selected result, or table. |
| Errors | The exception or error condition, its cause, and any structured fields. Distinguish a rejected outer request from failures retained inside batch or scenario rows. |
| State and ownership | Reusable fit state, shallow or deep copies, mutation, eager collection, and allocation/free pairs at a native boundary. |
| Interpretation | What a successful computation establishes. A fitted curve alone does not establish identifiability, calibrated uncertainty or empirical validity. |

For example, `predict(..., quantity="rate")` needs to explain the time unit and
instantaneous derivative semantics. Saying only “returns predictions” leaves the
reader unable to distinguish a level, a derivative and a count over an interval.
Likewise, R's `kernel_call(..., raw=TRUE)` preserves response text while still checking
the outer error. Raw text is a representation choice, not an alternative error
policy.

## Python: PEP 257 and Google sections

Use an attached triple-quoted docstring with a clear summary and a blank line
before the details, following [PEP 257](https://peps.python.org/pep-0257/). Keep
summaries consistent within the file. New binding documentation uses
[Google-style sections](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings):
`Args:`, `Returns:` or `Yields:`, `Raises:`, and `Attributes:` where applicable.
Type annotations carry types; the descriptions carry units, shape, defaults and
meaning. Document `*args` and `**kwargs` using their actual names, and omit an
ordinary instance method's `self` parameter.

The source parser also accepts existing
[NumPy-style sections](https://numpydoc.readthedocs.io/en/latest/format.html).
Choose one style within a docstring. An alternate spelling does not create a new
contract: `Args`, `Arguments` and `Parameters` identify the same parameter section.
Duplicate sections and repeated parameter, error or attribute entries fail, even
when a later entry would otherwise hide the earlier description.

The import-free source audit inventories modules, classes, methods, functions,
nested helpers and type stubs. Public callable parameter names must match their
signatures; returned values need descriptions; direct raises must be named; public
annotated attributes need descriptions. Innovate additionally requires an error
section for public callables. Describe actual interface failures, including the
native error representation, rather than filling it with hypothetical exceptions.

Use small, self-contained doctests with assertions about meaning: ordering,
schema, expected values within a stated tolerance, or a documented failure.
The binding's `execute` example checks the schema version and the discovered
model key. The installed PyO3 module is checked separately against its source
stub, because source docstrings alone cannot establish what Python `help()` sees.

## Rust: rustdoc and explicit unsafe contracts

Use `//!` for module or crate introductions and `///` for public items. Follow the
[rustdoc book](https://doc.rust-lang.org/rustdoc/how-to-write-documentation.html)
and [Rust API documentation guidelines](https://rust-lang.github.io/api-guidelines/documentation.html).
Link related items with intra-doc links such as ``[`innovate_free`]`` so compiler
checks can detect broken targets. Include a useful runnable example, or link to
an applicable example on a related item. Fallible examples should propagate
ordinary errors with `?` while asserting the outcome they demonstrate.

Use `# Errors` for fallible operations, `# Safety` for unsafe caller obligations,
and `# Panics` for relevant caller-reachable panic conditions. These are review
requirements; the configured missing-docs and link lints do not prove the prose
is complete. For the C ABI, describe the return pointer and out parameters
explicitly because a C error return is not a Rust `Result`.

The [FFI source](https://github.com/edithatogo/innovate/blob/main/rust/crates/innovate-ffi/src/lib.rs)
documents byte lengths, NUL termination, borrowed-buffer lifetimes, writable
output storage, error initialization and the matching free function. Its
`innovate_call` example parses the response and returns its allocation exactly
once. Invalid pointers are caller-contract violations; a JSON error envelope
does not make them recoverable errors.

Rust qualification requires the pinned compiler, rustdoc lints and doctests.
Private Rust functions exported through PyO3 also need Python-facing runtime
documentation; Rust visibility checks alone do not inventory that surface.
Cargo does not run doctests for the `innovate-python` crate's `cdylib` target.
Record that limitation alongside the available Rust doctest results and the
separate installed Python documentation and example checks.

## Julia: attached Base.Docs Markdown

Follow the [Julia documentation manual](https://docs.julialang.org/en/v1/manual/documentation/):
place the triple-quoted string immediately before the definition, begin with an
indented signature, then a summary and Markdown details. This makes the same
source available through Julia help and `Base.Docs`.

The bounded Innovate parser requires the documented signature and defaults to
match the source. Use `# Arguments` with one named bullet per argument,
`# Returns` for the result representation, and `# Throws` for native and Julia
failures. Document exported struct fields under `# Fields`. Repeated section or
argument entries fail. The parser supports the current explicit exports and
declarations; new declaration forms need a deliberate parser update.

`kernel_call` documents that JSON arrays become vectors, JSON null becomes
`nothing`, and a native validation report becomes `DatasetValidationError`.
Do not silently substitute `missing` for these meanings: an individual helper,
such as the SARIMA wrapper, may have a separately documented conversion.

Innovate's reviewed `# Examples` use literal `julia` fences and explicit `@assert`
calls. The native runner evaluates each in a fresh module and records the
assertions that actually ran. Julia's manual also describes Documenter.jl's
`jldoctest` format; that format needs a separate integration before this
repository's runner can qualify it. Changing a fence label alone does not turn
an example into executed evidence.

## R: roxygen comments and matching Rd help

Write attached `#'` documentation using the
[roxygen conventions](https://roxygen2.r-lib.org/articles/rd.html), and keep the
checked-in help consistent with
[Writing R Extensions](https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Writing-R-documentation).
The current parser accepts one `@param` entry per source argument, one `@return`
description, and one `@section Errors:`. Use R terms such as a named list, raw
vector, character scalar, `NULL`, or an error condition where they describe the
actual interface. Use Rd markup such as `\code{}` and `\link{}` for help text.

Update the corresponding `man/*.Rd` page in the same change. Its aliases must
resolve exported functions; `\usage` signatures and defaults must match the
source; `\arguments` must cover the parameter union when one page documents
several helpers; `\value` must describe each return representation. Keep the
Errors section and descriptive prose aligned as a review obligation. The source
checker validates structure and signatures, not word-for-word agreement of all
prose.

For the JSON bridge, explain list decoding and singleton array preservation.
For exact existing JSON, describe `kernel_call_json` and the effect of `raw`;
decimal formatting cannot recover precision already lost in an R numeric value.
For validation conditions, name the class and the `code` and `validation` fields
so readers can handle the error programmatically.

Place executable examples in `\examples` with outcome checks using `stopifnot`.
The native runner uses R's own Rd parser and example extraction, executes the
code against the installed package, and counts observed assertions. Skipped or
conditional examples do not satisfy this qualification path. The existing
pipeline uses `tools` and the binding's `jsonlite` dependency; maintaining these
contracts does not require adding another documentation framework.

## Run the checks and record their scope

Use the repository's pinned toolchains and installed native bindings as described
in the [site workspace README](https://github.com/edithatogo/innovate/blob/main/docs/astro-site/README.md).
The commands below start at the repository root. Here `python` means that pinned
Python environment; the native check additionally requires the installed
extension built from the source under review.

| Evidence | Command | What a pass establishes |
| --- | --- | --- |
| Python source | `python scripts/check_api_documentation.py --mode strict` | Inventoried source documentation satisfies the structural policy. |
| Julia/R source | `python scripts/check_polyglot_api_documentation.py --check` | The recorded inventory matches the current source, help files, examples and parser. |
| Installed Python help | `python scripts/check_native_documentation.py` | Observed native exports and runtime docstrings satisfy the stub contract; the report separately states its binary/source identity limits. |
| Rust reference | `RUSTDOCFLAGS='-D missing_docs -D rustdoc::broken_intra_doc_links -D rustdoc::invalid_codeblock_attributes' cargo doc --workspace --no-deps --locked` | The compiler's documentation and link checks pass for this build. |
| Rust examples | `cargo test --workspace --doc --locked` | The selected toolchain executed the crate doctests. |
| Julia help/examples | `julia --project=bindings/julia scripts/check_julia_documentation.jl` | Loaded documentation and the inventoried examples passed against the staged native library. |
| R help/examples | `Rscript scripts/check_r_documentation.R` | Installed help and inventoried examples passed the R parser and native runner. |

After changing Julia/R source documentation, refresh the inventory with
`python scripts/check_polyglot_api_documentation.py --write-inventory`; inspect
the diff before running `--check`. Refresh generated references with
`python scripts/generate_api_reference.py`, and then run the wider
[documentation quality checks](/innovate/maintainers/documentation-quality/).
Source checks, generated pages and compilation have distinct evidence scopes.
Record unavailable or unexecuted checks explicitly, and retain scientific review
for claims that no structural gate can decide.
