---
title: Retired documentation and current routes
description: Migrate from the retired pre-0.7 documentation to current native examples and recover exact historical sources when needed.
---

The maintained documentation describes the **0.7 development API**, with a Rust
scientific core and Python, Julia and R bindings. On 16 November 2026 the old
instructions were removed from the documentation tree. Earlier web URLs now
display a short retirement notice with a current destination and an immutable
link to their previous source in Git.

## Choose a current starting point

| Your task | Current guidance | Migration boundary |
| --- | --- | --- |
| Install and run a first example | [Installation](/innovate/user-guide/installation/) and [quickstart](/innovate/user-guide/getting-started/) | Use the specified development toolchain and installed native extension. |
| Choose a model or learning path | [Model atlas](/innovate/models/) and [tutorial catalogue](/innovate/tutorials/) | An advertised model is distinct from an executed or scientifically reviewed lesson. |
| Replace an earlier Python class interface | [Python API](/innovate/api/python/) and [native migration](/innovate/native-migration/) | Old `fit_model`, global backend switches and Python numerical fallbacks are outside the maintained API. |
| Call from Julia or R | [Binding overview](/innovate/bindings/) and generated [Julia](/innovate/api/generated/julia/) / [R](/innovate/api/generated/r/) references | Source documentation, installed examples and scientific parity have separate evidence. |
| Understand a specialist scientific feature | [Native capabilities and limits](/innovate/native-migration/) | Grouped fits, latent smoothing, regimes, simulation engines and inference methods have specific semantics and limits. |
| Prepare research software documentation | [Community standards](/innovate/operations/documentation-standards/) and [registry readiness](/innovate/operations/registry-readiness/) | Prepared documents do not establish external acceptance or a qualified release. |

## What was retired

The checked retirement inventory records **263 removed files**: 61 historical
pages at current URLs, 80 versioned pages, 94 Sphinx sources and templates,
17 notebooks, 10 obsolete guides and one legacy page generator. The 141 former
web routes remain available as migration notices; they contain no executable
legacy examples and are excluded from site search.

The retirement contract records each removed path, its SHA-256, Git blob and
source commit. The source check rejects reintroduced files and destinations that
are not current. The production check also inspects every rendered notice for
its source identity, current link, history link, search exclusion and absence of
code blocks. The version selector was removed because there is one maintained
documentation line.

## Recover an earlier analysis

Follow **Read the exact historical source in Git** on the retirement notice for
the earlier URL. The link is pinned to a commit, so it does not change as the
project develops. Restore a separate historical checkout when reproducing an
old analysis, using its original data, dependency environment and assumptions.
The old code's presence in history does not show that it runs with the current
library.

Historical ADRs, Conductor review receipts and `docs/source/_static` evidence
remain in the repository for traceability. They are not served as current
tutorials. Removing old prose does not close scientific feature obligations or
convert prior readiness flags into current results.

## Maintainer workflow

Author current pages in `docs/astro-site/src/content/docs`, attach public API
documentation to the source in each language, and register executable examples
with the [documentation quality contracts](/innovate/maintainers/documentation-quality/).
Use the [language conventions](/innovate/maintainers/language-conventions/) for
docstrings, rustdoc, Julia help and R help. New public-data and advanced lessons
must identify observations, assumptions, expected results and limitations before
they are counted as tutorial coverage.
