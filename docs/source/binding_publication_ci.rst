Binding publication and CI
==========================

The language bindings are intended to become installable packages, not only
in-repository examples. Publication must stay behind the same kernel contract,
schema compatibility checks, and language-specific CI gates.

Publication targets
-------------------

Python
  Publish the primary package to PyPI/TestPyPI through trusted publishing
  workflows. The package must pass ``uv build`` and
  ``twine check dist/*`` before publication.

TypeScript
  Publish ``innovate.ts`` to npm. The package must pass
  ``npm run schema:check``, ``npm run typecheck``, ``npm test``, and
  ``npm pack --dry-run`` before publication. Package metadata must include a
  public package name, license, repository, type declarations, and an explicit
  ``files`` allow-list so generated caches are not published.

Rust
  Publish ``innovate-rs`` to crates.io once crate metadata, license policy,
  and package contents are finalized. The user-facing suffix is
  ``innovate.rs``; the registry package uses ``innovate-rs`` because Cargo
  crate names do not use dots. The package must pass ``cargo fmt --check``,
  ``cargo clippy --all-targets --all-features -- -D warnings``,
  ``cargo test``, and ``cargo package`` before publication. The crate metadata
  must include a license, repository, readme, description, categories, keywords,
  and ``rust-version`` MSRV policy.

R
  Prepare ``innovate.R`` for R-universe first and CRAN only after support
  boundaries, examples, and reverse dependency expectations are stable. The
  package must pass the binding integration tests, ``R CMD build``,
  ``R CMD check --as-cran``, and local PDF manual generation with
  ``R CMD Rd2pdf bindings/r --output=innovate.R-manual.pdf``. R-universe
  publication is configured in the maintainer's R-universe registry by adding
  this repository as package ``innovate.R`` with subdirectory ``bindings/r``;
  the CI artifact to inspect before enabling publication is the
  ``innovate.R_*.tar.gz`` source tarball from the R package workflow job.
  Generated outputs such as source tarballs, ``.Rcheck`` directories, PDF
  manuals, and built vignette artifacts are release inspection artifacts and
  must not be committed.

  rOpenSci reviewer map:

  * package scope: thin bridge over the shared Python kernel, no R-native
    model duplication;
  * examples and documentation: ``bindings/r/README.md``, ``bindings/r/man/``,
    ``bindings/r/vignettes/``, and the PDF manual workflow;
  * test evidence: ``bindings/r/tests/`` and the package integration tests
    wired into CI;
  * maintenance: R binding maintainers own the package surface, while the
    language-independent kernel remains under the core maintainers;
  * check evidence: the current source package completes ``R CMD build`` and
    ``R CMD check --as-cran --no-manual`` with a single NOTE expected for a
    new submission.

Julia
  The ``Innovate`` Julia package is ready for Julia General registry submission
  through the standard Registrator workflow once package naming, UUID
  ownership, and compatibility bounds are finalized. The user-facing suffix is
  ``innovate.jl``; the registered Julia package keeps the valid module/package
  name ``Innovate``. The project must pass
  ``Pkg.instantiate()`` and ``Pkg.test()``. Registry metadata must include
  dependency compatibility bounds, including ``JSON`` and ``julia``. Registry
  readiness is validated by an installed-package smoke validation that runs the
  bridge from a copied package tree.

Go
  Publish ``innovate.go`` through Go modules by tagging releases that include
  the ``bindings/go`` module path, for example ``bindings/go/v0.5.0``. The
  package must pass ``go test ./...`` and module listing checks before release.
  Go versions are governed by the ``go`` directive in ``bindings/go/go.mod``;
  external module availability is validated through the pushed submodule tag.

C#
  Publish ``innovate.cs`` to NuGet after the .NET 10 and .NET 11 package
  targets pass restore, test, pack, symbol package generation, bridge-content
  inclusion, and NuGet metadata checks. The package metadata must include the
  Apache-2.0 license expression, repository URL and type, project URL, readme,
  package tags, SourceLink settings, and release notes before publication.

Version alignment
-----------------

The primary Python package and every binding package use the same release
version. The current aligned version is ``0.5.0``.

CI requirements
---------------

Every implemented binding needs a dedicated CI job in ``.github/workflows/ci.yml``:

* Python: ``uv build`` plus ``twine check dist/*``
* Rust: ``cargo fmt --check``, ``cargo clippy``, ``cargo test``, and
  ``cargo package``
* TypeScript: ``npm run schema:check``, ``npm run typecheck``, ``npm test``,
  and ``npm pack --dry-run`` on the supported Node matrix
* Go: ``go test ./...``
* Julia: ``Pkg.instantiate()`` and ``runtests.jl`` or ``Pkg.test()`` plus an
  installed-package smoke validation step before publication
* R: dependency installation, integration tests, ``R CMD build``, and
  ``R CMD check --as-cran``. Maintainers should also run
  ``R CMD Rd2pdf bindings/r --output=innovate.R-manual.pdf`` locally before
  publication when R documentation changes.
* C#: ``dotnet test`` and ``dotnet pack`` on .NET 10 and .NET 11

Release workflow
----------------

``.github/workflows/bindings-publish.yml`` is the binding publication gate. It
must run package checks for every language and only publish when release events
or explicit manual inputs are used with the required registry secrets.

For R, the release workflow builds ``innovate.R_*.tar.gz`` and runs
``R CMD check --as-cran --no-manual`` as the CI quality gate. The source tarball
is uploaded as the R package artifact for maintainer inspection before enabling
R-universe or preparing a CRAN submission. The package includes a source vignette
under ``bindings/r/vignettes/`` and release candidates must allow
``R CMD build`` and ``R CMD check --as-cran`` to build and check it by default.
The release workflow also uploads
``r-manual-${{ steps.r_metadata.outputs.package }}-${{ steps.r_metadata.outputs.version }}``
from ``R CMD Rd2pdf`` so maintainers can inspect the PDF manual before
R-universe or CRAN publication. Any bypass of vignette checks must be documented
as a temporary maintainer exception, not treated as the normal publication path.

For Julia, the release workflow runs an installed-package smoke step before the
registry guidance message. The smoke step exercises a copied package tree,
confirms ``inst/python/kernel_bridge.py`` is present, and calls the bridge
through the configured Python launcher.

For NuGet, the release workflow performs a dry-run style artifact gate before
``dotnet nuget push``: it packs ``innovate.cs``, requires both ``.nupkg``
and ``.snupkg`` outputs, inspects the generated ``.nuspec`` for publication
metadata, and verifies the package contains ``README.md`` and the packaged
``contentFiles/any/any/innovate/kernel_bridge.py`` bridge script.
