Binding publication and CI
==========================

The language bindings are intended to become installable packages, not only
in-repository examples. Publication must stay behind the same kernel contract,
schema compatibility checks, and language-specific CI gates.

Publication targets
-------------------

Python
  Publish the primary package to PyPI/TestPyPI through the existing Python
  package release workflow.

TypeScript
  Publish ``innovate.ts`` to npm. The package must pass
  ``npm run schema:check``, ``npm run typecheck``, ``npm test``, and
  ``npm pack --dry-run`` before publication.

Rust
  Publish ``innovate-rs`` to crates.io once crate metadata, license policy,
  and package contents are finalized. The user-facing suffix is
  ``innovate.rs``; the registry package uses ``innovate-rs`` because Cargo
  crate names do not use dots. The package must pass ``cargo fmt --check``,
  ``cargo test``, and ``cargo package`` before publication.

R
  Prepare ``innovate.R`` for R-universe first and CRAN only after support
  boundaries, examples, and reverse dependency expectations are stable. The
  package must pass ``R CMD build`` and ``R CMD check --as-cran``. R-universe
  publication is configured in the maintainer's R-universe registry by adding
  this repository as package ``innovate.R`` with subdirectory ``bindings/r``;
  the CI artifact to inspect before enabling publication is
  ``innovate.R_*.tar.gz``.

Julia
  Prepare the ``Innovate`` Julia package for Julia General registry submission
  through the standard Registrator workflow after package naming, UUID
  ownership, and compatibility bounds are finalized. The user-facing suffix is
  ``innovate.jl``; the registered Julia package keeps the valid module/package
  name ``Innovate``. The project must pass
  ``Pkg.instantiate()`` and ``Pkg.test()``.

Go
  Publish ``innovate.go`` through Go modules by tagging releases that include
  the ``bindings/go`` module path, for example ``bindings/go/v0.5.0``. The
  package must pass ``go test ./...`` and module listing checks before release.

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

* Rust: ``cargo fmt --check`` and ``cargo test``
* TypeScript: ``npm run schema:check``, ``npm run typecheck``, and ``npm test``
* Go: ``go test ./...``
* Julia: ``Pkg.instantiate()`` and ``runtests.jl`` or ``Pkg.test()``
* R: dependency installation plus ``Rscript bindings/r/tests/run.R``
* C#: ``dotnet test`` on .NET 10 and .NET 11

Release workflow
----------------

``.github/workflows/bindings-publish.yml`` is the binding publication gate. It
must run package checks for every language and only publish when release events
or explicit manual inputs are used with the required registry secrets.

For NuGet, the release workflow performs a dry-run style artifact gate before
``dotnet nuget push``: it packs ``innovate.cs``, requires both ``.nupkg``
and ``.snupkg`` outputs, inspects the generated ``.nuspec`` for publication
metadata, and verifies the package contains ``README.md`` and the packaged
``contentFiles/any/any/innovate/kernel_bridge.py`` bridge script.
