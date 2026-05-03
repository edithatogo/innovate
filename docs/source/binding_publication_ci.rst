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
  Publish ``innovate-typescript-bindings`` to npm. The package must pass
  ``npm run schema:check``, ``npm run typecheck``, ``npm test``, and
  ``npm pack --dry-run`` before publication.

Rust
  Publish ``innovate-rust`` to crates.io once crate metadata, license policy,
  and package contents are finalized. The package must pass ``cargo fmt
  --check``, ``cargo test``, and ``cargo package`` before publication.

R
  Prepare the R binding for R-universe first and CRAN only after support
  boundaries, examples, and reverse dependency expectations are stable. The
  package must pass ``R CMD build`` and ``R CMD check``.

Julia
  Prepare the Julia binding for Julia General registry submission through the
  standard Registrator workflow after package naming, UUID ownership, and
  compatibility bounds are finalized. The project must pass
  ``Pkg.instantiate()`` and ``Pkg.test()``.

Go
  Publish the Go binding through Go modules by tagging releases that
  include the ``bindings/go`` module path, for example
  ``bindings/go/v0.5.0``. The package must pass ``go test ./...`` and module
  listing checks before release.

C#
  Publish ``Innovate.Kernel`` to NuGet after the .NET 10 and .NET 11 package
  targets pass restore, test, pack, and NuGet metadata checks.

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
