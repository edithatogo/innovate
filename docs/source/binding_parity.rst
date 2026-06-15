Binding parity
==============

Innovate keeps supported language bindings aligned through the shared kernel
contract, golden fixtures, and language-native package checks. The current
binding families are Python, Rust, R, Julia, TypeScript, Go, and C#.

Evidence sources
----------------

Machine-readable evidence lives in:

* ``docs/source/_static/binding_conformance_inventory.json``
* ``docs/source/_static/binding_golden_fixtures.json``
* ``docs/source/_static/binding_hardening_evidence.json``

The ``binding_conformance_ci`` page documents the local fallback command and
the ``Binding Conformance`` workflow artifact. Package-manager receipts,
accepted publications, and maintainer-managed handoffs remain in
``registry_submission_receipts``.

Release interpretation
----------------------

The shared conformance evidence proves contract alignment across supported
bindings. It does not replace language-native package checks before release:
``cargo test``, ``npm test``, ``R CMD check``, ``Pkg.test()``,
``go test ./...``, and ``dotnet test`` remain required package checks for the
corresponding registry submissions.

Parity status
-------------

* Python is the canonical reference package and version source.
* Rust is the native runtime and crate surface.
* R, Julia, TypeScript, Go, and C# are thin package surfaces over the shared
  kernel contract.

