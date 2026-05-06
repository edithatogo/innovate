Polyglot repository and documentation architecture
==================================================

Purpose
-------

This page is the navigation and ownership map for the polyglot ``innovate``
repository. It keeps the current source layout stable while making the docs
clear for users, binding authors, HPC administrators, and maintainers.

No source tree move is required for the current release. The existing paths
remain canonical until a future track provides a migration plan, redirects, and
backward-compatible link checks.

Audience navigation
-------------------

Users
  Start with :doc:`innovate.kernel`, :doc:`tutorials`, and :doc:`bindings`.
  These pages explain the stable functional kernel, primary examples, and
  language-specific entry points without requiring repository-layout knowledge.

Binding authors
  Use :doc:`bindings`, :doc:`binding_publication_ci`, and the binding README
  files under ``bindings/*/README.md``. Binding work must stay thin over the
  kernel contract and preserve package-manager metadata for each language.

HPC administrators
  Use :doc:`scientific_hpc_readiness_roadmap`, :doc:`xla_backend_strategy`,
  and :doc:`rust_core_roadmap`. These pages distinguish deployment evidence,
  accelerator evidence, and native-core promotion from public binding APIs.

Maintainers
  Use :doc:`adr`, :doc:`release_notes_policy`, :doc:`binding_publication_ci`,
  and the Conductor track archive. Maintainer material should record why a path
  exists, who owns it, and which release gate protects it.

Ownership map
-------------

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Area
     - Canonical paths
     - Ownership rule
   * - Core package
     - ``src/innovate/``, ``docs/source/innovate.kernel.rst``,
       ``docs/source/api_reference.rst``
     - Python remains the reference user API while the functional kernel and
       schema contract define portable behavior.
   * - Language bindings
     - ``bindings/r/README.md``, ``bindings/rust/README.md``,
       ``bindings/julia/README.md``, ``bindings/typescript/README.md``,
       ``bindings/go/README.md``, ``bindings/csharp/README.md``,
       ``docs/source/bindings.rst``
     - R, Rust, Julia, TypeScript, Go, and C# bindings are language-owned
       adapters over the same kernel contract, not independent model forks.
   * - Packaging and release
     - ``docs/source/binding_publication_ci.rst``,
       ``docs/source/release_notes_policy.rst``, package manifests, and
       release workflows
     - Package-manager metadata and release gates are owned with each binding,
       then checked by the shared publication documentation and tests.
   * - Scientific and HPC ecosystem
     - ``docs/source/scientific_hpc_readiness_roadmap.rst``,
       ``docs/source/xla_backend_strategy.rst``,
       ``docs/source/rust_core_roadmap.rst``
     - HPC, accelerator, ABI, and Rust-core evidence lives in docs until a
       dedicated implementation track promotes code or packaging artifacts.
   * - Community submission dossiers
     - Future dossier docs and Conductor archives that reference reviewer
       evidence
     - Submission material should cite stable docs and release artifacts rather
       than duplicate package metadata.

Source-layout guidance
----------------------

Keep the current source layout. The repository already separates concerns:

* ``src/innovate/`` owns the Python public API, reference implementation, and
  stable functional kernel surface.
* ``bindings/`` owns language package surfaces and package-manager metadata.
* ``docs/source/`` owns Sphinx navigation, user docs, release docs, roadmaps,
  and reviewer-facing architecture decisions.
* ``conductor/tracks/`` owns active plans. ``conductor/archive/`` owns completed
  evidence for decisions and release gates.

Future source moves need a separate track before implementation. That track
must identify the old path, new path, public docs link, release artifact impact,
and package-manager impact before any files move.

Migration and redirect rules
----------------------------

Existing paths remain canonical unless the migration track proves a move is
necessary. A move is acceptable only when all of these conditions are met:

* the old docs page keeps a stable redirect or explicit forwarding stub;
* the new page is added to the Sphinx toctree before the old path is removed;
* tests cover the old path, new path, and affected audience navigation;
* package files, generated artifacts, and publication gates are not renamed as
  part of a docs-only cleanup;
* release notes call out any user-visible path change.

These rules protect existing links from README files, package registries,
release artifacts, Conductor archives, and external reviewer dossiers.
