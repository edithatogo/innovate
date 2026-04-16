# Specification: Canonical Public API and Package Topology

## Overview

Define a stable, documented public API for `innovate` and rationalize the package layout so the library has one canonical import path per concept. This track creates the package-topology foundation required for future kernel-contract, plugin, and multi-language work.

## Functional Requirements

1. Define the canonical public namespaces for model families, fitters, diagnostics, backend access, and any new `innovate.core` surface.
2. Inventory and resolve duplicate or ambiguous package areas, including current overlaps such as `backend` vs `backends` and `ecosystem` vs `ecosystems`.
3. Expose an explicit public API from top-level package entry points so examples, docs, and downstream users have a documented import contract.
4. Add a model capability registry or equivalent discoverability surface that lets callers inspect supported features for each stable model family.
5. Add a deprecation path for any legacy import path that will no longer be canonical.

## Non-Functional Requirements

1. The public API must be documented and versionable under semver.
2. The import topology must be deterministic and avoid circular-import traps.
3. Backward compatibility should be preserved where practical through deprecation warnings rather than silent breakage.
4. The final layout must be suitable for future R/Julia wrappers and a language-neutral kernel contract.

## Acceptance Criteria

1. `src/innovate/__init__.py` exposes a documented, tested top-level API.
2. Duplicate namespaces are either removed or explicitly deprecated with tests covering the behavior.
3. Docs and examples use the canonical imports only.
4. A capability registry exists for stable model families and is covered by unit tests.

## Out of Scope

1. Implementing the language-neutral functional kernel itself.
2. Adding new model families.
3. Creating non-Python bindings.
