# Specification

## Overview

Add a maintainers' `nox` session that wraps the repository version sync/check
tool so local version maintenance is one command away and remains consistent
with the repo's existing `nox`-based Python task workflow.

The session should support both read-only validation and manifest rewriting,
and the repository docs should tell maintainers how to use it during release
prep.

## Functional Requirements

1. Add a `nox` session for release-version synchronization.
2. Default the session to check mode and allow an explicit write mode.
3. Document the local maintainer workflow for running the session.
4. Keep the session aligned with the repository's `uv`-first, `nox`-orchestrated
   Python tooling model.
5. Keep the existing CI version drift guard intact.

## Non-Functional Requirements

1. The session should not introduce a new versioning source of truth.
2. The session should be deterministic and thin over the existing sync script.
3. The docs should stay concise and maintainable.

## Acceptance Criteria

1. A `nox` session exists for version synchronization.
2. The session can run in check mode and write mode.
3. The maintainer docs explain when to use the session.
4. Tests cover the presence of the new session and its documented workflow.

## Out of Scope

1. Replacing the canonical version source.
2. Introducing new release automation services.
3. Changing package-manager version semantics.
