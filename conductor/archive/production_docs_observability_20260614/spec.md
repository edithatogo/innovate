# Production Documentation and Observability

## Overview

The Starlight cutover is locally validated, but production maturity requires search credentials, production verification, observability, user-facing release dashboards, runnable examples, API completeness, and support/maintenance signals.

## Functional Requirements

1. Complete DocSearch production enablement when credentials are available, while preserving safe fallback behavior.
2. Add docs deployment verification for routes, redirects, sitemap, search configuration, versioned docs, and generated API pages.
3. Add observability artifacts for package health, docs freshness, release readiness, registry state, and binding conformance.
4. Add user-facing maturity dashboards or status pages in Starlight.
5. Validate runnable examples and API snippets across Python and bindings.
6. Add support, compatibility, deprecation, and maintenance policy pages appropriate for mature public release.

## Non-Functional Requirements

1. Docs must build without production secrets.
2. Production-only features must have explicit external gate evidence.
3. Status dashboards must be generated from machine-readable artifacts rather than manually duplicated claims.
4. User-facing language must not overclaim external acceptance or full Rust ownership.

## Acceptance Criteria

1. Starlight production verification runs locally and in CI.
2. DocSearch is enabled when credentials exist and gracefully disabled otherwise.
3. Maturity, release, registry, binding, and Rust ownership dashboards are generated from evidence artifacts.
4. Examples and API snippets are validated or explicitly classified.
5. Final GitHub Actions pass proves the docs production lane is healthy.

## Required Operational Cadence

Every task requires a task implementation commit, a separate plan-status commit, phase review with `conductor-review`, push plus GitHub Actions monitoring, final track review, final push, and passing GitHub Actions before archive.

## Out of Scope

1. Hard-coding DocSearch credentials.
2. Removing legacy Sphinx archival sources before a separate archival-retirement decision.
3. Claiming production deployment success without a deployment URL or CI artifact.
