# Starlight-Only Documentation Completion

## Overview

Complete the documentation migration so Astro/Starlight is the only active docs
site, with remaining RST files either removed, converted, generated through
Starlight/polyglot, or retained only as explicitly justified static evidence.

## Functional Requirements

- Migrate remaining core tutorials, modeling tutorials, backend tutorials, and
  bridge docs into Starlight current and latest routes.
- Replace generated Sphinx API bridge coverage with Starlight/polyglot route or
  inventory parity.
- Update tests and static evidence away from active `docs/source/*.rst` paths
  unless the file is an explicit archive/evidence artifact.
- Preserve route stability through redirect inventory and link validation.
- Update migration manifests, cutover inventories, and route coverage.

## Non-Functional Requirements

- No Sphinx build command may remain in active docs CI.
- The docs build must remain reproducible with pnpm and Python 3.14 polyglot
  generation.
- User-facing docs must keep current and latest content aligned.

## Acceptance Criteria

- Remaining RST files are zero or listed in a machine-readable archive/evidence
  allowlist.
- `pnpm --dir docs/astro-site check` passes.
- `uv run nox -s docs` passes.
- Link validation reports no internal link failures.
- Tests fail if removed RST paths are reintroduced as active docs.

## Out Of Scope

- Changing runtime library behavior.
- External registry submission.
