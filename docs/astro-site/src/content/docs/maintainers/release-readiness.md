---
title: Release Readiness
description: Mature release gate report generation and state interpretation.
---

# Release Readiness

Run the maintainer report with:

```bash
uv run nox -s release_readiness
```

The command writes `docs/source/_static/release_readiness/readiness-report.json`
from the committed release-readiness contract. It is intentionally conservative:
missing, stale, manual-only, deferred, unknown, or failing evidence keeps the
repository in a release candidate state.

## State Boundaries

- `release candidate`: feature work may be complete, but at least one required quality, security, provenance, reproducibility, compatibility, Rust, docs, package, or binding artifact is missing or not passing.
- `release-ready`: every required artifact is fresh and passing, so a maintainer can decide whether to cut a public release.
- `public release`: a maintainer-approved tag, release artifact, and release notes exist. Release-ready status is only the prerequisite.
- `external acceptance`: package registries, scientific communities, or HPC ecosystems have accepted the project. This remains separate from release readiness because third-party review and credentials control it.

Use the report as the release review input. Do not substitute registry receipts,
HPC readiness dossiers, or external acceptance notes for the mature release
gate.

## Final gate sequence

The dry-run record lives at
`docs/source/_static/release_readiness/release-dry-run.json`.

1. Generate supply-chain evidence with `uv run nox -s release_supply_chain`.
2. Generate reproducibility evidence with `uv run nox -s release_reproducibility`.
3. Generate the release-readiness report with `uv run nox -s release_readiness`.
4. Run package dry-runs, including `uv run nox -s package` for Python and the language-specific commands in `release-dry-run.json`.
5. Review release candidate blockers from `readiness-report.json`.
6. Require maintainer approval before public release, tagging, registry publication, or external acceptance claims.
