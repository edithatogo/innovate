---
title: Deployment Readiness
description: Production documentation deployment checklist and rollback notes.
slug: latest/maintainers/deployment-readiness
---

# Deployment Readiness

This page records the Starlight production deployment checklist and rollback
path.

Evidence:

- `docs/source/_static/astro_starlight/deployment_readiness.json`
- `docs/source/_static/astro_starlight/production_docs_verification.json`
- `docs/source/_static/astro_starlight/example_validation.json`

Release checklist:

- Run `pnpm build` and `pnpm check` from `docs/astro-site`.
- Run `uv run nox -s examples production_docs`.
- Confirm `ENABLE_PAGES_ACTIONS_DEPLOY` before production deployment.
- Confirm the uploaded Pages artifact is `docs/astro-site/dist/`.

Rollback:

- Disable `ENABLE_PAGES_ACTIONS_DEPLOY` to stop automatic Pages deployment.
- Revert the docs change or republish the last passing Pages artifact.
- Preserve failed deployment evidence for triage.
