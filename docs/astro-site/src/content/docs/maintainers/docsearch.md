---
title: DocSearch Gate
description: Production Algolia DocSearch enablement and fallback policy.
---

# DocSearch Gate

The production search provider is Algolia DocSearch. The Starlight build must
work without production secrets, so the active config only enables DocSearch
when all required environment variables are present.

Required deployment environment variables:

- `ALGOLIA_APP_ID`
- `ALGOLIA_API_KEY`
- `ALGOLIA_INDEX_NAME`

Statuses:

- `enabled`: all required variables are present and the DocSearch plugin is
  loaded.
- `disabled_without_credentials`: local builds and pull-request CI omit
  DocSearch when credentials are absent.
- `external_credentials_required`: production search remains gated on
  deployment-environment credentials.

Do not hard-code DocSearch credentials in the repository, generated evidence, or
Starlight content. Use the deployment environment or GitHub Actions environment
secrets for production enablement.

Machine-readable evidence:

- `docs/source/_static/astro_starlight/docsearch_gate.json`
- `docs/source/_static/astro_starlight/production_docs_verification.json`
