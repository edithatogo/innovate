# Migration Plan: Sphinx → Starlight

## Timeline

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0: Scaffold | Create docs/astro-site directory, install dependencies, configure Astro/Starlight | ✅ Complete |
| Phase 1: Configuration | Configure astro.config.mjs with Starlight, polyglot, versions, link validator plugins | ✅ Complete |
| Phase 2: Content Creation | Write real documentation content (index, getting-started, api-reference, user-guide) | ✅ Complete |
| Phase 3: Content Migration | Migrate existing Sphinx pages to Starlight MDX/Markdown | ✅ Complete |
| Phase 4: CI/CD | Update GitHub Actions workflow for GitHub Pages deployment | ✅ Complete |
| Phase 5: Validation | Validate all links, verify build, ensure accessibility | 🔄 In Progress |
| Phase 6: Launch | Set custom domain, configure redirects, announce migration | 📅 Planned |

## Steps Executed

### 1. Astro/Starlight Site Scaffold

- Created `docs/astro-site/` with `package.json`, `astro.config.mjs`, `tsconfig.json`
- Configured dependencies:
  - `@astrojs/starlight` — Documentation theme
  - `astro` — Static site generator
  - `starlight-polyglot` — Python API doc generation (file: dependency to local package)
  - `starlight-versions` — Multi-version support
  - `starlight-links-validator` — Link validation in CI
- Set up `src/content.config.ts` with Starlight docs loader and schema

### 2. Configuration

- Configured `astro.config.mjs`:
  - Site URL: `https://edithatogo.github.io/innovate`
  - Base path: `/innovate` (GitHub Pages project site)
  - Integrated starlight-polyglot with Python entry points pointing to `../../src/innovate`
  - Added sidebar navigation with Getting Started, User Guide, API Reference, Maintainers, Operations, Architecture, Migration sections
- Created `public/.nojekyll` for GitHub Pages compatibility

### 3. Documentation Content

Created the following documentation pages with full content:

- **Index** (`index.md`): Landing page with hero, feature cards, quick example, and next steps
- **Getting Started** (`user-guide/getting-started.mdx`): Installation instructions, first model walkthrough, forecasting example
- **Installation** (`user-guide/installation.md`): Detailed install guide for pip, uv, conda, Docker, and all backends
- **Fitting Models** (`user-guide/fitting.md`): Available models, fitting methods (MLE, NLS, Bayesian), model comparison, ensembles
- **Forecasting** (`user-guide/forecasting.md`): Forecast generation, confidence intervals, scenario analysis, peak timing
- **Backends** (`user-guide/backends.md`): NumPy, JAX, Numba, Bayesian backends with benchmarks and usage examples
- **API Reference** (`api/python.md`): Complete API documentation for all public functions, model families, diagnostics, ABM, Arrow interchange, and plugin system

### 4. CI/CD Pipeline

Updated `.github/workflows/docs.yml`:
- Trigger on push/PR to `main`
- Build step with pnpm install and astro build
- Deploy to GitHub Pages on push to `main`
- Uses `actions/deploy-pages@v4` for deployment
- Node.js 22 with pnpm caching

### 5. Conductor Documentation

Created `conductor/` directory with:
- `tech-stack.md` — Full technology stack reference
- `tracks/migrate_starlight/metadata.json` — Track metadata
- `tracks/migrate_starlight/spec.md` — Migration specification
- `tracks/migrate_starlight/plan.md` — This plan

### 6. Repository Configuration

- Added `.gitignore` entries for `node_modules/` and `dist/`
- Git remote configured to `https://github.com/edithatogo/innovate.git`

## Migration Inventory

### Migrated Content

| Sphinx Source | Starlight Target | Status |
|---------------|-----------------|--------|
| `docs/index.md` | `/` (index.md) | ✅ |
| `docs/quickstart.md` | `/user-guide/getting-started/` | ✅ |
| `docs/api.md` | `/api/python/` + polyglot generated | ✅ |
| `docs/source/tutorials.rst` | `/tutorials/` | ✅ |
| Architecture docs | `/architecture/` | ✅ |
| Operations docs | `/operations/` | ✅ |
| Maintainer docs | `/maintainers/` | ✅ |

### Remaining Work

- Set up custom domain (docs.innovate.example)
- Configure Algolia DocSearch for full-text search
- Set up Sphinx → Starlight redirect layer
- Validate all migrated content for accuracy
- Performance testing and Lighthouse audit

## Rollback Plan

If the Starlight site has issues:
1. The Sphinx site remains available and continues to build
2. Set `docs.yml` to deploy Sphinx instead
3. Redirect `docs.innovate.example` back to Sphinx output
4. Fix Starlight issues and re-deploy
