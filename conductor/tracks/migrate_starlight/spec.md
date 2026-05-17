# Migration Specification: Sphinx → Starlight

## Objective

Replace the existing Sphinx-based documentation system with an Astro/Starlight site that provides:
- A modern, responsive documentation experience
- Auto-generated API documentation from Python source via starlight-polyglot
- Multi-version documentation support
- GitHub Pages deployment
- Side-by-side operation during migration (parallel-run)

## Requirements

### Functional

1. **Documentation Pages** — All content from the Sphinx source (`docs/source/`) must be migrated to Starlight MDX/Markdown pages under `docs/astro-site/src/content/docs/`.
2. **Auto-generated API docs** — The starlight-polyglot plugin must parse Python source (`src/innovate/`) and generate API reference pages automatically.
3. **Navigation** — Sidebar navigation must mirror the logical structure of the documentation: Getting Started, User Guide, API Reference, Maintainers, Operations, Architecture, Migration.
4. **Search** — Full-text search across all documentation pages.
5. **Versioning** — Support for multiple documentation versions via starlight-versions.

### Non-Functional

1. **Deployment** — Deploy to GitHub Pages on push to `main`.
2. **Link Validation** — All internal links must be validated in CI via starlight-links-validator.
3. **Performance** — Build time under 2 minutes, Lighthouse score > 90.
4. **Accessibility** — WCAG 2.1 AA compliance.
5. **Responsive Design** — Full support for mobile, tablet, and desktop viewports.

### Migration Strategy: Parallel-Run

Both Sphinx and Starlight sites will coexist during the migration. The Sphinx site remains the canonical source until migration is complete. A redirect layer ensures smooth transition.

## Architecture

```
docs/astro-site/
├── astro.config.mjs          # Astro/Starlight configuration
├── package.json              # Node.js dependencies
├── tsconfig.json             # TypeScript configuration
├── public/
│   └── .nojekyll             # GitHub Pages config
└── src/
    ├── content.config.ts     # Content collections
    ├── content/docs/         # Markdown/MDX documentation
    │   ├── index.md          # Landing page
    │   ├── user-guide/       # Getting started, tutorials
    │   ├── api/              # API reference
    │   ├── core/             # Core concepts
    │   ├── architecture/     # Architecture docs
    │   ├── operations/       # Operations docs
    │   ├── maintainers/      # Maintainer docs
    │   ├── migration/        # Migration resources
    │   └── tutorials/        # Tutorials
    └── styles/
        └── custom.css        # Custom styles
```

## Plugin Integration

### starlight-polyglot

- **Purpose**: Auto-generate API documentation from Python source code
- **Configuration**: `entryPoints: ['../../src/innovate']` relative to `docs/astro-site/`
- **Output**: Generated pages at `/api/python/`
- **Parser**: Reads Python module structure, docstrings, and type annotations

### starlight-versions

- **Purpose**: Multi-version documentation selector
- **Current**: Single version (`latest` → v0.5.x)

### starlight-links-validator

- **Purpose**: Validate all internal links during CI builds
- **Configuration**: Enabled in plugin array

## Deployment

- **Platform**: GitHub Pages
- **URL**: `https://edithatogo.github.io/innovate`
- **Base path**: `/innovate` (project site, not org site)
- **Trigger**: Push to `main` or `workflow_dispatch`
- **CI/CD**: GitHub Actions workflow at `.github/workflows/docs.yml`
