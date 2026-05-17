# Tech Stack

## Documentation

- **Site Generator**: [Astro](https://astro.build) with [Starlight](https://starlight.astro.build)
- **Documentation Theme**: Starlight (Astro integration)
- **Polyglot Plugin**: [starlight-polyglot](https://github.com/edithatogo/starlight-polyglot) — auto-generates API docs from Python source code
- **Versioning**: starlight-versions for multi-version documentation
- **Link Validation**: starlight-links-validator for CI link checking
- **Deployment**: GitHub Pages via GitHub Actions

## Previous Documentation (Legacy)

- **Sphinx** with `sphinx_rtd_theme` — being migrated to Starlight
- **Extensions**: `autodoc`, `napoleon`, `viewcode`, `intersphinx`, `sphinx_autodoc_typehints`

## Runtime Components

### Language
- Python 3.10+ (primary)
- Rust (native kernel, bindings target)
- R, Julia, TypeScript, Go, C# (binding surfaces)

### Package Management
- **Python**: uv (primary), pip, conda
- **Node.js**: pnpm (docs site dependencies)

### CI/CD
- GitHub Actions (CI, docs deployment, package publishing)
- pnpm/action-setup (Node.js dependency management)

## Repository Structure

```
innovate/
├── docs/
│   ├── astro-site/          # Starlight documentation site
│   │   ├── src/content/docs/ # Markdown/MDX documentation
│   │   ├── astro.config.mjs  # Astro configuration
│   │   └── public/           # Static assets
│   ├── conf.py              # Legacy Sphinx config
│   └── source/              # Legacy Sphinx source
├── src/innovate/            # Python library source
├── conductor/               # Migration tracks and tech stack
└── .github/workflows/       # CI/CD workflows
```
