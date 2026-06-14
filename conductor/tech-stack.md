# Tech Stack

## Documentation

- **Site Generator**: [Astro](https://astro.build) with [Starlight](https://starlight.astro.build)
- **Documentation Theme**: Starlight (Astro integration)
- **Versioned Baseline**: `@astrojs/starlight` `0.38.4` is the documented
  baseline from the Starlight Versions and Plugin Roadmap Alignment track.
- **Polyglot Plugin**: [starlight-polyglot](https://github.com/edithatogo/starlight-polyglot) — auto-generates API docs from Python source code
- **Versioning**: starlight-versions `0.9.0` is enabled with versioned content
  under `latest/`; the active build validates Astro 6 non-doc routes such as
  `/404`.
- **Link Validation**: starlight-links-validator `0.24.0` for CI link checking
- **Prose Linting**: Vale with the `Repo/ValueProse` style checks governance
  prose for hedging and filler wording in CI.
- **Search**: `@astrojs/starlight-docsearch` `0.7.0` when DocSearch is
  selected; `@astrojs/sitemap` is the sitemap integration baseline.
- **Deployment**: GitHub Pages via GitHub Actions
- **Migration Track**: Astro/Starlight Documentation Site Migration records
  cutover gates, route inventory, and redirect inventory.
- **Build Gate**: `pnpm build && pnpm check` passes for the active
  Astro/Starlight site with `starlight-polyglot` Python generation and
  `starlight-links-validator` enabled.

## Previous Documentation (Legacy)

- Legacy Sphinx source with `sphinx_rtd_theme` is retained as archival and
  redirect-reference material while Astro/Starlight is the active docs stack.
- **Extensions**: `autodoc`, `napoleon`, `viewcode`, `intersphinx`, `sphinx_autodoc_typehints`

## Runtime Components

### Language
- Python 3.10+ (primary)
- Rust (native kernel, bindings target)
- R, Julia, TypeScript, Go, C# (binding surfaces)

### Package Management
- **Python**: uv (primary), pip, conda. Python dependency management remains `uv`-first, with **nox** — Python task orchestration for repeatable local and CI sessions.
- **Node.js**: pnpm (docs site dependencies)
- **Version Sync**: `scripts/sync_versions.py` keeps package metadata aligned.

### Acceleration and Native Runtime

- **XLA-Backed Libraries**: JAX, NumPyro, BlackJAX, TensorFlow Probability, and
  Diffrax are the preferred XLA-backed options when accelerator evidence
  justifies promotion.
- **Rust Profiling**: **cargo-flamegraph** captures CPU hot paths and **DHAT**
  captures memory evidence for promoted native slices.
- **GPU Profiling**: JAX/XLA device profilers remain the GPU evidence source
  until Rust owns a promoted native GPU execution backend.
- **Rust Ownership Boundary**: the remaining ownership gap is tracked as archived closure evidence under `Rust Core Full Native Migration and Ownership Closure`; the earlier Rust Core Migration Completion and Polyglot Claim Closure track remains prior closure evidence.

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
