# Astro 7 and Starlight 0.40 Dedicated Migration Plan

## Phase 1: Baseline Audit and Target Contract

- [ ] Task: Audit current docs frontend dependency surface
    - [ ] Read `docs/astro-site/package.json`, `docs/astro-site/pnpm-lock.yaml`, `astro.config.mjs`, `starlight.config.mjs`, docs workflows, and existing Astro/Starlight evidence artifacts.
    - [ ] Record the current versions and plugin relationships for Astro, Starlight, markdown processing, link validation, versioning, DocSearch, polyglot generation, TypeScript, and pnpm.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Define the Starlight 0.40 target decision
    - [ ] Write a target decision artifact that either selects Starlight 0.40.x or explicitly promotes the currently committed successor line with evidence.
    - [ ] Document external compatibility constraints such as peer metadata exceptions without weakening the selected target.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Write failing dependency-contract tests
    - [ ] Add or update tests that compare documented Astro/Starlight baseline, package manifest, lockfile, and migration evidence.
    - [ ] Confirm the tests fail before implementation if the target contract is absent or inconsistent.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Baseline Audit and Target Contract' (Protocol in workflow.md)

## Phase 2: Dependency Migration and Plugin Compatibility

- [ ] Task: Apply the selected Astro 7/Starlight target
    - [ ] Update `docs/astro-site/package.json` and `pnpm-lock.yaml` to the selected Astro 7/Starlight baseline.
    - [ ] Update `@astrojs/markdown-remark`, `starlight-links-validator`, `starlight-versions`, `@astrojs/starlight-docsearch`, `@astrojs/check`, TypeScript, and local plugin wiring as needed.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Update Astro and Starlight configuration
    - [ ] Validate `astro.config.mjs` uses the Astro 7-compatible markdown processor and plugin ordering.
    - [ ] Remove stale or duplicate Starlight config paths unless retained as explicit legacy reference material.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Prove plugin compatibility
    - [ ] Run `pnpm --dir docs/astro-site install --lockfile-only` if dependency metadata changed.
    - [ ] Run `pnpm --dir docs/astro-site check`.
    - [ ] Record peer-dependency exceptions or plugin compatibility workarounds in migration evidence.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Dependency Migration and Plugin Compatibility' (Protocol in workflow.md)

## Phase 3: Route, Content, and UX Cutover Validation

- [ ] Task: Validate route inventory and versioned content
    - [ ] Verify core routes, maintainer routes, operations routes, architecture routes, migration routes, versioned `latest/` routes, and `/404`.
    - [ ] Update route inventory evidence and tests for all tracked route groups.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Validate generated API docs and legacy source boundaries
    - [ ] Run Starlight polyglot generation and confirm generated Python API pages are present.
    - [ ] Ensure Sphinx remains legacy/archive-only unless required for redirect-reference validation.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Validate frontend/docs UX stability
    - [ ] Confirm sidebar structure, custom CSS, version switcher behavior, link validation, and DocSearch gating.
    - [ ] Add or update tests/evidence for navigation and content parity with the prior docs surface.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Route, Content, and UX Cutover Validation' (Protocol in workflow.md)

## Phase 4: CI, Release Evidence, and Documentation

- [ ] Task: Update CI and nox docs gates
    - [ ] Ensure GitHub Actions and `nox` docs sessions run the selected Astro/Starlight validation commands.
    - [ ] Verify docs workflows use the intended Node/pnpm setup and Python 3.14 polyglot context.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Refresh release-readiness evidence
    - [ ] Regenerate docs-build and compatibility evidence after Astro/Starlight validation.
    - [ ] Update release-readiness report without marking unrelated missing evidence as complete.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Update maintainer-facing migration documentation
    - [ ] Document the selected target, validation commands, plugin exceptions, rollback notes, and future upgrade rules.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: CI, Release Evidence, and Documentation' (Protocol in workflow.md)

## Phase 5: Final Review, CI Gate, and Archival Readiness

- [ ] Task: Run full docs and targeted regression validation
    - [ ] Run `pnpm --dir docs/astro-site check`.
    - [ ] Run `pnpm --dir docs/astro-site build`.
    - [ ] Run `uv run nox -s docs`.
    - [ ] Run targeted unit tests for Astro/Starlight migration and release-readiness contracts.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Run final Conductor review and apply findings
    - [ ] Invoke `conductor-review` for the full track.
    - [ ] Apply review fixes, rerun validation, and commit any review changes.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Push and monitor GitHub Actions
    - [ ] Push the completed track branch.
    - [ ] Monitor GitHub Actions until all triggered checks pass or a documented external blocker is reached.
    - [ ] Commit any CI-fix changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Final Review, CI Gate, and Archival Readiness' (Protocol in workflow.md)
