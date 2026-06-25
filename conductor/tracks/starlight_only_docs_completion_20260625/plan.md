# Starlight-Only Documentation Completion Plan

## Phase 1: Remaining RST Audit

- [ ] Task: Classify remaining RST files
    - [ ] Group files as tutorial, bridge, generated API, template, archive, or evidence.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Write fail-closed active-RST tests
    - [ ] Add an allowlist-based guard for any retained RST file.
    - [ ] Confirm tests fail for unclassified active RST files.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Remaining RST Audit' (Protocol in workflow.md)

## Phase 2: Tutorial and Bridge Migration

- [ ] Task: Migrate core contract tutorials
    - [ ] Promote functional kernel, diagnostics, and Arrow tutorials to Starlight current/latest pages.
    - [ ] Repoint tests and evidence.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Migrate modeling and integration tutorials
    - [ ] Promote Norton-Bass, advanced diffusion, Bayesian fitter, multi-product, seasonal, counterfactual, JAX, and NDLib docs.
    - [ ] Add route/sidebar entries where needed.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Tutorial and Bridge Migration' (Protocol in workflow.md)

## Phase 3: API and Evidence Parity

- [ ] Task: Replace generated API bridge dependency
    - [ ] Extend Starlight/polyglot or static inventories to cover module API parity.
    - [ ] Remove generated RST and autosummary templates when parity is proven.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Refresh route and migration evidence
    - [ ] Regenerate content inventory, redirect inventory, route coverage, and validation evidence.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: API and Evidence Parity' (Protocol in workflow.md)

## Phase 4: Final Docs Release Gate

- [ ] Task: Run full docs validation
    - [ ] Run `pnpm --dir docs/astro-site check`.
    - [ ] Run `uv run nox -s docs`.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Run final review, push, and CI monitor
    - [ ] Run conductor-review for the full track, apply findings, push, and monitor GitHub Actions.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Docs Release Gate' (Protocol in workflow.md)
