# Starlight-Only Documentation Completion Plan

## Phase 1: Remaining RST Audit [checkpoint: c9a4cf4]

- [x] Task: Classify remaining RST files
    - [x] Group files as tutorial, bridge, generated API, template, archive, or evidence.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Write fail-closed active-RST tests [53ebe96]
    - [x] Add an allowlist-based guard for any retained RST file.
    - [x] Confirm tests fail for unclassified active RST files.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Remaining RST Audit' (Protocol in workflow.md)

## Phase 2: Tutorial and Bridge Migration [checkpoint: b4123d3]

- [x] Task: Migrate core contract tutorials [706084d]
    - [x] Promote functional kernel, diagnostics, and Arrow tutorials to Starlight current/latest pages.
    - [x] Repoint tests and evidence.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Migrate modeling and integration tutorials [90eb871]
    - [x] Promote Norton-Bass, advanced diffusion, Bayesian fitter, multi-product, seasonal, counterfactual, JAX, and NDLib docs.
    - [x] Add route/sidebar entries where needed.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Tutorial and Bridge Migration' (Protocol in workflow.md)

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
