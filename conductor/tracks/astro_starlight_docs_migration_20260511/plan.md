# Implementation Plan: Astro/Starlight Documentation Site Migration

## Phase 1: Confirm the Migration Baseline

- [x] Task: Inventory the current docs site and migration surfaces [checkpoint: 85730d1]
    - [x] Identify the canonical Sphinx pages that must remain reachable
    - [x] Map the docs sections that will move to Astro/Starlight first
    - [x] Record the required redirect or forwarder behavior
    - [x] Classify each page as migrate, redirect, or archive-only
- [x] Task: Decide the migration mode and search provider [checkpoint: 85730d1]
    - [x] Choose parallel-run or full cutover before content migration starts
    - [x] Choose Algolia DocSearch or record the explicit alternative provider
    - [x] Decide whether `@astrojs/sitemap` is part of the baseline or an equivalent official Astro integration is used
- [x] Task: Confirm the versioned Starlight baseline [checkpoint: 85730d1]
    - [x] Pin `@astrojs/starlight` `0.38.4`
    - [x] Pin `starlight-versions` `0.5.4`
    - [x] Pin `starlight-links-validator` `0.18.0`
    - [x] Pin `@astrojs/starlight-docsearch` `0.6.1` if Algolia DocSearch is the chosen provider
- [x] Task: Write regression tests for the documented baseline [checkpoint: 85730d1]
    - [x] Require the tech stack and migration docs to mention the same versions
    - [x] Require the plugin shortlist to distinguish required versus optional pieces
    - [x] Require the migration plan to mention redirects, route inventory, and link checking
    - [x] Require a redirect inventory to match the content inventory
- [x] Task: Conductor - Automated Review and Checkpoint 'Confirm the Migration Baseline' (Protocol in workflow.md) [checkpoint: 85730d1]

## Phase 2: Scaffold the Astro/Starlight Site

- [x] Task: Create the initial Astro/Starlight project structure [checkpoint: 85730d1]
    - [x] Add the Astro config and Starlight content structure
    - [x] Add versioned navigation support
    - [x] Add link validation and search integration hooks
    - [x] Add sitemap generation if selected
- [x] Task: Create the machine-readable migration manifest [checkpoint: 85730d1]
    - [x] Record the transition mode, search decision, and sitemap decision
    - [x] Record the first-move pages, holdouts, and redirect targets
    - [x] Keep the manifest synchronized with the content and redirect inventories
- [x] Task: Add docs-site metadata and navigation [checkpoint: 85730d1]
    - [x] Define the site title, sidebar, and canonical sections
    - [x] Preserve the current docs ownership map in the new layout
    - [x] Keep the migration scaffolding explicit in the docs site
- [x] Task: Conductor - Automated Review and Checkpoint 'Scaffold the Astro/Starlight Site' (Protocol in workflow.md) [checkpoint: 85730d1]

## Phase 3: Migrate Content and Redirects

- [~] Task: Move canonical docs content into the Astro/Starlight structure
    - [~] Migrate the core contract and roadmap pages first
    - [ ] Migrate binding and release documentation
    - [ ] Migrate archive and migration reference pages needed by readers
- [ ] Task: Maintain the redirect inventory during cutover
    - [ ] Map each moved Sphinx path to a route or redirect
    - [ ] Keep the redirect inventory synchronized with the content inventory
    - [ ] Verify canonical URLs remain reachable throughout the transition
- [ ] Task: Preserve link stability during the cutover
    - [ ] Add redirects or forwarders for old Sphinx URLs
    - [ ] Keep versioned content routes stable
    - [ ] Verify internal links after migration
- [ ] Task: Conductor - Automated Review and Checkpoint 'Migrate Content and Redirects' (Protocol in workflow.md)

## Phase 4: Validate the New Docs Site

- [x] Task: Add and update tests for the Astro/Starlight migration [checkpoint: 2434e3d]
    - [x] Verify the chosen plugin baseline remains documented
    - [x] Verify the sitemap and search decisions remain explicit
    - [x] Verify redirect coverage for the canonical docs paths
    - [x] Verify route stability for the versioned docs nav
- [x] Task: Run the docs build and link validation [checkpoint: 2434e3d]
    - [x] Build the Astro/Starlight site
    - [x] Run the link validator
    - [x] Confirm the docs remain auditable from the repo
- [x] Task: Validate the transition policy and cutover inventory [checkpoint: 2434e3d]
    - [x] Confirm the chosen migration mode is still reflected in the docs
    - [x] Confirm the content inventory and redirect inventory still agree
    - [x] Confirm sitemap generation is either configured or explicitly deferred with rationale
- [x] Task: Conductor - Automated Review and Checkpoint 'Validate the New Docs Site' (Protocol in workflow.md) [checkpoint: 2434e3d]

## Phase 5: Final Review and Archive

- [ ] Task: Run final conductor review
    - [ ] Review the track diff against the spec, plan, workflow, and tests
    - [ ] Apply any high-confidence fixes surfaced by review
    - [ ] Re-run validation until stable
- [ ] Task: Archive the completed migration track
    - [ ] Move the track folder to the archive location
    - [ ] Update the tracks registry entry to completed
    - [ ] Preserve links to the migration baseline and build evidence
- [ ] Task: Conductor - Automated Review and Checkpoint 'Final Review and Archive' (Protocol in workflow.md)
