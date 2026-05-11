# Implementation Plan: Astro/Starlight Documentation Site Migration

## Phase 1: Confirm the Migration Baseline

- [~] Task: Inventory the current docs site and migration surfaces
    - [ ] Identify the canonical Sphinx pages that must remain reachable
    - [ ] Map the docs sections that will move to Astro/Starlight first
    - [ ] Record the required redirect or forwarder behavior
    - [ ] Classify each page as migrate, redirect, or archive-only
- [ ] Task: Decide the migration mode and search provider
    - [ ] Choose parallel-run or full cutover before content migration starts
    - [ ] Choose Algolia DocSearch or record the explicit alternative provider
    - [ ] Decide whether `@astrojs/sitemap` is part of the baseline or an equivalent official Astro integration is used
- [ ] Task: Confirm the versioned Starlight baseline
    - [ ] Pin `@astrojs/starlight` `0.38.4`
    - [ ] Pin `starlight-versions` `0.5.4`
    - [ ] Pin `starlight-links-validator` `0.18.0`
    - [ ] Pin `@astrojs/starlight-docsearch` `0.6.1` if Algolia DocSearch is the chosen provider
- [ ] Task: Write regression tests for the documented baseline
    - [ ] Require the tech stack and migration docs to mention the same versions
    - [ ] Require the plugin shortlist to distinguish required versus optional pieces
    - [ ] Require the migration plan to mention redirects, route inventory, and link checking
    - [ ] Require a redirect inventory to match the content inventory
- [ ] Task: Conductor - Automated Review and Checkpoint 'Confirm the Migration Baseline' (Protocol in workflow.md)

## Phase 2: Scaffold the Astro/Starlight Site

- [ ] Task: Create the initial Astro/Starlight project structure
    - [ ] Add the Astro config and Starlight content structure
    - [ ] Add versioned navigation support
    - [ ] Add link validation and search integration hooks
    - [ ] Add sitemap generation if selected
- [ ] Task: Create the machine-readable migration manifest
    - [ ] Record the transition mode, search decision, and sitemap decision
    - [ ] Record the first-move pages, holdouts, and redirect targets
    - [ ] Keep the manifest synchronized with the content and redirect inventories
- [ ] Task: Add docs-site metadata and navigation
    - [ ] Define the site title, sidebar, and canonical sections
    - [ ] Preserve the current docs ownership map in the new layout
    - [ ] Keep the migration scaffolding explicit in the docs site
- [ ] Task: Conductor - Automated Review and Checkpoint 'Scaffold the Astro/Starlight Site' (Protocol in workflow.md)

## Phase 3: Migrate Content and Redirects

- [ ] Task: Move canonical docs content into the Astro/Starlight structure
    - [ ] Migrate the core contract and roadmap pages first
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

- [ ] Task: Add and update tests for the Astro/Starlight migration
    - [ ] Verify the chosen plugin baseline remains documented
    - [ ] Verify the sitemap and search decisions remain explicit
    - [ ] Verify redirect coverage for the canonical docs paths
    - [ ] Verify route stability for the versioned docs nav
- [ ] Task: Run the docs build and link validation
    - [ ] Build the Astro/Starlight site
    - [ ] Run the link validator
    - [ ] Confirm the docs remain auditable from the repo
- [ ] Task: Validate the transition policy and cutover inventory
    - [ ] Confirm the chosen migration mode is still reflected in the docs
    - [ ] Confirm the content inventory and redirect inventory still agree
    - [ ] Confirm sitemap generation is either configured or explicitly deferred with rationale
- [ ] Task: Conductor - Automated Review and Checkpoint 'Validate the New Docs Site' (Protocol in workflow.md)

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
