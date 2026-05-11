# Implementation Plan: Astro/Starlight Documentation Site Migration

## Phase 1: Confirm the Migration Baseline

- [ ] Task: Inventory the current docs site and migration surfaces
    - [ ] Identify the canonical Sphinx pages that must remain reachable
    - [ ] Map the docs sections that will move to Astro/Starlight first
    - [ ] Record the required redirect or forwarder behavior
- [ ] Task: Confirm the versioned Starlight baseline
    - [ ] Pin `@astrojs/starlight` `0.38.4`
    - [ ] Pin `starlight-versions` `0.5.4`
    - [ ] Pin `starlight-links-validator` `0.18.0`
    - [ ] Decide whether `@astrojs/starlight-docsearch` `0.6.1` is the chosen search provider
    - [ ] Decide whether `@astrojs/sitemap` is included for public indexing
- [ ] Task: Write regression tests for the documented baseline
    - [ ] Require the tech stack and migration docs to mention the same versions
    - [ ] Require the plugin shortlist to distinguish required versus optional pieces
    - [ ] Require the migration plan to mention redirects and link checking
- [ ] Task: Conductor - Automated Review and Checkpoint 'Confirm the Migration Baseline' (Protocol in workflow.md)

## Phase 2: Scaffold the Astro/Starlight Site

- [ ] Task: Create the initial Astro/Starlight project structure
    - [ ] Add the Astro config and Starlight content structure
    - [ ] Add versioned navigation support
    - [ ] Add link validation and search integration hooks
    - [ ] Add sitemap generation if selected
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
- [ ] Task: Run the docs build and link validation
    - [ ] Build the Astro/Starlight site
    - [ ] Run the link validator
    - [ ] Confirm the docs remain auditable from the repo
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
