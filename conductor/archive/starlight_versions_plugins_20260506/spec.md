# Specification

## Overview

Update the roadmap and technology stack documentation to include Starlight as the documentation platform under consideration, with explicit version pinning and a small approved plugin set.

The goal is to make the docs roadmap specific enough that future Starlight work can be implemented, tested, and reviewed without re-litigating the baseline package choices.

## Functional Requirements

1. Record the Starlight docs stack in the roadmap/stack documentation with an explicit versioning policy.
2. Capture the current target version of `@astrojs/starlight` and any approved companion plugins at implementation time.
3. Document the purpose of each approved plugin and why it is included.
4. Include at least the following plugin categories where they are justified by the docs roadmap:
   - versioned documentation
   - internal link validation
   - search integration
5. Treat the initial plugin shortlist as:
   - `starlight-versions` for versioned documentation navigation
   - `starlight-links-validator` for internal link validation
   - `@astrojs/starlight-docsearch` as the search-provider option if Algolia DocSearch is selected
6. Keep the roadmap wording aligned with the repo's Conductor model: roadmap items should be actionable, testable, and easy to archive when complete.

## Non-Functional Requirements

1. The change should stay documentation-focused.
2. Any version references must be easy to update when the ecosystem releases new compatible versions.
3. The docs should not over-specify plugins that are not needed for the current roadmap.
4. The roadmap should distinguish the core Starlight package from optional plugins and clearly call out any provider-dependent choices such as DocSearch.

## Acceptance Criteria

1. The roadmap and/or tech-stack documentation explicitly mention Starlight versions.
2. The approved plugin set is documented with a short rationale for each entry.
3. The documentation distinguishes required plugins from optional or future candidates.
4. The related Conductor track can be reviewed and archived cleanly.
5. The roadmap names the current Starlight version and the initial plugin shortlist explicitly.

## Out of Scope

1. Building the Starlight site itself.
2. Migrating existing docs content into Starlight.
3. Implementing plugin code in this track.
