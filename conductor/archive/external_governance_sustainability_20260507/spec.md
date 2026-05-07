# Specification

## Overview

Prepare governance and sustainability evidence needed for community review,
NumFOCUS-style evaluation, and long-term stewardship.

## Dependencies

- Feeds community submission readiness.
- Feeds JOSS and NumFOCUS dossiers.
- Feeds HPC community trust and maintainer continuity.

## Functional Requirements

1. Define maintainer roles and decision responsibilities.
2. Add or verify citation, security, support, funding, and roadmap ownership
   documents.
3. Add sustainability evidence for multi-language maintenance.
4. Add governance tests that keep the evidence discoverable.

## Parallelization

- Agent A owns maintainer and governance policy.
- Agent B owns security and support policy.
- Agent C owns citation and JOSS metadata.
- Agent D owns funding and sustainability evidence.
- Agent E owns contributor onboarding.
- Agent F owns final governance review and tests.

## Acceptance Criteria

1. Governance evidence is explicit and discoverable.
2. Submission dossiers can link to current maintainer and support policies.
3. Multi-language maintenance responsibilities are documented.
