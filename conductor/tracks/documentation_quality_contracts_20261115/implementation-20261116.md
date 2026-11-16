# Documentation implementation — 16 November 2026

## Authority and baseline

Dylan authorised implementation of the learning programme, stronger docstrings and
language-native documentation conventions/contracts, removal of obsolete content,
and alignment with the open-science communities named in GitHub issues.

Inspected main: `6c226f92f14f01e771b1490c7fb3a7c77f4bc74a`.
Existing planning PR #659: `76f455fb568639cccb2370720eebcce399b2e158`.
The implementation uses an isolated worktree based on current main and retains
the published planning packet. Main differs from that packet's audit base only
by the inspected Renovate update. PR #656 remains open and proposes an archive
move; preserve active documentation obligations during its integration.

## Work in progress

- [~] Strengthen source documentation contracts for Python, Rust, Julia and R;
  improve conventions and examples using the existing generators/checkers.
- [~] Add source-derived Astro model/tutorial discovery with typed metadata.
- [~] Remove obsolete public documentation/notebooks with a versioned removal
  inventory and current migration routes; preserve historical acceptance receipts.
- [~] Qualify optional marimo execution against the intended native environment;
  record actual compatibility evidence and implement eligible notebook work.
- [~] Map current community issue requirements to concrete documentation evidence
  and explicit unfulfilled submission/registration requirements.
- [ ] Execute affected source, contract, generator and browser checks; reconcile
  exact revisions and independent review before publishing to the existing PR.

## Scope and ownership

The authorised documentation increment covers binding docstrings/comments and
manuals, documentation source/contracts/generators and their focused tests,
examples/notebook tooling, obsolete documentation/notebooks, and this track.
Rust scientific algorithms, shared CI/root dependencies and unrelated integration
remain with their existing owners. Needed dependency/workflow changes are
recorded explicitly rather than silently altering native or release boundaries.

Independent internal work is divided into nonoverlapping source-documentation,
Astro catalogue and issue/standards review tasks. The root integrates their changes
in this worktree and owns content retirement, notebook qualification and evidence.

## Verification record

Initial workspace has sufficient capacity (56 GiB free). Available default tools
report Python 3.12.14 and Node 24.22.1; neither is the repository's specified full
qualification environment. Rust, Julia and R were not on PATH at intake. Actual
installation attempts and the resulting limits are recorded below as work proceeds.

Source-only validation on the available interpreter must be labelled separately
from pinned installed-native, language-runtime, browser and scientific acceptance.
