# Implementation Plan: HEOR Process Mining Fixture Contract and Interface Decision

## Phase 1: Documented Fixture Contract

- [x] Task: Add failing process-mining fixture tests
    - [x] Validate manifest schema/versioning and dependency policy.
    - [x] Validate event-log, pathway, conformance, and bottleneck payloads.
    - [x] Validate CLI/MCP decisions and docs links.
- [x] Task: Add versioned process-mining fixture bundle
    - [x] Add manifest and deterministic CSV/JSON payloads.
    - [x] Keep PM4Py reference-only and exclude private Python object framing.
- [x] Task: Update ecosystem documentation
    - [x] Link process fixture from ecosystem specs.
    - [x] Record CLI-first and MCP-deferred decisions.
- [x] Task: Validate focused tests
    - [x] Run `uv run pytest tests/unit/test_process_mining_fixture_contract.py -q`.
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Documented Fixture Contract' (Protocol in workflow.md)
