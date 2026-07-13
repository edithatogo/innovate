# Operational Modeling Fixtures

This directory contains small deterministic ecosystem fixtures for
TreeAge-style decision analysis and discrete-event simulation (DES) workflows.
They are portable contract examples for sibling HEOR modules, not runtime
simulation engines inside `innovate`.

Fixtures:

- [treeage_style/manifest.json](./treeage_style/manifest.json): decision-tree
  and state-transition metadata for reimbursement examples.
- [des/manifest.json](./des/manifest.json): event-log, queue-metric, resource,
  pathway-state, and run metadata for DES pathway examples.

These fixtures are artifact-first. They do not require TreeAge, simply, PM4Py,
or private engine state. Runtime adapters must stay behind explicit optional
extras and follow the adapter promotion ladder before becoming experimental or
supported integrations.

Every operational-modeling fixture should record XLA eligibility or rejection
notes. Bounded state-transition matrices may be XLA-eligible; classic DES event
queues should be rejected when forcing them into XLA would distort dynamic event
semantics.
