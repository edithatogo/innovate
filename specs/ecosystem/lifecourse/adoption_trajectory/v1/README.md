# Lifecourse Adoption-Trajectory Fixture

This directory contains the first documented ecosystem fixture for a future
`lifecourse` scenario adapter.

Files:

- `manifest.json`: versioned contract, schema, provenance, dependency policy,
  and producer/consumer responsibilities.
- `adoption_trajectory.parquet`: deterministic 12-row Arrow-compatible payload
  for scenario smoke tests.

The fixture is intentionally small. It proves that a downstream consumer can
inspect adoption trajectories through public Arrow or Parquet tooling without
importing `innovate` internals or adding `lifecourse` as a base dependency.

Runtime adapter implementation remains future work. Promotion beyond the
documented stage requires an optional extra, smoke CI, and a compatibility
matrix that names supported `innovate`, adapter, and `lifecourse` versions.
