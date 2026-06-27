---
title: Runtime Logging
description: Structured logging configuration and observability for the innovate library.
---

# Runtime Logging

The innovate library uses structured logging to provide observability into model
fitting, simulation, and release-readiness operations. Structured log records
emit machine-readable JSON fields so that CI gates, release evidence, and
developer tooling can consume them deterministically.

## Structured Log Format

All runtime log records follow a structured schema:

- `timestamp` — ISO 8601 UTC timestamp
- `level` — `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `event` — short event identifier (e.g. `fit.complete`, `simulation.step`)
- `module` — source module name
- `message` — human-readable summary
- `context` — optional key-value payload (parameters, residuals, diagnostics)

## Configuration

Structured logging is configured via the standard `logging` module. In CI and
release lanes, logs are emitted as JSON lines so that evidence artifacts can
be collected and validated.

## Release Evidence Integration

Structured log output from release-readiness nox sessions (`coverage`,
`mutation`, `release_supply_chain`, `release_reproducibility`,
`release_readiness`) is captured as evidence artifacts under
`docs/source/_static/release_readiness/evidence/`. Each artifact includes a
`generated_at` timestamp so that the release-readiness evaluator can enforce
freshness gates.
