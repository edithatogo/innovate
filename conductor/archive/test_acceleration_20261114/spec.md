# Specification

## Problem and scope

[Issue #473](https://github.com/edithatogo/innovate/issues/473) requests measured,
bounded local feedback profiles. Its August Python-first inventory predates the
Rust-native migration. Python 3.15.2, the six current `tests/native` files and the
fresh full native/installed-consumer CI suites are the starting qualification
surface. Selective results must never imply Rust, scientific or release acceptance.

## Requirements

1. Provide an opt-in serial testmon runner, separate from default pytest/Nox/CI.
   Conservatively rebuild its local database after dependency, interpreter, test
   inventory, native code/binary, fixture or subprocess-helper changes. Reject
   concurrent cache writers and incompatible inherited pytest/coverage options.
2. Record actual wall time, collected/selected/executed/deselected counts, failures,
   tool versions, scope and cache state. An entirely deselected warm run is no new
   verification. Empty inventories, errors and failed cold runs cannot seed reuse.
3. Trial pinned pytest-gremlins on `scripts/check_migration.py` with ordinary
   normal-import policy tests, two operator families and explicit subprocess time
   budgets. Report surviving, timed-out, errored and skipped mutations honestly.
4. Compare Coverage's ctrace and sysmon on the same serial policy scope. Verify
   the actual core and compare line/branch evidence. Keep testmon, mutation,
   coverage plugins, dynamic contexts and concurrent execution separate.
5. Document why VCR.py, Freezegun and pytest-picked do not fit the current native
   boundary. Do not record live HTTP or restricted payloads.
6. Preserve default full clean-environment CI, thresholds, all 38 scientific
   obligations, rights, security and publication gates.

## Allowed changes and dependencies

New local profile tooling under `scripts/test_acceleration/`, focused tests under
`tests/native/`, one maintainer documentation page and this Conductor track.
Optional tooling pins are isolated from the production manifest and lockfile.
The registry gains only this track's entry. No shared workflow, native production
code, scientific tolerances, root dependency pins or release rules change.

## Acceptance

Execute real before/after runs on the supported interpreter; qualify cache
invalidation and failure handling with synthetic repositories. Review independently,
run ordinary affected and full available gates, and record exact-source evidence
and any remaining integration blockers. No speedup is assumed before measurement.
