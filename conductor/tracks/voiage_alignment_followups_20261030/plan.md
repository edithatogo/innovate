# Plan

## Bounded issue acceptance

- [x] Inventory pandas/xarray use and the Polars/PyArrow boundary; preserve the optional Arrow extra.
- [x] Implement and qualify normalized IPC/Parquet bundle consumption, digest verification, multi-chunk and normalized-schema round trips.
- [x] Record current Python/Rust targets and the retained basedpyright/Presidio exceptions; ty is not represented as executed.
- [x] Inspect the authenticated Dependabot inventory: zero open alerts; no dismissals or unreviewed closed-alert dispositions.
- [x] Verify a real Renovate bot commit and its actual protected contexts (PR #612; signing receipt).
- [x] Reconcile installed native artifact evidence and all seven required contexts on PR #605's actual final head and receiving tree.
- [x] Owner authorisation: Dylan requested implementation of #559 and continuation of its remaining acceptance work.
- [x] Complete independent review of the merged source, restored raw consumer receipt and current acceptance wording; apply the invocation/import-path precision correction.
- [~] Restore the omitted historical evidence and publish the current acceptance record; the resulting record-restoration PR needs its own protected checks.
- [ ] Existing integration owner completes promotion and track archival/shared-registry reconciliation after those checks pass.

See [current acceptance record](acceptance-20261115.md) for the exact revisions,
runs, artifact hashes, limitations and independently inspected observations.
The original [source audit](alignment-audit-20261114.md),
[qualification checkpoint](qualification-20261114.md) and restored
[hosted continuation](continuation-20261115.md) remain dated historical evidence.

## Source and integration checkpoints

PR #605's final head `5bfafaede6bb1001463480cc16a429cee052eb33` passed all seven
required contexts and merged as `eeac954a5fa335e00632f0aec4f3cbb94f75b656` with
an identical tree. Native consumer runs on that head, receiving main,
`f6c7d058` and `384ed3b0` passed. The `f6c7d058` archive was independently
downloaded, hashed and parsed; the later `384ed3b0` archive was subsequently
verified in the same way after executor access returned. Its earlier API-only
snapshot remains explicitly time-bounded.

Real Renovate head `59438016` passed all seven required contexts and merged
as `663002b03ccb92abee2ece934d69116cb4bf3415`. Its signed receiving tree equals
the verified bot head; the one-line validator patch does not change the
qualified consumer source.

Six continuation/evidence files from `4ddd0c7` were omitted during integration.
Restore their exact blobs and retain separate current receipts. Preserve
#606's central Renovate configuration and #611's independently owned changes.

The scoped issue criteria are evidenced; completion of the record proposal's
hosted checks and coordinator archival is still recorded as pending. Successful
earlier runs are not claimed for a later untested head. This track remains
inside the existing scientific ecosystem programme; the broader 38 scientific
obligations, release restrictions and offline security draft #590 are unchanged.
