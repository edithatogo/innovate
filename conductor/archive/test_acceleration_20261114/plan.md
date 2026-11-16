# Plan

- [x] Reconcile issue, main, open PRs, current native architecture and CI ownership.
- [x] Implement isolated serial profiles and regression tests for invalidation,
  concurrency, real test accounting and fail-closed results.
- [x] Trial bounded migration-policy mutation with normal imports; retain all outcomes.
- [x] Measure baseline/cold/warm/changed testmon runs and ctrace/sysmon parity.
- [x] Document applicability, exclusions, reproduction commands and limitations.
- [x] Run independent review and ordinary verification. Fix interrupted process
  cleanup, stale success stamps, skip-only results and Gremlins report isolation.
- [!] Local full native build: both locked release and reduced-memory dev builds
  were killed with SIGKILL. Do not claim complete native or clean-package execution.
- [~] Prepare focused PR and inspect exact-head hosted gates before integration.

All broader scientific/release gates remain open. This track does not authorize
changes to other active tracks or adoption of an unqualified native runtime.

See [qualification.md](qualification.md) and [evidence.json](evidence.json) for
commands, source hashes, actual counts, timing samples and remaining gates.
