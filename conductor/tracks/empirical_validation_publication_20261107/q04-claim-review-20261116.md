# Q04 capability-claim review lane

- Owner: Codex chat `01a1e3d8-fb04-7010-a038-2a5478662a5d`.
- Worktree: `/Users/doughnut/.codex/worktrees/innovate-q04-claim-review/innovate`.
- Branch: `codex/innovate-q04-claim-review`; base `57d2916586d107c8affe90355e453abe27f01c26`.
- Bounded issue: fail closed when the existing capability checker cannot prove the claimed native receipt and source revision. Allowed files: this lane handoff, `scripts/validate_capability_claims.py`, `tests/unit/test_capability_claims.py`, and `tests/validation/test_release_claims.py`.
- No native or external object scope. Shared capability manifests, receipt schemas, track plans, registries, release decisions and CI remain with the existing integrator.
- Next operation: hand off this reviewed scoped commit for integration and source-bound receipt-contract work; Q04 acceptance remains open.

## Finding and bounded correction

At the base revision, a fabricated `rust_native` capability with an arbitrary
revision and `/dev/null` as `native_receipt` returned no errors. The checked-in
matrix also uses `.rs` source or test files as native and scientific receipts.
Existence of those files does not show execution, source-bound acceptance or
scientific review.

The checker now verifies the Git checkout, requires an explicitly supplied
source commit that exists locally and matches the matrix, rejects missing,
absolute, escaping and symlink-escaping paths, rejects unsupported backends,
and reports all existing non-null receipt references as unqualified. A JSON
file with a self-declared pass is also insufficient. This is an interim
fail-closed guard: the current path-only matrix has no qualified mechanism to
bind receipt bytes, exact source and capability. Consequently this version
cannot qualify any native, scientific, empirical or release receipt. The
integrator must establish and qualify an evidence contract and update the
shared matrix before a claim can pass; this lane does not invent that schema
or claim Q04 or R10 complete.

## Evidence and review

- The first focused test run at base failed because the old checker accepted
  the source-file references as qualifying evidence. This was the red state.
- During review, an initial guard incorrectly required the recorded source
  commit to equal the newer checkout HEAD. A focused red test exposed that
  overconstraint; the correction permits a locally present historical source
  revision while still refusing unbound receipts.
- Repo-local `.venv/bin/python` reported CPython 3.15.2. pytest 9.2.0 and Ruff
  0.17.5 were installed under constraints exported from the unchanged lock.
- `uv run --locked --no-sync --python .venv/bin/python pytest tests/unit/test_capability_claims.py tests/validation/test_release_claims.py -q`:
  **12 passed** on macOS arm64. Cases include the prior fabrication, stale or
  unavailable checkout revision, missing files, absolute/traversing paths,
  symlink escape, invalid path input, unsupported backends and the currently
  unqualified checked-in matrix. The symlink case runs where symlink creation
  is supported; the recorded local run did not skip it.
- Ruff format and lint checks passed for the three changed Python files.
  `.venv/bin/python scripts/check_repository_policy.py` passed locally; live
  settings were not checked. `git diff --check` passed.
- `.venv/bin/python scripts/validate_capability_claims.py conductor/capabilities.json`
  exited **1 as expected** because no source revision was supplied and the
  four capabilities' native and scientific references are not qualified
  source-bound receipts. Supplying the matrix's historical source commit with
  `--source-revision 44cb96358b073cf899a3d5e2360456bc0e7c5ca9` still exited 1
  for the unqualified references. A historical source commit is not rejected
  merely because a later commit adds evidence; the missing receipt contract
  must establish source-to-artifact identity. These expected failures are open
  gates, not passing release checks.
- Reviewed SHA-256: validator
  `9af1ba34234df91cc823929891198e044e94d6fb5531783b48f5eaff78b60bc4`;
  unit tests `66604d2e36fd850b61689b75c245539e73c0ff48d3ea86e2be9a30daed7c038f`;
  release-claim tests `2f1617cc891a5f8c05edd50d962141f818b304cf41e9fa107bdeb5b39447b51f`.

The user-requested empirical-track review also found unsupported completion
marks and additional V01/V02/V03/V04 gaps, recorded in the separate M01 lane
handoff. The shared plan and registry require integrator reconciliation. No
full native suite, hosted CI, scientific acceptance, release, or archive is
claimed by this correction.
