# Contributing to Innovate

The 0.7 development line moves scientific computation into Rust, tables into
Polars, and exposes only Python, Julia and R bindings. Read README.md and
migration-ledger.json before adding a capability. All existing and planned
features remain required; a temporarily unavailable feature is not a retired requirement.

Read [AGENTS.md](AGENTS.md) for execution and ownership rules,
[SECURITY.md](SECURITY.md) for vulnerability reporting, and the
[Conductor index](conductor/index.md) for project context. Resolve the owning
track through [the track registry](conductor/tracks.md), then read its specification
and plan before implementation. Historical completion labels apply to their
recorded scope; they do not establish current release acceptance.

## Environment

Use Rust 1.99.1 and an explicitly selected repo-local Python 3.15.2 environment:

```sh
uv venv .venv-native --python 3.15.2
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-native"
uv sync --locked --python 3.15.2
.venv-native/bin/python -c 'import sys; print(sys.version, sys.prefix)'
```

Keep other writers, environments and in-flight native operations intact. Use a
separate branch/worktree for parallel work. The repository integrator owns shared
promotion and hook installation; existing hooks remain enabled. The local hook
runner checks the selected interpreter's actual version before invoking tools.

## Development checks

From the repository root, run the dependency-free repository-policy check:

```sh
python scripts/check_repository_policy.py
```

Its result covers repository policy and drift within the reported scope. Live
GitHub settings, bot operation and external coverage delivery require their own
current evidence. This command does not run the scientific implementation or
replace required CI checks.

After installing the development environment, the full development validation
command is:

```sh
.venv-native/bin/nox -s lint types tests native docs package
```

Keep migration and version consistency checks explicit:

```sh
.venv-native/bin/python scripts/check_migration.py
.venv-native/bin/python scripts/sync_versions.py --check
```

Coverage, security, clean-package/platform and release gates remain separate.
`scripts/check_migration.py --release` must continue to fail while mandatory
migration obligations are unresolved; a passing development run does not override it.

During edits, select affected pytest files/nodes and Cargo tests. Keep pytest's
cache and use `--lf --last-failed-no-failures=none` only for rerunning known
failures. No selected tests means no new verification. Do not add global retries
or skips. Run clean-package, installed-consumer and integration gates at coherent
milestones. The historical Python suite is preserved in Git and its path/hash
inventory; current native tests do not replace its feature acceptance obligations.

For the documentation site, use Node 26 and pnpm 12.18.0. Set
`STARLIGHT_POLYGLOT_PYTHON` to the selected environment's Python executable.
Keep the lockfile, package release-age policy and recorded compatibility exceptions.

## Scientific changes and review

The repository supports one maintainer: zero mandatory human approvals, code-owner
approvals or team gates. Automated review and required checks on the exact proposed
commit control routine merges. CODEOWNERS is routing information only. Preserve
owner recovery without using it to waive failed checks, and coordinate shared-file
changes and promotion with the integrator.

Rust owns validation, numeric algorithms and capability reporting. Bindings adapt
transport only. Tests should use independent analytic/reference oracles and cover
failure boundaries. Unavailable methods must fail explicitly. Document parameter,
time, output and uncertainty semantics, not just successful calls.

Project-owned code uses Apache-2.0; preserve third-party licences and notices.
Follow CODE_OF_CONDUCT.md. Report bugs with the model, schema, environment, minimal
input and expected behavior, excluding private data and credentials. Use scoped
commits and cite retained evidence in the Conductor track. The migration release
gate must remain closed until every required feature and distribution gate passes.
