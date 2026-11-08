# Contributing to Innovate

The 0.7 development line moves scientific computation into Rust, tables into
Polars, and exposes only Python, Julia and R bindings. Read README.md and
migration-ledger.json before adding a capability. All existing and planned
features remain required; a temporarily unavailable feature is not a retired requirement.

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

```sh
.venv-native/bin/nox -s lint types tests native
.venv-native/bin/python scripts/check_migration.py
.venv-native/bin/python scripts/sync_versions.py --check
```

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

Rust owns validation, numeric algorithms and capability reporting. Bindings adapt
transport only. Tests should use independent analytic/reference oracles and cover
failure boundaries. Unavailable methods must fail explicitly. Document parameter,
time, output and uncertainty semantics, not just successful calls.

Project-owned code uses Apache-2.0; preserve third-party licences and notices.
Follow CODE_OF_CONDUCT.md. Report bugs with the model, schema, environment, minimal
input and expected behavior, excluding private data and credentials. Use scoped
commits and cite retained evidence in the Conductor track. The migration release
gate must remain closed until every required feature and distribution gate passes.
