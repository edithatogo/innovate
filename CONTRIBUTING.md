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

## Game capability intake

UOGTO owns game-theoretic semantics and Paignia owns reusable game
representations and solvers. Every newly identified, added or materially changed
game must have linked records under both upstream contracts:

- [UOGTO semantic coverage and extension intake](https://github.com/edithatogo/UOGTO/blob/main/docs/contracts/downstream-game-intake.md).
- [Paignia generic capability and fixture intake](https://github.com/edithatogo/paignia/blob/main/docs/integrations/downstream-game-intake.md).

Use the same intake key in both records and link them from the local plan or
pull request. Identify the originating source path and immutable revision,
players, actions, information, timing, payoff meaning and requested solution.
Record the semantic and implementation dispositions separately:

| Upstream | Allowed dispositions |
| --- | --- |
| UOGTO | `reuse_existing`, `extension_required`, `unsupported`, `mapping_review_pending` |
| Paignia | `reuse_existing`, `extension_required`, `unsupported`, `domain_only`, `assessment_pending` |

For `reuse_existing`, add or refine the UOGTO mapping/example with full term
IRIs and link the supported Paignia API/profile, its limits and fixture evidence.
For `extension_required`, link the upstream proposal, a minimal synthetic example
and independently expected results. `unsupported` records the missing requirement
and its effect on claims or execution. Paignia's `domain_only` records why the
change needs no generic implementation. Pending assessments need a reason,
owner and next action; they do not establish semantic or executable support.

Inspect existing upstream work before proposing another term, solver or fixture.
An unchanged game with new parameters can reuse its intake key and add an
example. Innovate remains a parallel innovation/domain-method library: diffusion,
adoption, substitution, demand methods and domain assumptions remain here.
Paignia receives only reusable game representations, algorithms and conformance
fixtures. Preserve source revision, licence and attribution for
any transferred example or implementation.

On adoption, record the exact UOGTO revision/profile and full IRIs, Paignia
revision or artifact hash, fixture identity, and focused semantic/native/consumer
checks actually run, including independent expected results and invalid or
unsupported cases. Ontology conformance and numerical verification are separate.
A record or dependency pin alone does not complete a migration.

Migrate mature generic game implementations to Paignia method by method, only
after numerical/behavioural parity and caller compatibility are demonstrated.
Retain the working implementation until the replacement is qualified, with scope
and limitations labelled; keep the Paignia integration optional and acyclic.
New game changes still require intake. This policy does not itself rewrite
algorithms or complete existing migration or scientific acceptance gates.
