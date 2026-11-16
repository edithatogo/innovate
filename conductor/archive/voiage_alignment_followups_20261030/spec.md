# voiage alignment follow-ups (innovate)

Bounded follow-up to the existing scientific ecosystem track; it reuses that owner and adds no parallel programme.

## Scope
1. Confirm the public tabular/array boundary uses Polars (>=2.1) and PyArrow (>=26); record any retained exceptions.
2. Consume `voiage.contracts.normalized_input` bundles; verify `content_digest` round trips (multi-chunk columns, Parquet-normalised types).
3. Align toolchain: Python 3.15.1, uv, ruff, ty (Rust 1.92 where a Rust core exists); record retained-version exceptions.
4. Resolve or scope open Dependabot alerts; keep signed-commit rules satisfied for bot updates.
5. Installed-artifact consumer evidence outside the checkout (command, exit status, imported paths, revisions, hashes).

## Acceptance
Existing gates stay required. Unexecuted checks are not claimed. No release or external submission is implied.
Implementation was authorised by Dylan's 14 November 2026 request to address #559.
The later native migration selects Python 3.15.2 and Rust 1.99.1; the older
versions above are retained as the original scope, not downgrade instructions.
