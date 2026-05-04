#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

output="${INNOVATE_RUST_CPU_PROFILE_OUTPUT:-flamegraph-native-kernels.svg}"
frequency="${INNOVATE_RUST_CPU_PROFILE_FREQUENCY:-}"
metadata="${output}.metadata.txt"

if ! command -v cargo-flamegraph >/dev/null 2>&1; then
  cat <<'EOF'
cargo-flamegraph is not installed.
Install it with:
  cargo install flamegraph
EOF
  exit 1
fi

mkdir -p "$(dirname "$output")"

export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
export CARGO_PROFILE_RELEASE_DEBUG="${CARGO_PROFILE_RELEASE_DEBUG:-true}"

command=(cargo flamegraph --bench native_kernel --output "$output")
if [[ -n "$frequency" ]]; then
  command+=(--freq "$frequency")
fi
if [[ $# -gt 0 ]]; then
  command+=(-- "$@")
fi

git_revision="$(git rev-parse --short HEAD 2>/dev/null || true)"
if [[ -n "$git_revision" ]] && [[ -n "$(git status --porcelain -- . 2>/dev/null)" ]]; then
  git_revision="${git_revision}-dirty"
fi

cat <<EOF
Running Rust-native CPU flamegraph profiling.
Output: ${repo_root}/${output}
Metadata: ${repo_root}/${metadata}
CARGO_INCREMENTAL=${CARGO_INCREMENTAL}
CARGO_PROFILE_RELEASE_DEBUG=${CARGO_PROFILE_RELEASE_DEBUG}
EOF

{
  printf 'crate_dir=%s\n' "$repo_root"
  printf 'git_revision=%s\n' "${git_revision:-unknown}"
  printf 'rustc=%s\n' "$(rustc --version)"
  printf 'cargo=%s\n' "$(cargo --version)"
  printf 'cargo_flamegraph=%s\n' "$(cargo flamegraph --version 2>/dev/null || printf 'unknown')"
  printf 'CARGO_INCREMENTAL=%s\n' "$CARGO_INCREMENTAL"
  printf 'CARGO_PROFILE_RELEASE_DEBUG=%s\n' "$CARGO_PROFILE_RELEASE_DEBUG"
  if [[ -n "$frequency" ]]; then
    printf 'INNOVATE_RUST_CPU_PROFILE_FREQUENCY=%s\n' "$frequency"
  fi
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} >"$metadata"

exec "${command[@]}"
