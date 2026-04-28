#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v cargo-flamegraph >/dev/null 2>&1; then
  cat <<'EOF'
cargo-flamegraph is not installed.
Install it with:
  cargo install flamegraph
EOF
  exit 1
fi

exec cargo flamegraph --bench native_kernel --output flamegraph-native-kernels.svg
