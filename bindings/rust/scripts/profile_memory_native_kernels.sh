#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

iterations="${INNOVATE_RUST_MEMORY_PROFILE_ITERATIONS:-10000}"

cat <<EOF
Running Rust-native memory profiling with DHAT.
Iterations: ${iterations}
Output: ${repo_root}/dhat-heap.json
View with: https://nnethercote.github.io/dh_view/dh_view.html
EOF

INNOVATE_RUST_MEMORY_PROFILE_ITERATIONS="$iterations" \
  cargo run --release --example profile_memory_native_kernels
