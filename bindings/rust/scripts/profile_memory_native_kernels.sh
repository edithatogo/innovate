#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

iterations="${INNOVATE_RUST_MEMORY_PROFILE_ITERATIONS:-10000}"
output="${INNOVATE_RUST_MEMORY_PROFILE_OUTPUT:-${repo_root}/dhat-native-kernels-heap-${iterations}.json}"

if ! [[ "$iterations" =~ ^[1-9][0-9]*$ ]]; then
  echo "INNOVATE_RUST_MEMORY_PROFILE_ITERATIONS must be a positive integer, got: ${iterations}" >&2
  exit 2
fi

mkdir -p "$(dirname "$output")"
rm -f "$output"

cat <<EOF
Running Rust-native memory profiling with DHAT.
Iterations: ${iterations}
Workload: logistic fit/predict/simulate/summary/diagnose, Bass predict/simulate
Output: ${output}
View with: https://nnethercote.github.io/dh_view/dh_view.html
EOF

INNOVATE_RUST_MEMORY_PROFILE_ITERATIONS="$iterations" \
INNOVATE_RUST_MEMORY_PROFILE_OUTPUT="$output" \
  cargo run --release --example profile_memory_native_kernels
