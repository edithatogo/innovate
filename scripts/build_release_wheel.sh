#!/usr/bin/env bash
set -euo pipefail
SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
export SOURCE_DATE_EPOCH
environment="${1:-$PWD/.venv-release}"
output="${2:-$PWD/dist}"
python="$environment/bin/python"
maturin="$environment/bin/maturin"
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
"$maturin" sdist --out "$output"
mkdir "$staging/source"
tar -xzf "$output"/innovate-*.tar.gz -C "$staging/source"
source_root=("$staging"/source/innovate-*)
test "${#source_root[@]}" -eq 1
(
  cd "${source_root[0]}"
  "$maturin" build --release --locked -i "$python" --out "$staging/wheels"
)
for wheel in "$staging"/wheels/*.whl; do
  "$python" scripts/normalize_wheel_sbom.py "$wheel" "${source_root[0]}" "$output/${wheel##*/}"
done
