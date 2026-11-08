#!/usr/bin/env bash
set -euo pipefail
SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
export SOURCE_DATE_EPOCH
environment="${1:-$PWD/.venv-release}"
output="${2:-$PWD/dist}"
environment="$(cd "$environment" && pwd)"
mkdir -p "$output"
output="$(cd "$output" && pwd)"
python="$environment/bin/python"
maturin="$environment/bin/maturin"
staging="$(mktemp -d "${TMPDIR:-/tmp}/innovate-release.XXXXXX")"
trap 'rm -rf "$staging"' EXIT
mkdir "$staging/sdist" "$staging/source"
"$maturin" sdist --out "$staging/sdist"
archives=("$staging"/sdist/innovate-*.tar.gz)
test "${#archives[@]}" -eq 1
test -f "${archives[0]}"
tar -xzf "${archives[0]}" -C "$staging/source"
source_root=("$staging"/source/innovate-*)
test "${#source_root[@]}" -eq 1
(
  cd "${source_root[0]}"
  "$maturin" build --release --locked -i "$python" --out "$staging/wheels"
)
for wheel in "$staging"/wheels/*.whl; do
  "$python" scripts/normalize_wheel_sbom.py "$wheel" "${source_root[0]}" "$output/${wheel##*/}"
done
cp "${archives[0]}" "$output/"
