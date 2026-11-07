#!/usr/bin/env bash
set -euo pipefail
SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
export SOURCE_DATE_EPOCH
python="$PWD/.venv-release/bin/python"
maturin="$PWD/.venv-release/bin/maturin"
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
"$maturin" sdist --out dist
mkdir "$staging/source"
tar -xzf dist/innovate-*.tar.gz -C "$staging/source"
source_root=("$staging"/source/innovate-*)
test "${#source_root[@]}" -eq 1
(
  cd "${source_root[0]}"
  "$maturin" build --release --locked -i "$python" --out "$staging/wheels"
)
for wheel in "$staging"/wheels/*.whl; do
  "$python" scripts/normalize_wheel_sbom.py "$wheel" "${source_root[0]}" "dist/${wheel##*/}"
done
