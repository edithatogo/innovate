#!/usr/bin/env bash
# Read the wheel SBOM and install only the exact frozen runtime dependency set.
set -euo pipefail
shopt -s nullglob
wheels=(dist/*.whl)
sdists=(dist/*.tar.gz)
if [ "${#wheels[@]}" -ne 1 ] || [ "${#sdists[@]}" -ne 1 ]; then
  echo 'Expected exactly one normalized wheel and one source distribution.' >&2
  exit 1
fi
test ! -e .grype-runtime
mkdir -p .grype-inputs
sha256sum "${wheels[@]}" "${sdists[@]}" > .grype-inputs/subjects.sha256
.venv-release/bin/python scripts/extract_wheel_sbom.py "${wheels[0]}" .grype-inputs/wheel-rust.sbom.cdx.json
uv export --frozen --no-dev --no-emit-project --all-extras \
  --format requirements-txt --output-file .grype-inputs/runtime-requirements.txt
uv venv --python 3.15.2 .grype-runtime
uv pip install --python .grype-runtime/bin/python --require-hashes \
  -r .grype-inputs/runtime-requirements.txt
uv pip install --python .grype-runtime/bin/python --no-deps "${wheels[0]}"
uv pip check --python .grype-runtime/bin/python
uvx --from cyclonedx-bom==7.3.0 cyclonedx-py environment \
  .grype-runtime/bin/python --pyproject pyproject.toml --mc-type library \
  --sv 1.6 --output-reproducible --of JSON --validate \
  -o .grype-inputs/python.sbom.cdx.json
sha256sum .grype-inputs/*.cdx.json >> .grype-inputs/subjects.sha256
sha256sum --check .grype-inputs/subjects.sha256
