#!/usr/bin/env bash
# Run hooks in the explicitly selected repository environment without installing it.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
selected_env="${UV_PROJECT_ENVIRONMENT:-$repo_root/.venv-native}"
if [[ "$selected_env" != /* ]]; then selected_env="$repo_root/$selected_env"; fi
if [[ -x "$selected_env/bin/python" ]]; then
  selected_python="$selected_env/bin/python"
else
  selected_python="$selected_env/Scripts/python.exe"
fi
"$selected_python" -c 'import sys; expected=(3,15,2); sys.exit(0 if sys.version_info[:3] == expected else "Repository tools require Python 3.15.2")'
exec "$selected_python" -m "$@"
