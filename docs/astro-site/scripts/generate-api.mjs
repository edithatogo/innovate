import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const executable = process.env.INNOVATE_DOCS_PYTHON ?? process.env.STARLIGHT_POLYGLOT_PYTHON
  ?? path.join(root, '.venv-docs', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
// Keep source-derived learning resources and retirement policy in the same
// build path as API generation. --check never rewrites a generated artifact.
for (const [script, args] of [
  ['generate_api_reference.py', process.argv.slice(2)],
  ['generate_learning_catalogue.py', process.argv.slice(2)],
  ['check_retired_documentation.py', []],
  ['check_community_documentation.py', []],
  ['check_learning_notebooks.py', []],
]) {
  const result = spawnSync(executable, [path.join(root, 'scripts', script), ...args], {
    cwd: root,
    stdio: 'inherit',
    timeout: 60_000,
  });
  if (result.error) console.error(result.error.message);
  if (result.status !== 0) process.exit(result.status ?? 1);
}
