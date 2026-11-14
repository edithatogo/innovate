import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const executable = process.env.INNOVATE_DOCS_PYTHON ?? process.env.STARLIGHT_POLYGLOT_PYTHON
  ?? path.join(root, '.venv-docs', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const result = spawnSync(executable, [path.join(root, 'scripts/generate_api_reference.py'), ...process.argv.slice(2)], {
  cwd: root,
  stdio: 'inherit',
  timeout: 60_000,
});
if (result.error) console.error(result.error.message);
process.exit(result.status ?? 1);
