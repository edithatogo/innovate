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
if (result.error) {
  switch (result.error.code) {
    case 'ENOENT': console.error('API reference generator could not start: interpreter not found.'); break;
    case 'EACCES': console.error('API reference generator could not start: interpreter is not executable.'); break;
    case 'ETIMEDOUT': console.error('API reference generator timed out.'); break;
    default: console.error('API reference generator could not start.');
  }
}
process.exit(result.status ?? 1);
