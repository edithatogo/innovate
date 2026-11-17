/** Run dependency-free documentation regression suites in the configured Python. */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const executable = process.env.INNOVATE_DOCS_PYTHON ?? process.env.STARLIGHT_POLYGLOT_PYTHON
  ?? path.join(root, '.venv-docs', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');

for (const [directory, suite] of [
  ['tests', 'test_retired_documentation.py'],
  ['tests', 'test_community_documentation.py'],
  ['tests', 'test_learning_notebooks.py'],
  ['tests/native', 'test_learning_catalogue.py'],
  ['tests/native', 'test_documentation_conventions.py'],
]) {
  test(`documentation contract: ${suite}`, () => {
    const result = spawnSync(executable, ['-I', '-S', '-m', 'unittest', 'discover', '-s', directory, '-p', suite], {
      cwd: root,
      encoding: 'utf8',
      timeout: 60_000,
    });
    assert.ifError(result.error);
    assert.equal(result.status, 0, `${suite}\n${result.stdout}\n${result.stderr}`);
    assert.match(result.stderr, /Ran [1-9]\d* tests?/u, 'Discovery must run tests rather than accept an empty suite');
  });
}
