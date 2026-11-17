import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const script = fileURLToPath(new URL('./generate-api.mjs', import.meta.url));

test('interpreter launch failure does not echo the configured path', () => {
  const marker = 'PRIVATE_INTERPRETER_MARKER_94712';
  for (const variable of ['INNOVATE_DOCS_PYTHON', 'STARLIGHT_POLYGLOT_PYTHON']) {
    const env = { ...process.env };
    delete env.INNOVATE_DOCS_PYTHON;
    delete env.STARLIGHT_POLYGLOT_PYTHON;
    env[variable] = `/not-an-interpreter/${marker}`;
    const result = spawnSync(process.execPath, [script, '--check'], { env, encoding: 'utf8' });
    assert.equal(result.status, 1);
    assert.doesNotMatch(result.stderr + result.stdout, new RegExp(marker, 'u'));
    assert.match(result.stderr, /API reference generator/u);
  }
});
