import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { test } from 'node:test';

const fromAstro = createRequire(import.meta.resolve('astro/package.json'));
const fromStorage = createRequire(fromAstro.resolve('unstorage'));
const fromAnymatch = createRequire(fromStorage.resolve('anymatch'));

for (const [parent, requireDependency] of [
  ['Astro', fromAstro],
  ['anymatch', fromAnymatch],
]) {
  test(`${parent} uses a picomatch parser that rejects repeated extglobs`, () => {
    const match = requireDependency('picomatch');
    const isMarkdown = match('**/*.md');
    assert.equal(isMarkdown('guides/start.md'), true);
    assert.equal(isMarkdown('guides/start.txt'), false);

    // Literal branches exercise the workspace's syntax boundary without
    // feeding the matcher a costly expression or input.
    for (const pattern of ['+(guide)', '*(guide)']) {
      assert.throws(() => match(pattern), {
        name: 'SyntaxError',
        message: 'Repeated extglob patterns are disabled in this workspace',
      });
    }
  });
}
