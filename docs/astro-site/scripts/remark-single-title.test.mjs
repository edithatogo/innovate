import assert from 'node:assert/strict';
import test from 'node:test';
import remarkSingleTitle from './remark-single-title.mjs';

test('demotes body headings together without changing their text or IDs', () => {
  const tree = { children: [
    { type: 'heading', depth: 1, data: { id: 'stable' }, children: [{ type: 'text', value: 'Title' }] },
    { type: 'heading', depth: 2, children: [{ type: 'text', value: 'Child' }] },
  ] };
  remarkSingleTitle()(tree);
  assert.deepEqual(tree.children.map((n) => n.depth), [2, 3]);
  assert.equal(tree.children[0].data.id, 'stable');
  assert.equal(tree.children[0].children[0].value, 'Title');
});

test('a hash inside code does not change existing heading hierarchy', () => {
  const tree = { children: [{ type: 'code', value: '# example' }, { type: 'heading', depth: 2 }] };
  const before = structuredClone(tree);
  remarkSingleTitle()(tree);
  assert.deepEqual(tree, before);
});
