'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const test = require('node:test');

// Exercise the actual event/actor predicate without launching the tag writer.
// GitHub compares strings without case sensitivity. Normalize both literals and
// inputs for this predicate's ASCII string comparisons and boolean operators;
// this is not a general GitHub expression evaluator.
const workflow = fs.readFileSync('.github/workflows/tagbot.yml', 'utf8');
const condition = workflow.match(/^    if: (.+)$/m);
assert.ok(condition, 'the write-scoped TagBot job must have an event/actor guard');
const expression = condition[1].replace(/'([^']*)'/g, (_, value) => JSON.stringify(value.toLowerCase()));
const evaluate = new Function('github', `return (${expression});`);
const normalize = value => typeof value === 'string' ? value.toLowerCase() : value;
const permits = github => evaluate(Object.fromEntries(
  Object.entries(github).map(([name, value]) => [name, normalize(value)])
));

test('TagBot accepts its pinned upstream actor and explicit manual dispatch', () => {
  for (const actor of ['JuliaTagBot', 'juliatagbot', 'JULIAtagBOT']) {
    assert.equal(permits({event_name: 'issue_comment', actor}), true, actor);
  }
  assert.equal(permits({event_name: 'workflow_dispatch', actor: 'edithatogo'}), true);
});

test('ordinary issue comments never schedule the write-scoped tag job', () => {
  for (const actor of ['edithatogo', 'contributor', 'github-actions[bot]', 'JuliaRegistrator', 'JuliaTagBot-evil', 'JuliaTagBot[bot]', '']) {
    assert.equal(permits({event_name: 'issue_comment', actor}), false, actor);
  }
});

test('a trusted actor does not authorize an unsupported event', () => {
  for (const event_name of ['pull_request', 'push', 'schedule', '', undefined]) {
    assert.equal(permits({event_name, actor: 'JuliaTagBot'}), false, event_name);
  }
});
