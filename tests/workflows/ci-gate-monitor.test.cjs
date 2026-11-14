'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const test = require('node:test');

// Exercise the actual github-script body without needing credentials or network.
const workflow = fs.readFileSync('.github/workflows/ci-gate-monitor.yml', 'utf8');
const start = workflow.indexOf('          script: |\n');
const end = workflow.indexOf('\n      - name:', start);
assert.ok(start >= 0 && end > start, 'monitor script must exist');
const script = workflow.slice(start + '          script: |\n'.length, end)
  .split('\n').map(line => line.replace(/^ {12}/, '')).join('\n');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const monitor = new AsyncFunction('github', 'context', 'core', script);

const baseRun = {
  id: 200, workflow_id: 17, run_attempt: 1, status: 'completed',
  conclusion: 'success', event: 'pull_request', name: 'CI',
  run_started_at: '2026-11-07T22:42:12Z', updated_at: '2026-11-07T22:46:48Z',
  head_branch: 'codex/fix', head_repository: {full_name: 'edithatogo/innovate'},
  html_url: 'https://github.com/edithatogo/innovate/actions/runs/200',
  head_commit: {author: {username: 'maintainer'}}
};
const title = '🚨 CI failed on codex/fix';
const baseIssue = {
  number: 576, title, comments: 0,
  body: '- **Run URL:** https://github.com/edithatogo/innovate/actions/runs/100\n'
};
const originalRun = {...baseRun, id: 100, conclusion: 'failure',
  run_started_at: '2026-11-07T10:55:45Z', updated_at: '2026-11-07T10:59:02Z'};
const repeatRun = {...originalRun, id: 150, run_started_at: '2026-11-07T14:23:44Z',
  updated_at: '2026-11-07T14:30:58Z'};
const repeatComment = {
  user: {login: 'github-actions[bot]', type: 'Bot'},
  body: '**CI Failure Detected Again**\n\n- Workflow: CI\n- Branch: codex/fix\n' +
    '- Run: https://github.com/edithatogo/innovate/actions/runs/150\n- Conclusion: failure\n\n@maintainer — please investigate.'
};

function fixture({run = baseRun, latest = run, issues = [baseIssue], original = originalRun,
  followups = [], reportedRuns = {150: repeatRun}} = {}) {
  const calls = {queries: [], updates: [], comments: [], creates: [], originals: [], outputs: [], followups: []};
  const github = {
    rest: {
      issues: {
        listForRepo: () => {},
        listComments: () => {},
        update: async args => calls.updates.push(args),
        createComment: async args => calls.comments.push(args),
        create: async args => { calls.creates.push(args); return {data: {number: 999}}; }
      },
      actions: {
        listWorkflowRuns: async args => {
          calls.queries.push(args);
          return {data: {workflow_runs: latest ? [latest] : []}};
        },
        getWorkflowRun: async args => {
          calls.originals.push(args);
          const target = args.run_id === 100 ? original : reportedRuns[args.run_id];
          if (target instanceof Error) throw target;
          if (!target) throw Object.assign(new Error('not found'), {status: 404});
          return {data: target};
        }
      }
    },
    paginate: async (method, args) => {
      if (method === github.rest.issues.listForRepo) {
        assert.deepEqual(args.labels, ['ci-failure']);
        assert.equal(args.state, 'open');
        return issues;
      }
      assert.equal(method, github.rest.issues.listComments);
      calls.followups.push(args);
      return followups;
    }
  };
  const context = {repo: {owner: 'edithatogo', repo: 'innovate'}, payload: {workflow_run: run}};
  const core = {info: () => {}, setOutput: (name, value) => calls.outputs.push({name, value})};
  return {calls, execute: () => monitor(github, context, core)};
}

test('a current same-origin recovery closes only the exact original incident', async () => {
  const f = fixture({issues: [
    {...baseIssue, number: 10, pull_request: {}},
    {...baseIssue, number: 11, title: `${title}-other`},
    baseIssue
  ]});
  await f.execute();
  assert.deepEqual(f.calls.queries, [{owner: 'edithatogo', repo: 'innovate',
    workflow_id: 17, branch: 'codex/fix', event: 'pull_request', per_page: 1}]);
  assert.deepEqual(f.calls.originals, [{owner: 'edithatogo', repo: 'innovate', run_id: 100}]);
  assert.deepEqual(f.calls.updates, [{owner: 'edithatogo', repo: 'innovate',
    issue_number: 576, state: 'closed', state_reason: 'completed'}]);
  assert.equal(f.calls.creates.length + f.calls.comments.length, 0);
});

test('no matching issue causes no recovery action', async () => {
  const f = fixture({issues: [{...baseIssue, title: '🚨 Release Please failed on codex/fix'}]});
  await f.execute();
  assert.equal(f.calls.queries.length + f.calls.updates.length, 0);
});

test('stale success, another attempt, a newer failure or in-progress run never closes', async () => {
  for (const latest of [
    {...baseRun, id: 201},
    {...baseRun, run_attempt: 2},
    {...baseRun, id: 201, conclusion: 'failure'},
    {...baseRun, id: 201, status: 'in_progress', conclusion: null},
    {...baseRun, status: 'in_progress', conclusion: null},
    null
  ]) {
    const f = fixture({latest});
    await f.execute();
    assert.equal(f.calls.updates.length, 0, JSON.stringify(latest));
  }
});

test('foreign forks and unrelated original runs stay open', async () => {
  const foreign = {full_name: 'other/innovate'};
  const cases = [
    {run: {...baseRun, head_repository: foreign}},
    {latest: {...baseRun, head_repository: foreign}},
    {latest: {...baseRun, event: 'schedule'}},
    {latest: {...baseRun, workflow_id: 18}},
    {latest: {...baseRun, head_branch: 'main'}},
    {issues: [{...baseIssue, comments: 1}]},
    {issues: [{...baseIssue, body: 'No run reference'}]},
    {original: {...originalRun, head_repository: foreign}},
    {original: {...originalRun, event: 'schedule'}},
    {original: {...originalRun, workflow_id: 18}},
    {original: {...originalRun, head_branch: 'main'}}
  ];
  for (const scenario of cases) {
    const f = fixture(scenario);
    await f.execute();
    assert.equal(f.calls.updates.length, 0, JSON.stringify(scenario));
  }
});

test('verified bot-only repeated failures close after a newer matching success', async () => {
  const f = fixture({issues: [{...baseIssue, comments: 1}], followups: [repeatComment]});
  await f.execute();
  assert.deepEqual(f.calls.followups, [{owner: 'edithatogo', repo: 'innovate', issue_number: 576}]);
  assert.deepEqual(f.calls.originals.map(call => call.run_id), [100, 150]);
  assert.equal(f.calls.updates[0].issue_number, 576);
});

test('human, unknown, incomplete or mismatched follow-ups keep the issue open', async () => {
  const scenarios = [
    {followups: [{...repeatComment, user: {login: 'alice', type: 'User'}}]},
    {followups: [{...repeatComment, user: {login: 'other-bot[bot]', type: 'Bot'}}]},
    {followups: [{...repeatComment, body: 'Unresolved manual follow-up'}]},
    {followups: [{...repeatComment, body: repeatComment.body.replace('codex/fix', 'main')}]},
    {followups: [{...repeatComment, body: repeatComment.body.replace('edithatogo/innovate', 'other/innovate')}]},
    {followups: [{...repeatComment, body: repeatComment.body.replace('github.com/edithatogo/innovate', '@@@')}]},
    {followups: []},
    {followups: [repeatComment], reportedRuns: {150: {...repeatRun, event: 'schedule'}}},
    {followups: [repeatComment], reportedRuns: {150: {...repeatRun, head_repository: {full_name: 'other/innovate'}}}},
    {followups: [repeatComment], reportedRuns: {150: {...repeatRun, status: 'in_progress'}}},
    {followups: [repeatComment], reportedRuns: {150: {...repeatRun, updated_at: '2026-11-07T23:00:00Z'}}},
    {followups: [repeatComment], reportedRuns: {150: {...repeatRun, updated_at: null}}},
    {followups: [repeatComment], reportedRuns: {}}
  ];
  for (const scenario of scenarios) {
    const f = fixture({issues: [{...baseIssue, comments: 1}], ...scenario});
    await f.execute();
    assert.equal(f.calls.updates.length, 0, JSON.stringify(scenario));
  }
});

test('a late rerun or missing timestamps cannot be mistaken for a recovery', async () => {
  const scenarios = [
    {latest: {...baseRun, run_started_at: null}},
    {original: {...originalRun, status: 'in_progress'}},
    {original: {...originalRun, updated_at: '2026-11-07T23:00:00Z'}},
    {original: {...originalRun, updated_at: null}}
  ];
  for (const scenario of scenarios) {
    const f = fixture(scenario);
    await f.execute();
    assert.equal(f.calls.updates.length, 0, JSON.stringify(scenario));
  }
});

test('a successful rerun of the original run may resolve the incident', async () => {
  const rerun = {...baseRun, id: 100, run_attempt: 2};
  const f = fixture({run: rerun, latest: rerun, original: rerun});
  await f.execute();
  assert.equal(f.calls.updates.length, 1);
});

test('an expired original run stays open and other API errors remain visible', async () => {
  const missing = fixture({original: Object.assign(new Error('not found'), {status: 404})});
  await missing.execute();
  assert.equal(missing.calls.updates.length, 0);

  const denied = fixture({original: Object.assign(new Error('forbidden'), {status: 403})});
  await assert.rejects(denied.execute(), /forbidden/);
  assert.equal(denied.calls.updates.length, 0);
});

test('a current failure still records the matching incident or creates one', async () => {
  const run = {...baseRun, conclusion: 'failure'};
  const existing = fixture({run});
  await existing.execute();
  assert.equal(existing.calls.comments.length, 1);
  assert.equal(existing.calls.comments[0].issue_number, 576);
  assert.equal(existing.calls.updates.length + existing.calls.creates.length, 0);

  const missing = fixture({run, issues: []});
  await missing.execute();
  assert.equal(missing.calls.creates.length, 1);
  assert.equal(missing.calls.creates[0].title, title);
  assert.deepEqual(missing.calls.creates[0].labels, ['ci-failure', 'bug']);
});

test('an older failure is ignored only after a newer same-origin success', async () => {
  const run = {...baseRun, conclusion: 'failure', updated_at: '2026-11-07T14:30:58Z'};
  const resolved = fixture({run, latest: {...baseRun, id: 201}});
  await resolved.execute();
  assert.equal(resolved.calls.comments.length + resolved.calls.creates.length, 0);
  assert.deepEqual(resolved.calls.outputs, [{name: 'superseded', value: 'true'}]);

  for (const latest of [
    {...baseRun, id: 201, status: 'in_progress', conclusion: null},
    {...baseRun, id: 201, head_repository: {full_name: 'other/innovate'}},
    {...baseRun, run_attempt: 2, conclusion: 'failure'}
  ]) {
    const pending = fixture({run, latest});
    await pending.execute();
    assert.equal(pending.calls.comments.length, 1);
  }
});

test('a retry that fails after a newer run succeeds still produces an alert', async () => {
  const run = {...baseRun, id: 100, run_attempt: 2, conclusion: 'failure',
    updated_at: '2026-11-07T23:00:00Z'};
  const f = fixture({run, latest: baseRun});
  await f.execute();
  assert.equal(f.calls.comments.length, 1);
  assert.equal(f.calls.outputs.length, 0);
});

test('a run without a branch does not alter incidents', async () => {
  const f = fixture({run: {...baseRun, head_branch: null}});
  await f.execute();
  assert.equal(f.calls.queries.length + f.calls.updates.length + f.calls.creates.length, 0);
});
