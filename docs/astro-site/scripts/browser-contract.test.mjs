import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { artifactPath, budgets, caseEvidenceComplete, darkEntryRoutes, expectedCaseTitles, fingerprint, loadInventory, measureArtifact, projectNames, qualificationCoverage, routeForSource, safeRelative, sha256, wcagTags } from './browser-contract.mjs';

function temporary(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'innovate-browser-contract-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  return root;
}

function write(root, relative, contents) {
  const absolute = path.join(root, relative);
  fs.mkdirSync(path.dirname(absolute), { recursive: true });
  fs.writeFileSync(absolute, typeof contents === 'string' ? contents : JSON.stringify(contents));
}

test('source routes preserve index directories and MDX paths', () => {
  assert.equal(routeForSource('index.md'), '/innovate/');
  assert.equal(routeForSource('api/generated/index.md'), '/innovate/api/generated/');
  assert.equal(routeForSource('user-guide/getting-started.mdx'), '/innovate/user-guide/getting-started/');
  assert.equal(routeForSource('core/myindex.md'), '/innovate/core/myindex/');
});

test('resource mapping rejects base escapes and decoded traversal', () => {
  assert.equal(artifactPath('/innovate/'), 'index.html');
  assert.equal(artifactPath('/innovate/api/python/'), 'api/python/index.html');
  assert.equal(artifactPath('/innovate/pagefind/pagefind.js'), 'pagefind/pagefind.js');
  for (const bad of ['/innovate-other/a', '/_astro/a.js', '/innovate/%2e%2e/secret', '/innovate/a%5cb']) {
    assert.throws(() => artifactPath(bad));
  }
  for (const bad of ['../escape', '/absolute', 'a/../b', 'a//b', 'a?b']) assert.throws(() => safeRelative(bad));
});

test('the route inventory includes classified and generated pages and rejects stale generation', (t) => {
  const root = temporary(t);
  const sourceRoot = 'docs/astro-site/src/content/docs';
  const pages = {
    'index.md': { kind: 'index' },
    'user-guide/getting-started.mdx': { kind: 'tutorial' },
    'api/python.md': { kind: 'index' },
    'native-migration.md': { kind: 'reference' },
    'old-design.md': { kind: 'historical' },
  };
  for (const name of Object.keys(pages)) write(root, `${sourceRoot}/${name}`, '---\ntitle: Fixture\n---\nCurrent text.\n');
  const manifest = { pages: {} };
  for (const name of ['julia.md', 'r.md']) {
    const contents = `---\ntitle: ${name}\n---\nGenerated text.\n`;
    write(root, `${sourceRoot}/api/generated/${name}`, contents);
    manifest.pages[name] = { sha256: sha256(contents) };
  }
  write(root, `${sourceRoot}/api/generated/manifest.json`, manifest);
  write(root, 'docs/contracts/content-quality.json', { schema_version: 1, pages, generated_api: { manifest: `${sourceRoot}/api/generated/manifest.json` } });
  write(root, 'docs/contracts/site-quality.json', { schema_version: 1, base: '/innovate/' });
  const inventory = loadInventory(root);
  assert.deepEqual(inventory.pages.map((page) => page.route).sort(), [...darkEntryRoutes].sort());
  assert.equal(inventory.historical.length, 1);
  write(root, `${sourceRoot}/api/generated/r.md`, 'Unrecorded edit');
  assert.throws(() => loadInventory(root), /Stale generated API page/u);
});

test('fingerprints bind paths and bytes, exclude reports, and reject symlinks', (t) => {
  const root = temporary(t);
  fs.mkdirSync(path.join(root, '.reports'));
  fs.writeFileSync(path.join(root, 'source.md'), 'original');
  fs.writeFileSync(path.join(root, '.reports/result.json'), 'first');
  const original = fingerprint(root, [''], ['.reports']);
  fs.writeFileSync(path.join(root, '.reports/result.json'), 'changed');
  assert.deepEqual(fingerprint(root, [''], ['.reports']), original);
  fs.writeFileSync(path.join(root, 'source.md'), 'modified');
  assert.notEqual(fingerprint(root, [''], ['.reports']).sha256, original.sha256);
  fs.writeFileSync(path.join(root, 'source.md'), 'original');
  fs.renameSync(path.join(root, 'source.md'), path.join(root, 'renamed.md'));
  assert.notEqual(fingerprint(root, [''], ['.reports']).sha256, original.sha256);
  fs.symlinkSync(path.join(root, 'renamed.md'), path.join(root, 'linked.md'));
  assert.throws(() => fingerprint(root, [''], ['.reports']), /Symlinks/u);
});

test('missing, filtered, skipped, and duplicated cases cannot qualify', () => {
  const inventory = { pages: darkEntryRoutes.map((route) => ({ route })) };
  const cases = projectNames.flatMap((project) => expectedCaseTitles(inventory).map((title) => ({ project, title, status: 'passed' })));
  assert.equal(cases.length, 34);
  const complete = qualificationCoverage(inventory, cases);
  assert.equal(complete.expected, cases.length);
  assert.deepEqual(complete.missing, []);
  assert.deepEqual(complete.unsuccessful, []);
  assert.equal(qualificationCoverage(inventory, cases.slice(1)).missing.length, 1);
  const skipped = structuredClone(cases);
  skipped[0].status = 'skipped';
  assert.equal(qualificationCoverage(inventory, skipped).unsuccessful.length, 1);
  assert.equal(qualificationCoverage(inventory, [...cases, cases[0]]).duplicate_count, 1);
  assert.equal(qualificationCoverage(inventory, cases.filter((entry) => entry.project === projectNames[0])).missing.length, cases.length / 2);
});

test('a passing test without its browser, document, and axe receipts cannot qualify', () => {
  const entry = {
    title: 'current page: /innovate/',
    evidence: {
      'runtime-observation': { browser_version: 'fixture-version', console_errors: [], page_errors: [], failed_internal_resources: [], responses: [{ file: 'index.html', status: 200 }] },
      'classified-page': { route: '/innovate/' },
      'transfer-budget': { decoded_body_bytes: 50, successful_internal_responses: 1, byte_limit: budgets.initialPageBytes, request_limit: budgets.initialPageRequests },
      'axe-light-page': { violations: [], tags: wcagTags },
    },
  };
  assert.equal(caseEvidenceComplete(entry), true);
  const missingAxe = structuredClone(entry);
  delete missingAxe.evidence['axe-light-page'];
  assert.equal(caseEvidenceComplete(missingAxe), false);
  const missingDocument = structuredClone(entry);
  missingDocument.evidence['runtime-observation'].responses = [];
  assert.equal(caseEvidenceComplete(missingDocument), false);
  const consoleError = structuredClone(entry);
  consoleError.evidence['runtime-observation'].console_errors.push('Uncaught error');
  assert.equal(caseEvidenceComplete(consoleError), false);
});

test('asset budgets fail on excessive decoded asset bytes', (t) => {
  const root = temporary(t);
  fs.writeFileSync(path.join(root, 'index.html'), '<h1>Example</h1>');
  fs.writeFileSync(path.join(root, 'entry.js'), 'export const ready = true;');
  assert.equal(measureArtifact(root, 2_000_000).files, 2);
  fs.writeFileSync(path.join(root, 'entry.js'), Buffer.alloc(budgets.singleAssetBytes + 1));
  assert.throws(() => measureArtifact(root, 2_000_000), /entry\.js.*budget/u);
});

test('receipt budgets reject oversized, invalid, or substituted measurements and limits', () => {
  const entry = {
    title: 'keyboard Pagefind search: Current Python API',
    evidence: {
      'runtime-observation': { browser_version: 'fixture-version', console_errors: [], page_errors: [], failed_internal_resources: [], responses: [] },
      'pagefind-results': { first_result: '/innovate/api/python/' },
      'axe-search-dialog': { violations: [], tags: wcagTags },
      'transfer-budget': { decoded_body_bytes: 50, successful_internal_responses: 1, byte_limit: budgets.searchJourneyBytes, request_limit: budgets.searchJourneyRequests },
    },
  };
  assert.equal(caseEvidenceComplete(entry), true);
  for (const change of [
    { decoded_body_bytes: budgets.searchJourneyBytes + 1 },
    { successful_internal_responses: budgets.searchJourneyRequests + 1 },
    { byte_limit: budgets.searchJourneyBytes + 1 },
    { request_limit: budgets.searchJourneyRequests + 1 },
    { byte_limit: budgets.initialPageBytes, request_limit: budgets.initialPageRequests },
    { decoded_body_bytes: NaN },
    { decoded_body_bytes: -1 },
    { successful_internal_responses: 1.5 },
    { successful_internal_responses: -1 },
  ]) {
    const invalid = structuredClone(entry);
    Object.assign(invalid.evidence['transfer-budget'], change);
    assert.equal(caseEvidenceComplete(invalid), false, JSON.stringify(change));
  }
});
