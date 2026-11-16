import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { artifactPath, budgets, captureInputs, caseEvidenceComplete, darkEntryRoutes, expectedCaseTitles, fingerprint, loadInventory, measureArtifact, projectNames, qualificationCoverage, retirementEntryRoutes, routeForSource, safeRelative, sha256, wcagTags } from './browser-contract.mjs';

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

function writeInventory(root) {
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
  write(root, 'docs/contracts/site-quality.json', {
    schema_version: 1, base: '/innovate/', retired_documentation: 'docs/contracts/retired-documentation.json',
    source_roots: ['docs/contracts'], exclude_source_parts: [],
  });
  const commit = 'a'.repeat(40);
  const retirement = {
    schema_version: 1, source_commit: commit,
    entries: ['latest/api/python.md', 'migration/index.md'].map((relative) => ({
      path: `${sourceRoot}/${relative}`,
      route: routeForSource(relative),
      replacement: '/innovate/api/python/',
      history_url: `https://github.com/edithatogo/innovate/blob/${commit}/${sourceRoot}/${relative}`,
    })),
  };
  write(root, 'docs/contracts/retired-documentation.json', retirement);
  return { sourceRoot, retirement };
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
  const { sourceRoot } = writeInventory(root);
  const inventory = loadInventory(root);
  assert.deepEqual(inventory.pages.map((page) => page.route).sort(), [...darkEntryRoutes].sort());
  assert.equal(inventory.historical.length, 1);
  assert.deepEqual(inventory.retired.map((entry) => entry.route), retirementEntryRoutes);
  write(root, `${sourceRoot}/api/generated/r.md`, 'Unrecorded edit');
  assert.throws(() => loadInventory(root), /Stale generated API page/u);
});

test('retirement inventory rejects mutable history, redirected identities, missing samples and current targets', (t) => {
  const root = temporary(t);
  const { retirement } = writeInventory(root);
  for (const mutate of [
    (value) => { value.source_commit = 'main'; },
    (value) => { value.entries[0].history_url = value.entries[0].history_url.replace('a'.repeat(40), 'main'); },
    (value) => { value.entries[0].history_url = value.entries[0].history_url.replace('github.com', 'example.org'); },
    (value) => { value.entries[0].route = '/innovate/another-old-page/'; },
    (value) => { value.entries[0].replacement = '/innovate/removed/'; },
    (value) => { value.entries.push(value.entries[0]); },
    (value) => { value.entries.pop(); },
  ]) {
    const broken = structuredClone(retirement);
    mutate(broken);
    write(root, 'docs/contracts/retired-documentation.json', broken);
    assert.throws(() => loadInventory(root));
  }
});

test('artifact qualification requires every retired route, not just the browser samples', (t) => {
  const root = temporary(t);
  const { retirement, sourceRoot } = writeInventory(root);
  retirement.entries.push({
    path: `${sourceRoot}/another-old-page.md`, route: '/innovate/another-old-page/', replacement: '/innovate/api/python/',
    history_url: `https://github.com/edithatogo/innovate/blob/${retirement.source_commit}/${sourceRoot}/another-old-page.md`,
  });
  write(root, 'docs/contracts/retired-documentation.json', retirement);
  const inventory = loadInventory(root);
  for (const entry of [...inventory.pages, ...inventory.retired]) {
    write(root, `docs/astro-site/dist/${artifactPath(entry.route)}`, '<h1>A fixture</h1>');
  }
  write(root, 'docs/astro-site/dist/pagefind/pagefind.js', 'export const fixture = true;');
  assert.equal(captureInputs(root).inventory.retired.length, 3);
  fs.unlinkSync(path.join(root, 'docs/astro-site/dist/another-old-page/index.html'));
  assert.throws(() => captureInputs(root), /Missing rendered retirement page/u);
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
  assert.equal(cases.length, 38);
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

test('retirement cases require notice identity, search exclusion, no code, accessibility and budget receipts', () => {
  const entry = {
    title: 'retired page: /innovate/latest/api/python/',
    evidence: {
      'runtime-observation': { browser_version: 'fixture-version', console_errors: [], page_errors: [], failed_internal_resources: [], responses: [{ file: 'latest/api/python/index.html', status: 200 }] },
      'retirement-page': {
        route: '/innovate/latest/api/python/', source: 'docs/astro-site/src/content/docs/latest/api/python.md',
        replacement: '/innovate/api/python/',
        history_url: `https://github.com/edithatogo/innovate/blob/${'a'.repeat(40)}/docs/astro-site/src/content/docs/latest/api/python.md`,
        code_elements: 0, version_selectors: 0, pagefind_bodies: 0, robots: 'noindex, follow',
      },
      'transfer-budget': { decoded_body_bytes: 100, successful_internal_responses: 1, byte_limit: budgets.initialPageBytes, request_limit: budgets.initialPageRequests },
      'axe-retired-page': { violations: [], tags: wcagTags },
    },
  };
  assert.equal(caseEvidenceComplete(entry), true);
  for (const change of [
    { route: '/innovate/migration/' }, { source: '' }, { replacement: 'https://example.org/' },
    { history_url: 'https://github.com/edithatogo/innovate/blob/main/old.md' },
    { code_elements: 1 }, { version_selectors: 1 }, { pagefind_bodies: 1 }, { robots: 'index, follow' },
  ]) {
    const invalid = structuredClone(entry);
    Object.assign(invalid.evidence['retirement-page'], change);
    assert.equal(caseEvidenceComplete(invalid), false, JSON.stringify(change));
  }
  for (const missing of ['axe-retired-page', 'retirement-page', 'transfer-budget']) {
    const invalid = structuredClone(entry);
    delete invalid.evidence[missing];
    assert.equal(caseEvidenceComplete(invalid), false, missing);
  }
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
