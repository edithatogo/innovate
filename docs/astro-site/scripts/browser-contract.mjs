import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const repositoryRoot = fileURLToPath(new URL('../../../', import.meta.url));
export const siteRoot = path.join(repositoryRoot, 'docs/astro-site');
export const browserOrigin = 'http://127.0.0.1:4321';
export const projectNames = ['chromium-desktop', 'chromium-mobile-390'];
export const wcagTags = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'];
export const darkEntryRoutes = [
  '/innovate/',
  '/innovate/user-guide/getting-started/',
  '/innovate/api/python/',
  '/innovate/api/generated/julia/',
  '/innovate/api/generated/r/',
  '/innovate/native-migration/',
];
// Sample both a formerly versioned page and an unversioned retired route.
// Source/HTML checks and captureInputs still cover every manifest route.
export const retirementEntryRoutes = [
  '/innovate/latest/api/python/',
  '/innovate/migration/',
];
// Byte counts are decoded response bodies, not timing or compression scores.
export const budgets = Object.freeze({
  artifactBytes: 25_000_000,
  singleAssetBytes: 512_000,
  initialPageBytes: 1_500_000,
  initialPageRequests: 40,
  searchJourneyBytes: 2_000_000,
  searchJourneyRequests: 65,
});

export function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

export function safeRelative(value) {
  assert.equal(typeof value, 'string', 'A repository-relative path is required');
  assert(value && !/[\\?#\0]/u.test(value) && !path.posix.isAbsolute(value), `Unsafe path: ${value}`);
  assert(value.split('/').every((part) => part && part !== '.' && part !== '..'), `Unsafe path: ${value}`);
  return value;
}

export function routeForSource(relative, base = '/innovate/') {
  safeRelative(relative);
  assert(/\.mdx?$/u.test(relative), `Not a Markdown source: ${relative}`);
  assert(base.startsWith('/') && base.endsWith('/'), 'A trailing-slash base is required');
  const stem = relative.replace(/\.mdx?$/u, '').replace(/(^|\/)index$/u, '');
  return base + (stem ? `${stem.replace(/\/$/u, '')}/` : '');
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

export function loadInventory(root = repositoryRoot) {
  const contentRoot = 'docs/astro-site/src/content/docs';
  const policy = readJson(path.join(root, 'docs/contracts/content-quality.json'));
  const sitePolicy = readJson(path.join(root, 'docs/contracts/site-quality.json'));
  assert.equal(policy.schema_version, 1, 'Unsupported content policy');
  assert.equal(sitePolicy.schema_version, 1, 'Unsupported site policy');
  assert.equal(sitePolicy.base, '/innovate/', 'Update the browser entry routes when the site base changes');
  const pages = [];
  const historical = [];
  for (const [relative, entry] of Object.entries(policy.pages)) {
    assert(['index', 'reference', 'tutorial', 'historical'].includes(entry.kind), `Unknown classification: ${relative}`);
    const route = routeForSource(relative, sitePolicy.base);
    if (entry.kind === 'historical') {
      historical.push({ route, source: `${contentRoot}/${relative}` });
      continue;
    }
    const source = `${contentRoot}/${relative}`;
    const sourceText = fs.readFileSync(path.join(root, source), 'utf8');
    const frontmatter = sourceText.match(/^---\r?\n([\s\S]*?)\r?\n---/u)?.[1] ?? '';
    assert(!/^slug\s*:/mu.test(frontmatter), `Explicit slug needs browser route support: ${source}`);
    pages.push({ route, source, kind: entry.kind });
  }
  const manifestPath = safeRelative(policy.generated_api.manifest);
  const manifest = readJson(path.join(root, manifestPath));
  assert(Object.keys(manifest.pages).length > 0, 'The generated API inventory is empty');
  for (const [relative, entry] of Object.entries(manifest.pages)) {
    safeRelative(relative);
    const source = `${path.posix.dirname(manifestPath)}/${relative}`;
    assert(source.startsWith(`${contentRoot}/`), 'Generated page is outside the content directory');
    const bytes = fs.readFileSync(path.join(root, source));
    assert.equal(sha256(bytes), entry.sha256, `Stale generated API page: ${source}`);
    pages.push({ route: routeForSource(source.slice(contentRoot.length + 1), sitePolicy.base), source, kind: 'generated-api' });
  }
  pages.sort((a, b) => a.route.localeCompare(b.route, 'en'));
  assert(pages.length > 0, 'No current pages were classified');
  assert.equal(new Set(pages.map((page) => page.route)).size, pages.length, 'Duplicate current routes');
  for (const route of darkEntryRoutes) {
    assert(pages.some((page) => page.route === route), `Required current entry is absent: ${route}`);
  }
  const retirement = readJson(path.join(root, safeRelative(sitePolicy.retired_documentation)));
  assert.equal(retirement.schema_version, 1, 'Unsupported retirement policy');
  assert(/^[a-f0-9]{40}$/u.test(retirement.source_commit), 'Retirement history must name an immutable commit');
  assert(Array.isArray(retirement.entries), 'Retirement entries must be an array');
  const retired = [];
  for (const entry of retirement.entries) {
    assert(entry && typeof entry === 'object', 'Invalid retirement entry');
    if (!('route' in entry)) continue;
    const source = safeRelative(entry.path);
    assert(source.startsWith(`${contentRoot}/`), `Retired web source is outside the content directory: ${source}`);
    const route = routeForSource(source.slice(contentRoot.length + 1), sitePolicy.base);
    assert.equal(entry.route, route, `Retirement route does not match its source: ${source}`);
    assert(!pages.some((page) => page.route === route), `Retirement route collides with a current page: ${route}`);
    assert(pages.some((page) => page.route === entry.replacement), `Retirement target is not current: ${entry.replacement}`);
    const history = new URL(entry.history_url);
    assert(history.origin === 'https://github.com' && !history.username && !history.password
      && !history.search && !history.hash
      && decodeURIComponent(history.pathname) === `/edithatogo/innovate/blob/${retirement.source_commit}/${source}`,
    `Retirement history is not bound to its immutable source: ${source}`);
    retired.push({ route, source, replacement: entry.replacement, history_url: entry.history_url });
  }
  retired.sort((a, b) => a.route.localeCompare(b.route, 'en'));
  assert.equal(new Set(retired.map((page) => page.route)).size, retired.length, 'Duplicate retirement routes');
  for (const route of retirementEntryRoutes) {
    assert(retired.some((page) => page.route === route), `Required retirement entry is absent: ${route}`);
  }
  return { base: sitePolicy.base, pages, historical, retired, sitePolicy };
}

function visitFiles(root, relative, excluded, files) {
  const parts = relative.split('/');
  if (parts.some((part) => excluded.has(part))) return;
  const absolute = path.join(root, relative);
  const stat = fs.lstatSync(absolute);
  assert(!stat.isSymbolicLink(), `Symlinks cannot be qualified: ${relative}`);
  if (stat.isDirectory()) {
    for (const name of fs.readdirSync(absolute).sort()) {
      visitFiles(root, relative ? `${relative}/${name}` : name, excluded, files);
    }
  } else if (stat.isFile()) {
    files.add(relative);
  }
}

export function fingerprint(root, roots, excluded = []) {
  const names = new Set();
  for (const relative of roots) {
    if (relative) safeRelative(relative);
    let current = root;
    for (const part of relative.split('/').filter(Boolean)) {
      current = path.join(current, part);
      assert(!fs.lstatSync(current).isSymbolicLink(), `Symlinked input: ${relative}`);
    }
    visitFiles(root, relative, new Set(excluded), names);
  }
  assert(names.size > 0, 'The fingerprint inventory is empty');
  const files = Object.fromEntries([...names].sort().map((name) => [name, sha256(fs.readFileSync(path.join(root, name)))]));
  return { sha256: sha256(JSON.stringify(files)), files };
}

export function captureInputs(root = repositoryRoot) {
  const inventory = loadInventory(root);
  const dist = path.join(root, 'docs/astro-site/dist');
  const source = fingerprint(root, inventory.sitePolicy.source_roots, inventory.sitePolicy.exclude_source_parts);
  const artifact = fingerprint(dist, ['']);
  for (const page of inventory.pages) {
    assert(artifact.files[artifactPath(page.route, inventory.base)], `Missing rendered current page: ${page.route}`);
  }
  for (const page of inventory.retired) {
    assert(artifact.files[artifactPath(page.route, inventory.base)], `Missing rendered retirement page: ${page.route}`);
  }
  assert(artifact.files['pagefind/pagefind.js'], 'Build Pagefind before browser qualification');
  return { source, artifact, inventory };
}

export function artifactPath(pathname, base = '/innovate/') {
  assert(pathname.startsWith(base), `Resource escaped the configured base: ${pathname}`);
  const relative = decodeURIComponent(pathname.slice(base.length));
  const file = relative.endsWith('/') || !relative ? `${relative}index.html` : relative;
  return safeRelative(file);
}

export function expectedCaseTitles(inventory) {
  return [
    ...inventory.pages.map(({ route }) => `current page: ${route}`),
    ...darkEntryRoutes.map((route) => `dark entry: ${route}`),
    ...retirementEntryRoutes.map((route) => `retired page: ${route}`),
    'keyboard skip link and visible focus',
    'keyboard Pagefind search: native diffusion curve',
    'keyboard Pagefind search: Current Python API',
    'current navigation and retirement boundaries',
    'deterministic artifact budgets',
  ];
}

export function qualificationCoverage(inventory, cases) {
  const expected = new Set(projectNames.flatMap((project) => expectedCaseTitles(inventory).map((title) => `${project}\0${title}`)));
  const actual = new Set(cases.map(({ project, title }) => `${project}\0${title}`));
  return {
    expected: expected.size,
    observed: cases.length,
    missing: [...expected].filter((key) => !actual.has(key)).map((key) => key.replace('\0', ': ')),
    unexpected: [...actual].filter((key) => !expected.has(key)).map((key) => key.replace('\0', ': ')),
    duplicate_count: cases.length - actual.size,
    unsuccessful: cases.filter((entry) => entry.status !== 'passed').map(({ project, title, status }) => ({ project, title, status })),
  };
}

export function caseEvidenceComplete(entry) {
  const evidence = entry.evidence;
  const runtime = evidence?.['runtime-observation'];
  if (!runtime || typeof runtime.browser_version !== 'string' || !Array.isArray(runtime.responses)
    || runtime.console_errors?.length !== 0 || runtime.page_errors?.length !== 0
    || runtime.failed_internal_resources?.length !== 0) return false;
  let scan;
  let route;
  if (entry.title.startsWith('current page: ')) {
    scan = evidence['axe-light-page'];
    route = entry.title.slice('current page: '.length);
    if (evidence['classified-page']?.route !== route || !evidence['transfer-budget']) return false;
  } else if (entry.title.startsWith('retired page: ')) {
    scan = evidence['axe-retired-page'];
    route = entry.title.slice('retired page: '.length);
    const notice = evidence['retirement-page'];
    if (notice?.route !== route || typeof notice.source !== 'string'
      || !notice.source.startsWith('docs/astro-site/src/content/docs/')
      || typeof notice.replacement !== 'string' || !notice.replacement.startsWith('/innovate/')
      || !/^https:\/\/github\.com\/edithatogo\/innovate\/blob\/[a-f0-9]{40}\//u.test(notice.history_url ?? '')
      || notice.code_elements !== 0 || notice.version_selectors !== 0
      || notice.pagefind_bodies !== 0 || notice.robots !== 'noindex, follow') return false;
  } else if (entry.title === 'current navigation and retirement boundaries') {
    scan = evidence['axe-primary-navigation'];
    route = '/innovate/api/python/';
    if (!Array.isArray(evidence['primary-navigation']) || evidence['primary-navigation'].length === 0) return false;
  } else if (entry.title.startsWith('dark entry: ')) {
    scan = evidence['axe-dark-page'];
    route = entry.title.slice('dark entry: '.length);
    if (!evidence['transfer-budget']) return false;
  } else if (entry.title.startsWith('keyboard Pagefind search: ')) {
    scan = evidence['axe-search-dialog'];
    if (!evidence['pagefind-results']?.first_result || !evidence['transfer-budget']) return false;
  } else if (entry.title === 'deterministic artifact budgets' && !evidence['artifact-budget']) return false;
  if (route && !runtime.responses.some((response) => response.file === artifactPath(route) && response.status === 200)) return false;
  if (route || entry.title.startsWith('keyboard Pagefind search: ')) {
    const measurement = evidence['transfer-budget'];
    const search = entry.title.startsWith('keyboard Pagefind search: ');
    const byteLimit = search ? budgets.searchJourneyBytes : budgets.initialPageBytes;
    const requestLimit = search ? budgets.searchJourneyRequests : budgets.initialPageRequests;
    if (!measurement || !Number.isFinite(measurement.decoded_body_bytes) || measurement.decoded_body_bytes < 0
      || !Number.isInteger(measurement.successful_internal_responses) || measurement.successful_internal_responses < 0
      || measurement.byte_limit !== byteLimit || measurement.request_limit !== requestLimit
      || measurement.decoded_body_bytes > byteLimit || measurement.successful_internal_responses > requestLimit) return false;
  }
  if (scan && (scan.violations?.length !== 0 || JSON.stringify(scan.tags) !== JSON.stringify(wcagTags))) return false;
  if ((route || entry.title.startsWith('keyboard Pagefind search: ')) && !scan) return false;
  return true;
}

export function measureArtifact(dist, maxHtmlBytes) {
  const artifact = fingerprint(dist, ['']);
  const files = Object.keys(artifact.files).map((name) => ({ name, bytes: fs.statSync(path.join(dist, name)).size }));
  const total = files.reduce((sum, file) => sum + file.bytes, 0);
  assert(total <= budgets.artifactBytes, `Artifact is ${total} bytes; budget is ${budgets.artifactBytes}`);
  for (const file of files) {
    const limit = file.name.endsWith('.html') ? maxHtmlBytes : budgets.singleAssetBytes;
    assert(file.bytes <= limit, `${file.name} is ${file.bytes} bytes; budget is ${limit}`);
  }
  return { total_bytes: total, files: files.length, largest: files.sort((a, b) => b.bytes - a.bytes).slice(0, 10), budgets };
}
