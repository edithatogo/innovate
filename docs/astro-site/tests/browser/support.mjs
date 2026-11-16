import { test as base, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import path from 'node:path';
import { browserOrigin, budgets, fingerprint, prefetchProbePath, siteRoot, wcagTags } from '../../scripts/browser-contract.mjs';
import { OriginalResponseCapture, verifiedArtifactResponse } from './original-response-capture.mjs';

export { expect };
export const test = base.extend({
  artifactHashes: [async ({}, use) => {
    await use(fingerprint(path.join(siteRoot, 'dist'), ['']).files);
  }, { scope: 'worker' }],
  runtime: [async ({ page, browser, artifactHashes }, use, testInfo) => {
    const consoleErrors = [];
    const pageErrors = [];
    const failedResources = [];
    const responses = [];
    const pending = [];
    const capture = new OriginalResponseCapture(await page.context().newCDPSession(page), { origin: browserOrigin });
    // Prefetch clones the active URL loader at navigation; installation must
    // finish before the test can navigate or trigger its mandatory probe.
    await capture.start();
    page.on('console', (message) => {
      if (message.type() === 'error') consoleErrors.push(message.text());
    });
    page.on('pageerror', (error) => pageErrors.push(error.message));
    page.on('requestfailed', (request) => {
      if (new URL(request.url()).origin === browserOrigin) {
        failedResources.push({ url: request.url(), error: request.failure()?.errorText });
      }
    });
    page.on('response', (response) => {
      const url = new URL(response.url());
      if (url.origin !== browserOrigin) return;
      pending.push((async () => {
        try {
          const observation = await verifiedArtifactResponse(response, capture, artifactHashes);
          if (observation) responses.push(observation);
        } catch (error) {
          failedResources.push({ url: url.pathname + url.search, error: String(error) });
        }
      })());
    });
    const settle = async () => {
      let count;
      do {
        count = pending.length + capture.tasks.length;
        await capture.settle();
        await Promise.all(pending);
      } while (count !== pending.length + capture.tasks.length);
    };
    try {
      await use({ responses, settle });
    } finally {
      await settle();
      failedResources.push(...await capture.close());
      await settle();
      await attachJson(testInfo, 'runtime-observation', {
        browser_version: browser.version(),
        browser_executable: process.env.INNOVATE_DOCS_CHROMIUM_EXECUTABLE || browser.browserType().executablePath(),
        browser_executable_override: process.env.INNOVATE_DOCS_CHROMIUM_EXECUTABLE || null,
        viewport: page.viewportSize(),
        console_errors: consoleErrors,
        page_errors: pageErrors,
        failed_internal_resources: failedResources,
        responses,
        speculative_capture_scope: 'Original response-stage CDP Fetch bytes, continued unchanged after EOF and matched to Network completion; interception can affect timing/cache behavior. Playwright tracing is separate from this byte source.',
      });
      expect(consoleErrors, 'Browser console errors').toEqual([]);
      expect(pageErrors, 'Uncaught browser exceptions').toEqual([]);
      expect(failedResources, 'Internal resource or artifact-identity failures').toEqual([]);
    }
  }, { auto: true }],
});

export async function attachJson(testInfo, name, value) {
  await testInfo.attach(name, { body: Buffer.from(JSON.stringify(value)), contentType: 'application/json' });
}

export async function openPage(page, route) {
  const response = await page.goto(route, { waitUntil: 'load' });
  expect(response?.status(), `Document response for ${route}`).toBe(200);
  await expect(page.locator('h1')).toHaveCount(1);
  await expect(page.locator('h1')).toBeVisible();
  await expect(page.getByRole('main')).toHaveCount(1);
  await expect(page.locator('site-search button[data-open-modal]')).toBeEnabled();
  // Wait for the real Pagefind UI, including Starlight's idle initialization.
  await expect(page.locator('#starlight__search .pagefind-ui__search-input')).toBeAttached();
  await page.evaluate(() => document.fonts.ready);
  await page.waitForLoadState('networkidle');
}

export async function assertNoOverflow(page) {
  const dimensions = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    document: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
  }));
  expect(dimensions.document, `Horizontal document overflow: ${JSON.stringify(dimensions)}`).toBeLessThanOrEqual(dimensions.viewport + 1);
}

export async function scanAccessibility(page, testInfo, name) {
  const results = await new AxeBuilder({ page }).withTags(wcagTags).analyze();
  await attachJson(testInfo, `axe-${name}`, {
    url: page.url(),
    engine: results.testEngine,
    tags: wcagTags,
    violations: results.violations,
    incomplete: results.incomplete,
    passed_rule_ids: results.passes.map((entry) => entry.id),
    inapplicable_rule_ids: results.inapplicable.map((entry) => entry.id),
    manual_review_required: results.incomplete.length > 0,
    wcag_conformance_certified: false,
  });
  expect(results.violations, `Axe violations in ${name}`).toEqual([]);
}

export async function assertTableKeyboardScroll(page, testInfo) {
  const tables = page.locator('.sl-markdown-content table');
  const count = await tables.count();
  const observations = [];
  for (let index = 0; index < count; index += 1) {
    const table = tables.nth(index);
    const overflowing = await table.evaluate((element) => element.scrollWidth > element.clientWidth + 1);
    if (!overflowing) continue;
    await expect(table, 'Scrollable tables must join the normal keyboard focus order').toHaveAttribute('tabindex', '0');
    await table.focus();
    await expect(table).toBeFocused();
    const before = await table.evaluate((element) => ({
      left: element.scrollLeft,
      maximum: element.scrollWidth - element.clientWidth,
    }));
    const direction = before.left >= before.maximum - 1 ? 'ArrowLeft' : 'ArrowRight';
    await page.keyboard.press(direction);
    await expect.poll(async () => {
      const left = await table.evaluate((element) => element.scrollLeft);
      return direction === 'ArrowRight' ? left > before.left : left < before.left;
    }, { message: 'A focused overflowing table must scroll with the arrow keys' }).toBe(true);
    const after = await table.evaluate((element) => {
      const style = getComputedStyle(element);
      return { left: element.scrollLeft, outline: style.outlineStyle, outlineWidth: parseFloat(style.outlineWidth) };
    });
    expect(after.outline !== 'none' && after.outlineWidth > 0, 'Focused tables need a visible indicator').toBe(true);
    observations.push({ index, direction, before, after });
  }
  await attachJson(testInfo, 'table-keyboard-scroll', observations);
}

export async function assertTransferBudget(runtime, testInfo, search = false) {
  await runtime.settle();
  const bytes = runtime.responses.reduce((sum, response) => sum + response.bytes, 0);
  const requests = runtime.responses.length;
  const byteLimit = search ? budgets.searchJourneyBytes : budgets.initialPageBytes;
  const requestLimit = search ? budgets.searchJourneyRequests : budgets.initialPageRequests;
  await attachJson(testInfo, 'transfer-budget', { decoded_body_bytes: bytes, successful_internal_responses: requests, byte_limit: byteLimit, request_limit: requestLimit });
  expect(bytes, 'Decoded internal response-body budget').toBeLessThanOrEqual(byteLimit);
  expect(requests, 'Internal response count budget').toBeLessThanOrEqual(requestLimit);
}

/** Exercise real speculative loading, including its ordinary byte/request cost. */
export async function assertOriginalPrefetchCapture(page, runtime, testInfo) {
  const relative = prefetchProbePath(testInfo.project.name);
  const url = new URL(relative, browserOrigin).href;
  await Promise.all([
    page.waitForResponse((response) => response.url() === url),
    page.evaluate((href) => {
      const link = document.createElement('link');
      link.rel = 'prefetch';
      link.href = href;
      document.head.append(link);
    }, url),
  ]);
  await runtime.settle();
  const observed = runtime.responses.filter((response) => response.url === relative);
  expect(observed, 'The original prefetch must produce exactly one verified response').toHaveLength(1);
  const receipt = observed[0];
  expect(receipt.capture_method).toBe('cdp-fetch-original-response');
  expect(receipt.bytes).toBeGreaterThan(0);
  expect(receipt.capture_complete).toBe(true);
  expect(receipt.continued_unchanged).toBe(true);
  expect(receipt.network_loading_finished).toBe(true);
  await attachJson(testInfo, 'original-prefetch-probe', { trigger: 'link-rel-prefetch', ...receipt });
}

export async function focusWithTab(page, target, limit = 30) {
  for (let step = 0; step < limit; step += 1) {
    if (await target.evaluate((element) => element === document.activeElement)) return;
    await page.keyboard.press('Tab');
  }
  await expect(target, `Control was not keyboard-reachable within ${limit} Tab presses`).toBeFocused();
}

export async function showMobileMenu(page, isMobile) {
  if (!isMobile) return false;
  const button = page.locator('button[popovertarget="starlight__sidebar"]');
  if (await button.count() === 0) {
    // Splash pages have no sidebar. A missing toggle on a sidebar page is a defect.
    expect(await page.locator('html').getAttribute('data-has-sidebar')).toBeNull();
    return false;
  }
  await focusWithTab(page, button);
  await page.keyboard.press('Enter');
  await expect(page.locator('#starlight__sidebar')).toBeVisible();
  await expect.poll(() => page.locator('#starlight__sidebar').evaluate((element) => element.matches(':popover-open'))).toBe(true);
  return true;
}
