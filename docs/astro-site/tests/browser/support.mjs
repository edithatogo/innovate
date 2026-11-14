import { test as base, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import path from 'node:path';
import { artifactPath, browserOrigin, budgets, fingerprint, sha256, siteRoot, wcagTags } from '../../scripts/browser-contract.mjs';

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
          const status = response.status();
          if (status >= 400) throw new Error(`HTTP ${status}`);
          if (status >= 300) return;
          const file = artifactPath(url.pathname);
          const bytes = await response.body();
          const digest = sha256(bytes);
          if (!artifactHashes[file]) throw new Error(`Response is not in the qualified artifact: ${file}`);
          if (digest !== artifactHashes[file]) throw new Error(`Response differs from artifact: ${file}`);
          responses.push({ url: url.pathname + url.search, file, status, bytes: bytes.length, sha256: digest });
        } catch (error) {
          failedResources.push({ url: url.pathname + url.search, error: String(error) });
        }
      })());
    });
    const settle = async () => {
      let count;
      do {
        count = pending.length;
        await Promise.all(pending);
      } while (count !== pending.length);
    };
    try {
      await use({ responses, settle });
    } finally {
      await settle();
      await attachJson(testInfo, 'runtime-observation', {
        browser_version: browser.version(),
        viewport: page.viewportSize(),
        console_errors: consoleErrors,
        page_errors: pageErrors,
        failed_internal_resources: failedResources,
        responses,
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
