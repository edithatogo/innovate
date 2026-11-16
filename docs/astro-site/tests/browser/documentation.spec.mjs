import path from 'node:path';
import { budgets, darkEntryRoutes, loadInventory, measureArtifact, siteRoot } from '../../scripts/browser-contract.mjs';
import { test, expect, openPage, assertNoOverflow, assertTableKeyboardScroll, scanAccessibility, assertTransferBudget, assertOriginalPrefetchCapture, attachJson, focusWithTab, showMobileMenu } from './support.mjs';

const inventory = loadInventory();
const currentRoutes = new Set(inventory.pages.map((entry) => entry.route));
const historicalRoutes = new Set(inventory.historical.map((entry) => entry.route));

for (const entry of inventory.pages) {
  test(`current page: ${entry.route}`, async ({ page, runtime }, testInfo) => {
    await openPage(page, entry.route);
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
    await assertNoOverflow(page);
    await assertTableKeyboardScroll(page, testInfo);
    await scanAccessibility(page, testInfo, 'light-page');
    if (entry.route === '/innovate/') await assertOriginalPrefetchCapture(page, runtime, testInfo);
    await assertTransferBudget(runtime, testInfo);
    await attachJson(testInfo, 'classified-page', entry);
  });
}

for (const route of darkEntryRoutes) {
  test(`dark entry: ${route}`, async ({ page, isMobile, runtime }, testInfo) => {
    await openPage(page, route);
    const menuOpened = await showMobileMenu(page, isMobile);
    const theme = page.locator('starlight-theme-select select:visible').first();
    let themeMethod = 'visible-select';
    if (isMobile && !menuOpened && await theme.count() === 0) {
      // A mobile splash layout exposes no theme picker; qualify the OS preference.
      await page.emulateMedia({ colorScheme: 'dark' });
      themeMethod = 'preferred-color-scheme';
    } else {
      await theme.selectOption('dark');
    }
    await attachJson(testInfo, 'theme-change', { method: themeMethod, menu_opened: menuOpened });
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    if (menuOpened) {
      await scanAccessibility(page, testInfo, 'dark-mobile-menu');
      await page.keyboard.press('Escape');
      await expect(page.locator('#starlight__sidebar')).not.toBeVisible();
    }
    await assertNoOverflow(page);
    await scanAccessibility(page, testInfo, 'dark-page');
    await assertTransferBudget(runtime, testInfo);
  });
}

test('keyboard skip link and visible focus', async ({ page }) => {
  await openPage(page, '/innovate/user-guide/getting-started/');
  await page.keyboard.press('Tab');
  const skip = page.locator('a.sl-skip-link');
  await expect(skip).toBeFocused();
  await expect(skip).toBeInViewport();
  expect(await skip.evaluate((element) => getComputedStyle(element).clip)).toBe('auto');
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/#_top$/u);
  // Native fragment navigation sets the next sequential focus position even
  // when the target heading itself is not focusable.
  await page.keyboard.press('Tab');
  expect(await page.evaluate(() => Boolean(document.activeElement?.closest('main')))).toBe(true);
  const focus = await page.evaluate(() => {
    const element = document.activeElement;
    const style = getComputedStyle(element);
    const box = element.getBoundingClientRect();
    return { visible: box.width > 0 && box.height > 0, outline: style.outlineStyle, outlineWidth: parseFloat(style.outlineWidth), boxShadow: style.boxShadow };
  });
  expect(focus.visible).toBe(true);
  expect((focus.outline !== 'none' && focus.outlineWidth > 0) || focus.boxShadow !== 'none', 'Keyboard focus must have a visible indicator').toBe(true);
});

for (const { query, expectedRoute } of [
  { query: 'native diffusion curve', expectedRoute: '/innovate/user-guide/getting-started/' },
  { query: 'Current Python API', expectedRoute: '/innovate/api/python/' },
]) {
  test(`keyboard Pagefind search: ${query}`, async ({ page, runtime }, testInfo) => {
    await openPage(page, '/innovate/');
    const opener = page.locator('site-search button[data-open-modal]');
    const dialog = page.locator('site-search dialog');
    const input = page.locator('#starlight__search .pagefind-ui__search-input');
    await focusWithTab(page, opener);
    await page.keyboard.press('Enter');
    await expect(dialog).toBeVisible();
    await expect(input).toBeFocused();
    await input.pressSequentially(query);
    const results = page.locator('#starlight__search .pagefind-ui__result-inner > .pagefind-ui__result-title .pagefind-ui__result-link');
    await expect(results.first()).toBeVisible();
    await expect.poll(async () => new URL(await results.first().getAttribute('href'), page.url()).pathname).toBe(expectedRoute);
    const urls = await results.evaluateAll((links) => links.map((link) => new URL(link.href).pathname));
    expect(urls.length).toBeGreaterThan(0);
    expect(urls.filter((url) => !currentRoutes.has(url)), 'Search must return only classified current pages, including no current-path historical material').toEqual([]);
    await attachJson(testInfo, 'pagefind-results', { query, first_result: urls[0], result_routes: urls });
    await assertNoOverflow(page);
    await scanAccessibility(page, testInfo, 'search-dialog');
    await page.keyboard.press('Escape');
    await expect(dialog).not.toBeVisible();
    await expect(opener).toBeFocused();
    await page.keyboard.press('Control+k');
    await expect(dialog).toBeVisible();
    await expect(input).toBeFocused();
    // Pagefind clears the query on Escape; reopening starts a fresh search.
    await expect(input).toHaveValue('');
    await input.pressSequentially(query);
    await expect(results.first()).toBeVisible();
    await expect.poll(async () => new URL(await results.first().getAttribute('href'), page.url()).pathname).toBe(expectedRoute);
    await focusWithTab(page, results.first(), 12);
    await page.keyboard.press('Enter');
    await expect(page).toHaveURL((url) => url.pathname === expectedRoute);
    await expect(page.locator('h1')).toBeVisible();
    await page.waitForLoadState('networkidle');
    await assertTransferBudget(runtime, testInfo, true);
  });
}

test('current navigation and archived version labels', async ({ page, isMobile }, testInfo) => {
  // Both API versions have sidebars; the archived home is a splash page.
  await openPage(page, '/innovate/api/python/');
  await showMobileMenu(page, isMobile);
  const sidebar = page.locator('nav.sidebar');
  await expect(sidebar.locator('a[href="/innovate/user-guide/getting-started/"]')).toHaveAccessibleName(/Quickstart/u);
  await expect(sidebar.locator('a[href="/innovate/api/python/"]')).toHaveAccessibleName(/Python API/u);
  const navigation = await sidebar.locator('.top-level a[href]').evaluateAll((links) => links.map((link) => ({
    route: new URL(link.href).pathname,
    group: link.closest('details')?.querySelector(':scope > summary .large')?.textContent.trim() ?? '',
  })));
  await attachJson(testInfo, 'primary-navigation', navigation);
  expect(navigation.filter((entry) => historicalRoutes.has(entry.route)), 'Historical pages must be reached through the historical map, not mixed into primary navigation').toEqual([]);
  const requiredGroups = {
    'User Guide': ['/innovate/core/', '/innovate/user-guide/fitting/', '/innovate/user-guide/forecasting/', '/innovate/user-guide/backends/'],
    'Tutorials': ['/innovate/tutorials/', '/innovate/tutorials/scenario-experiment-workflows/', '/innovate/tutorials/norton-bass/', '/innovate/tutorials/bayesian-fitter/', '/innovate/tutorials/counterfactual-analysis/', '/innovate/tutorials/arrow-interchange/', '/innovate/tutorials/data-ingestion-provenance/'],
    'API Reference': ['/innovate/api/python/', '/innovate/api/generated/julia/', '/innovate/api/generated/r/', '/innovate/bindings/', '/innovate/core/kernel/', '/innovate/core/arrow-interchange/'],
    'Historical Reference': ['/innovate/historical-documentation/'],
  };
  for (const [group, routes] of Object.entries(requiredGroups)) {
    for (const route of routes) {
      expect(navigation.filter((entry) => entry.route === route), `${route} must appear once under ${group}`).toEqual([{ route, group }]);
    }
  }
  const version = page.locator('starlight-version-select select:visible').first();
  await expect(version.locator('option:checked')).toHaveText('0.7 Development');
  await expect(version.locator('option')).toHaveText(['0.7 Development', '0.6 Reference']);
  if (isMobile) await scanAccessibility(page, testInfo, 'light-mobile-menu');
  await version.selectOption({ label: '0.6 Reference' });
  await expect(page).toHaveURL((url) => url.pathname === '/innovate/latest/api/python/');
  await expect(page.getByRole('main')).toContainText('0.6 Reference');
  await page.waitForLoadState('networkidle');
  await assertNoOverflow(page);
  await showMobileMenu(page, isMobile);
  const archived = page.locator('starlight-version-select select:visible').first();
  await expect(archived.locator('option:checked')).toHaveText('0.6 Reference');
  await archived.selectOption({ label: '0.7 Development' });
  await expect(page).toHaveURL((url) => url.pathname === '/innovate/api/python/');
  await page.waitForLoadState('networkidle');
  await openPage(page, '/innovate/migration/');
  await expect(page.getByRole('main')).toContainText('Pre-0.7 reference');
  await expect(page.getByRole('main').getByRole('link', { name: 'native migration guide', exact: true })).toHaveAttribute('href', '/innovate/native-migration/');
});

test('deterministic artifact budgets', async ({}, testInfo) => {
  const measurement = measureArtifact(path.join(siteRoot, 'dist'), inventory.sitePolicy.max_html_bytes);
  expect(measurement.total_bytes).toBeLessThanOrEqual(budgets.artifactBytes);
  await attachJson(testInfo, 'artifact-budget', measurement);
});
