import { test, expect, openPage, assertNoOverflow, scanAccessibility, attachJson } from './support.mjs';

// This is an automated reflow/resize proxy, not a certification of WCAG or
// a replacement for browser zoom, AT and human-reader evaluation.
for (const route of [
  '/innovate/',
  '/innovate/native-migration/',
  '/innovate/tutorials/public-data-fit/',
]) {
  test(`reflow at 320 CSS pixels: ${route}`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 320, height: 568 });
    await openPage(page, route);
    await assertNoOverflow(page);
    await expect(page.getByRole('main')).toBeVisible();
    const geometry = await page.evaluate(() => ({
      viewport: document.documentElement.clientWidth,
      document: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
      mainVisible: Boolean(document.querySelector('main')?.getBoundingClientRect().width),
    }));
    expect(geometry.viewport).toBe(320);
    expect(geometry.mainVisible).toBe(true);
    await scanAccessibility(page, testInfo, 'reflow-320-css-px');
    await attachJson(testInfo, 'reflow-observation', {
      route,
      geometry,
      criterion: 'WCAG 2.2 SC 1.4.10 reflow width proxy',
      browser_zoom_executed: false,
      human_at_evaluation: false,
    });
  });
}

test('enlarged-root-text proxy retains the public-data lesson', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 640, height: 768 });
  await openPage(page, '/innovate/tutorials/public-data-fit/');
  const before = await page.evaluate(() => parseFloat(getComputedStyle(document.documentElement).fontSize));
  await page.evaluate(() => { document.documentElement.style.fontSize = '200%'; });
  const after = await page.evaluate(() => parseFloat(getComputedStyle(document.documentElement).fontSize));
  expect(after, 'Root text scale should genuinely increase').toBeGreaterThanOrEqual(before * 1.9);
  await expect(page.getByRole('main')).toBeVisible();
  await expect(page.locator('h1')).toBeVisible();
  await assertNoOverflow(page);
  await scanAccessibility(page, testInfo, 'root-text-200-percent-proxy');
  await attachJson(testInfo, 'text-enlargement-observation', {
    before_px: before,
    after_px: after,
    viewport_css_px: { width: 640, height: 768 },
    criterion: 'WCAG 2.2 SC 1.4.4 text-scale proxy',
    actual_browser_zoom_executed: false,
    human_screen_reader_review: false,
  });
});
