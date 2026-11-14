import { defineConfig } from '@playwright/test';
import { browserOrigin, siteRoot } from './scripts/browser-contract.mjs';

export default defineConfig({
  testDir: './tests/browser',
  testMatch: '**/*.spec.mjs',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: true,
  timeout: 90_000,
  globalTimeout: 20 * 60_000,
  expect: { timeout: 15_000 },
  outputDir: './.reports/browser/test-results',
  reporter: [
    ['list'],
    ['json', { outputFile: '.reports/browser/playwright.json' }],
    ['html', { outputFolder: '.reports/browser/html', open: 'never' }],
    ['./scripts/browser-report.mjs'],
  ],
  use: {
    baseURL: `${browserOrigin}/innovate/`,
    browserName: 'chromium',
    locale: 'en-AU',
    timezoneId: 'UTC',
    colorScheme: 'light',
    reducedMotion: 'reduce',
    serviceWorkers: 'block',
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'chromium-desktop', use: { viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 } },
    { name: 'chromium-mobile-390', use: { viewport: { width: 390, height: 844 }, deviceScaleFactor: 1, isMobile: true, hasTouch: true } },
  ],
  // Preview the built artifact: Astro dev deliberately does not serve Pagefind.
  // Reject an existing server to avoid qualifying another build accidentally.
  webServer: {
    command: 'pnpm exec astro preview --host 127.0.0.1 --port 4321',
    cwd: siteRoot,
    url: `${browserOrigin}/innovate/`,
    reuseExistingServer: false,
    timeout: 60_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
