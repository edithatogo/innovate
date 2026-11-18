# Browser execution blocked by the environment

The full suite discovered **164 cases**. No browser connected, no page interaction completed, and no browser qualification passed.

## Distinct attempts

1. **Managed Chromium installation:** the saved installation log records five failed downloads of Chrome for Testing 159.0.8154.6 (Playwright Chromium v1255). ZIP extraction failed with “End of central directory record signature not found. Either not a zip file, or file is truncated.” Root separately reported observing HTML bytes with curl; no saved response artifact was supplied for that observation.
2. **Initial preview setup:** Astro 7 detected the agent and started a background daemon; Playwright exited before any case with “Process from config.webServer exited early.” The saved qualification receipt records 0 of 164 cases.
3. **Foreground correction and system-browser attempt:** the authorised one-line change appended `--ignore-lock` to `pnpm exec astro preview --host 127.0.0.1 --port 4321`. Playwright still rejects an existing server, and all resource, artifact and error gates remain unchanged. The foreground preview worked. The installed `/usr/bin/chromium` reports version **156.0.8078.79**, but every attempted browser launch failed before connection:

```text
FATAL:chrome/browser/process_singleton_posix.cc:297 Check failed: . socket() failed: Operation not permitted (1)
```

Crashpad also reported `ptrace: Operation not permitted (1)`. The run was interrupted after **64 completed setup-failure case results**, each with a saved trace and error context showing the same socket error. The remaining **100 selected cases have no reported result**; they are not counted as test failures or skips. The precise in-progress interrupted count is unknown because the runner did not finalize its JSON report. `qualification.json` and `playwright.json` therefore still describe the earlier zero-case preview failure. No restricted workaround was attempted.

## Static receipt remains current

The production artifact receipt was refreshed after the harness correction: **211 HTML pages, 11,360 local links, zero errors**. Every built artifact file hash stayed identical; only `docs/astro-site/playwright.config.mjs` changed in the source inventory. No HTML rebuild was needed. This static result does not establish browser accessibility or interaction behaviour.

The companion JSON records exact file paths, SHA-256 hashes, source/artifact identities and the failure-count limitations. `launch-failure-artifacts.json` binds all 64 saved traces and 64 error contexts. The next browser qualification is the full hosted PR run using the pinned managed browser in a browser-capable environment.
