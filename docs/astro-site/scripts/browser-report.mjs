import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { captureInputs, caseEvidenceComplete, qualificationCoverage, repositoryRoot, siteRoot, wcagTags, budgets } from './browser-contract.mjs';

export default class BrowserQualificationReporter {
  constructor() {
    this.cases = [];
    this.errors = [];
  }

  onBegin() {
    this.startedAt = new Date().toISOString();
    try {
      this.inputs = captureInputs();
      this.commit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repositoryRoot, encoding: 'utf8' }).trim();
      this.dirty = Boolean(execFileSync('git', ['status', '--porcelain', '--untracked-files=normal'], { cwd: repositoryRoot, encoding: 'utf8' }).trim());
    } catch (error) {
      this.errors.push(String(error));
    }
  }

  onTestEnd(test, result) {
    const attachments = {};
    for (const attachment of result.attachments) {
      if (attachment.contentType !== 'application/json') continue;
      try {
        const bytes = attachment.body ?? fs.readFileSync(attachment.path);
        attachments[attachment.name] = JSON.parse(bytes.toString());
      } catch (error) {
        this.errors.push(`Unreadable evidence attachment ${attachment.name}: ${error}`);
      }
    }
    this.cases.push({
      project: test.parent.project()?.name,
      title: test.title,
      status: result.status,
      expected_status: test.expectedStatus,
      retry: result.retry,
      duration_ms: result.duration,
      errors: result.errors.map((error) => error.message),
      evidence: attachments,
    });
  }

  onError(error) {
    this.errors.push(error.message ?? String(error));
  }

  async onEnd(result) {
    let finalInputs;
    try {
      finalInputs = captureInputs();
    } catch (error) {
      this.errors.push(String(error));
    }
    const unchanged = Boolean(this.inputs && finalInputs
      && this.inputs.source.sha256 === finalInputs.source.sha256
      && this.inputs.artifact.sha256 === finalInputs.artifact.sha256);
    if (!unchanged) this.errors.push('Source or artifact inputs changed, disappeared, or could not be fingerprinted.');
    const coverage = this.inputs ? qualificationCoverage(this.inputs.inventory, this.cases) : null;
    const complete = coverage && coverage.missing.length === 0 && coverage.unexpected.length === 0
      && coverage.unsuccessful.length === 0 && coverage.duplicate_count === 0
      && this.cases.every((entry) => entry.expected_status === 'passed' && entry.retry === 0 && caseEvidenceComplete(entry));
    if (!complete) this.errors.push('The complete, unfiltered two-viewport qualification did not pass with full evidence and without skips or retries.');
    const passed = result.status === 'passed' && unchanged && complete && this.errors.length === 0;
    const browserOverride = process.env.INNOVATE_DOCS_CHROMIUM_EXECUTABLE || null;
    const browserVersions = [...new Set(this.cases.map((entry) => entry.evidence?.['runtime-observation']?.browser_version)
      .filter((version) => typeof version === 'string'))].sort();
    const report = {
      schema_version: 1,
      status: passed ? 'passed' : 'failed',
      scope: `Automated Chromium qualification of the local built artifact${browserOverride
        ? ' using an explicitly selected executable; this run does not qualify the Playwright-managed CI browser'
        : ' using the Playwright-managed browser'}; no manual usability, assistive-technology, scientific, or live-deployment certification.`,
      started_at: this.startedAt,
      finished_at: new Date().toISOString(),
      git_commit: this.commit,
      worktree_dirty: this.dirty,
      github_run_id: process.env.GITHUB_RUN_ID ?? null,
      github_run_attempt: process.env.GITHUB_RUN_ATTEMPT ?? null,
      node_version: process.version,
      browser_executable_override: browserOverride,
      observed_browser_versions: browserVersions,
      playwright_managed_browser_qualified: passed && browserOverride === null,
      wcag_tags: wcagTags,
      budgets,
      coverage,
      source: this.inputs?.source,
      artifact: this.inputs?.artifact,
      inputs_unchanged_during_run: unchanged,
      served_resources_match_artifact: passed,
      source_artifact_binding_verified: false,
      fingerprint_scope: 'Source and artifact identities are captured before and after the run; every successful internal response is compared with the artifact bytes. Build-chain provenance must be established by the workflow that built this artifact.',
      manual_accessibility_reviewed: false,
      manual_usability_reviewed: false,
      wcag_conformance_certified: false,
      live_deployment_verified: false,
      errors: this.errors,
      cases: this.cases,
    };
    const output = path.join(siteRoot, '.reports/browser/qualification.json');
    fs.mkdirSync(path.dirname(output), { recursive: true });
    fs.writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
    console.log(`Browser qualification: ${report.status}; ${coverage?.observed ?? 0}/${coverage?.expected ?? 0} cases; ${output}`);
    return { status: passed ? 'passed' : 'failed' };
  }
}
