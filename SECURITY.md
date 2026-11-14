# Security Policy

## Development and support scope

Active development is the Rust-native 0.7 prerelease line on `main`; the package
manifests record its exact version. It is undergoing scientific and distribution
qualification and is not a claim of stable-release or production readiness.
See [README.md](README.md), the [migration ledger](migration-ledger.json) and
[release gates](conductor/release-gates-20261107.md).

Historical release records remain available for provenance. This development
branch does not establish a current security-support or backport commitment for
every earlier release. Report the exact affected version or commit so the
maintainer can assess exposure and any required backport.

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to [dylan.mordaunt@vuw.ac.nz](mailto:dylan.mordaunt@vuw.ac.nz).

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker could exploit it

## Preferred Languages

We prefer all communications to be in English.

## Configured security controls

[Security and Quality](.github/workflows/security-quality.yml) runs on pull
requests and pushes targeting `main` or `work`, a weekly schedule and manual
dispatch. It configures dependency review for pull requests, a RustSec audit of
`Cargo.lock`, Gitleaks history scanning, workflow/policy lint and Python binding
checks including Bandit. Bandit checks source code; it is not a dependency
vulnerability audit. Installing `pip-audit` as a development tool does not prove
that an audit has run.

The separate [Privacy and secrets](.github/workflows/privacy-secrets.yml) workflow
uses the pinned shared Presidio/Gitleaks action. [OpenSSF
Scorecard](.github/workflows/scorecard.yml) records repository posture. These
workflow definitions describe configured controls; only completed runs on the
relevant source establish their outcomes. A queued offline-security job or a
draft rollout is not completed security evidence.

Python dependencies use `uv.lock`, Rust uses `Cargo.lock` with explicit upstream
revisions, and documentation dependencies use the documentation workspace lock.
The hosted Renovate app owns scheduled version updates through
[its repository configuration](.github/renovate.json). Confirm an active Renovate
dashboard or update pull request before retiring a Dependabot version-update
schedule. Dependabot alerts and security-update settings are separate GitHub
controls and must be verified independently.

Required automated checks gate routine merges. Keep zero mandatory human,
code-owner or team approvals and retain owner recovery for this solo-maintainer
repository. Do not weaken checks, grant bots broad bypasses or treat a missing
check as successful. See [AGENTS.md](AGENTS.md) for ownership and execution rules.

Run `python scripts/check_repository_policy.py` from the repository root for the
dependency-free policy/drift check. Its result is scoped to the checks reported;
it does not establish live enforcement, scanner success, external coverage
delivery or scientific release acceptance. [CONTRIBUTING.md](CONTRIBUTING.md)
lists the separate development and migration gates.
