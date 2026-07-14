# Security Policy

## Supported Versions

The following versions of `innovate` are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| < 0.5   | :x:                |

## Reporting a Vulnerability

We take the security of `innovate` seriously. If you believe you have found a security vulnerability, please report it to us as described below.

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

## Security Measures

- Dependencies are scanned for vulnerabilities using `safety` and `bandit`
- All dependencies are managed via `uv` with a locked `uv.lock` file
- Renovate manages dependency update PRs, groups non-major updates by ecosystem,
  and can automerge vulnerability fixes only after the required CI and branch
  protection gates pass. GitHub Dependabot security updates remain enabled as
  an additional vulnerability-alert source; scheduled version updates are
  owned by Renovate to avoid duplicate PRs.
- CI pipeline includes security scanning on every push
