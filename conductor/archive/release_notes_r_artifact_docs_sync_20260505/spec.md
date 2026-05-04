# Specification: Release Notes and R Artifact Documentation Synchronization

## Problem

The repository has multiple release-note mechanisms and package publication
documents. After the R PDF manual and vignette work, release-facing documents
still contained stale statements about missing R vignettes, while
`CHANGELOG.md` did not cover the current aligned `0.5.0` package version. The
relationship between Release Please, Release Drafter, Commitizen, and
`CHANGELOG.md` was also not documented.

## Goals

- Document the source of truth and responsibilities for release notes.
- Bring `CHANGELOG.md` up to the current aligned package version.
- Correct R publication docs so they match the current source vignette and PDF
  manual artifact behavior.
- Add static tests so the release-note policy and R artifact documentation do
  not silently drift again.
- Normalize completed Conductor archive status text discovered during the
  six-agent governance audit.

## Non-Goals

- Publishing a release.
- Replacing Release Please, Release Drafter, or Commitizen.
- Reconstructing a full historical changelog from every past commit.

## Acceptance Criteria

- `CHANGELOG.md` contains `0.4.0` and `0.5.0` sections.
- A Sphinx release-notes policy page documents Release Please, Release Drafter,
  Commitizen, and `CHANGELOG.md` responsibilities.
- Release Drafter configuration links to the policy instead of saying the
  policy is undefined.
- Binding publication docs say the R package has a source vignette and uploads
  a versioned R manual artifact.
- Static tests enforce the policy page, changelog coverage, and R artifact
  documentation.
- Completed Conductor archive index status blocks match completed metadata.
