# Innovate Development Protocol v2

## 1. Introduction

This document outlines the development protocol for the `innovate` library. The `innovate` library is a Python package for simplifying innovation and policy diffusion modeling. The goal of this protocol is to ensure that the development process is consistent, transparent, and results in high-quality, well-documented, and maintainable software.

## 2. Version Control

### 2.1. Git and GitHub

All code and documentation for the `innovate` library will be managed using Git and hosted on GitHub. The `main` branch will be protected and will always represent the latest stable release. All new development will be done on feature branches.

### 2.2. Branching Strategy

A feature-based branching strategy will be used. Each new feature or bug fix will be developed in its own branch, named according to the following convention:

-   `feature/<feature-name>` for new features.
-   `bugfix/<bug-name>` for bug fixes.
-   `release/<version-number>` for release preparation.
-   `hotfix/<issue-number>` for critical bug fixes that need to be released immediately.

### 2.3. Commits

Commit messages will follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. A good commit message should be a short, descriptive summary of the change, followed by a more detailed explanation of the change and the reasoning behind it. For more information on how to write good commit messages, see the [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/) guide.

### 2.4. Pull Requests

All changes to the `main` branch will be made through pull requests (PRs). Each PR must be reviewed and approved by at least one other developer before it can be merged. All PRs must include a clear description of the changes and a link to the relevant issue in the issue tracker.

## 3. Testing

### 3.1. Test-Driven Development

Test-Driven Development (TDD) will be used for all new features. This means that tests will be written before the code that implements the feature. The TDD process will be as follows:

1.  Write a failing test for the new feature.
2.  Write the minimum amount of code required to make the test pass.
3.  Refactor the code to improve its design.

### 3.2. Test Suite

The `pytest` framework will be used for all tests. The test suite will be run automatically on every commit to a feature branch and on every PR. The test suite must pass before a PR can be merged.

### 3.3. Code Coverage

Code coverage will be measured using the `coverage.py` tool. The goal is to maintain a high level of code coverage (at least 90%) for all new code. In cases where it is not possible to achieve 90% code coverage (e.g., for code that is difficult to test), the reasons for this will be documented in the code.

## 4. Documentation

### 4.1. Sphinx

All documentation will be written in reStructuredText and built using Sphinx. The documentation will be hosted on Read the Docs and will be updated automatically on every commit to the `main` branch. The documentation build process will be as follows:

1.  The documentation will be built automatically on every commit to a feature branch and on every PR.
2.  The documentation build must pass before a PR can be merged.

### 4.2. Docstrings

All modules, classes, and functions will have docstrings that follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). For more information on how to write good docstrings, see the [Epydoc](http://epydoc.sourceforge.net/manual-usage.html) guide.

### 4.3. Tutorials and Examples

Tutorials and examples will be provided in the form of Jupyter notebooks. These notebooks will be tested as part of the documentation build process to ensure that they are always up-to-date.

## 5. Release Management

### 5.1. Semantic Versioning

The `innovate` library will follow the [Semantic Versioning](https://semver.org/) specification. This means that version numbers will be in the format `MAJOR.MINOR.PATCH`. Pre-releases will be indicated by appending a hyphen and a series of dot-separated identifiers immediately following the patch version (e.g., `1.0.0-alpha.1`).

### 5.2. Release Process

When a new version of the library is ready to be released, a release branch will be created from the `main` branch. The version number will be updated in the `pyproject.toml` and `conda.recipe/meta.yaml` files, and the `CHANGELOG.md` will be updated with the changes in the new release. The release branch will then be merged into the `main` branch, and a new release will be created on GitHub. The package will then be uploaded to PyPI and TestPyPI.

### 5.3. Rollbacks

In the event that a release needs to be rolled back, a new release will be created with the previous version number. The `CHANGELOG.md` will be updated to indicate that the release has been rolled back, and the package will be uploaded to PyPI and TestPyPI.
