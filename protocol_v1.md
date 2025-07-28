# Innovate Development Protocol v1

## 1. Introduction

This document outlines the development protocol for the `innovate` library. The purpose of this protocol is to ensure that the development process is consistent, transparent, and results in high-quality, well-documented, and maintainable software.

## 2. Version Control

### 2.1. Git and GitHub

All code and documentation for the `innovate` library will be managed using Git and hosted on GitHub. The `work` branch will be protected and will always represent the latest stable release. All new development will be done on feature branches.

### 2.2. Branching Strategy

A feature-based branching strategy will be used. Each new feature or bug fix will be developed in its own branch, named according to the following convention:

-   `feature/<feature-name>` for new features.
-   `bugfix/<bug-name>` for bug fixes.
-   `release/<version-number>` for release preparation.

### 2.3. Commits

Commit messages will follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. This will allow for automated changelog generation and a more readable commit history.

### 2.4. Pull Requests

All changes to the `work` branch will be made through pull requests (PRs). Each PR must be reviewed and approved by at least one other developer before it can be merged. All PRs must include a clear description of the changes and a link to the relevant issue in the issue tracker.

## 3. Testing

### 3.1. Test-Driven Development

Test-Driven Development (TDD) will be used for all new features. This means that tests will be written before the code that implements the feature. This will ensure that all new code is tested and that the tests accurately reflect the requirements of the feature.

### 3.2. Test Suite

The `pytest` framework will be used for all tests. The test suite will be run automatically on every commit to a feature branch and on every PR. The test suite must pass before a PR can be merged.

### 3.3. Code Coverage

Code coverage will be measured using the `coverage.py` tool. The goal is to maintain a high level of code coverage (at least 90%) for all new code.

## 4. Documentation

### 4.1. Sphinx

All documentation will be written in reStructuredText and built using Sphinx. The documentation will be hosted on Read the Docs and will be updated automatically on every commit to the `work` branch.

### 4.2. Docstrings

All modules, classes, and functions will have docstrings that follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). This will ensure that the documentation is consistent and easy to read.

### 4.3. Tutorials and Examples

Tutorials and examples will be provided in the form of Jupyter notebooks. These notebooks will be tested as part of the documentation build process to ensure that they are always up-to-date.

## 5. Release Management

### 5.1. Semantic Versioning

The `innovate` library will follow the [Semantic Versioning](https://semver.org/) specification. This means that version numbers will be in the format `MAJOR.MINOR.PATCH`.

### 5.2. Release Process

When a new version of the library is ready to be released, a release branch will be created from the `work` branch. The version number will be updated in the `pyproject.toml` and `conda.recipe/meta.yaml` files, and the `CHANGELOG.md` will be updated with the changes in the new release. The release branch will then be merged into the `work` branch, and a new release will be created on GitHub. The package will then be uploaded to PyPI and TestPyPI.
