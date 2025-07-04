# Contributing to Heartflow

We welcome contributions from the community! Whether you're fixing a bug, adding a new feature, or improving documentation, your help is greatly appreciated.

## Getting Started

1.  Fork the repository on GitHub.
2.  Clone your fork locally:
    ```bash
    git clone https://github.com/your-username/heartflow.git
    cd heartflow
    ```
3.  Install the project in development mode:
    ```bash
    pip install -e .[dev]
    ```
4.  Create a new branch for your changes:
    ```bash
    git checkout -b my-feature-branch
    ```

## Making Changes

1.  Make your changes to the codebase.
2.  Ensure your code follows the project's style guidelines by running `black .` and `flake8`.
3.  Add or update tests for your changes in the `tests/` directory.
4.  Run the full test suite to ensure everything is working correctly:
    ```bash
    pytest
    ```

## Submitting Your Contribution

1.  Commit your changes with a clear and descriptive commit message.
2.  Push your branch to your fork on GitHub:
    ```bash
    git push origin my-feature-branch
    ```
3.  Open a pull request from your branch to the `main` branch of the original repository.
4.  In the pull request description, please provide details about the changes you've made.

Thank you for contributing to Heartflow!
