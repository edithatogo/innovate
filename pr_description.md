🧹 Refactor `compose_regime_ensemble` parameter list

🎯 **What:** Introduced a `RegimeEnsembleConfig` dataclass to encapsulate the parameters for `compose_regime_ensemble` and updated callers.

💡 **Why:** The `compose_regime_ensemble` function had too many parameters (6 parameters). Wrapping these in a configuration object reduces cognitive load, improves the readability of the function signature, and enhances maintainability when adding new properties in the future.

✅ **Verification:** Ran formatting (`ruff format`), linting (`ruff check`), and the full test suite (`pytest`) confirming the workflow operates correctly and cleanly. Checked that `compose_regime_ensemble` results and inputs remain unchanged.

✨ **Result:** The code is easier to maintain and read, successfully reducing function parameter complexity and passing all tests without breaking any functionality.
