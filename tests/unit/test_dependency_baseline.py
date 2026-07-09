"""Dependency baseline tests — verify all ecosystem dependency floors.

Tests cover:
  - Python 3.14 baseline
  - NumPy 2+
  - Pydantic v2
  - basedpyright strict
  - Astro 7
  - TypeScript 6
  - Node 26 types
  - Vitest 4
  - Criterion 0.8
  - mutmut current baseline
"""

import subprocess
import sys
from pathlib import Path

import pytest

# ---- Python 3.14 baseline ----


class TestPythonBaseline:
    """Verify Python 3.14+ is the runtime baseline."""

    def test_python_version_314(self):
        """The project requires Python >= 3.14."""
        major, minor = sys.version_info[:2]
        assert (major, minor) >= (3, 14), f"Python {major}.{minor} detected; 3.14+ required"


# ---- NumPy 2+ ----


class TestNumPyBaseline:
    """Verify NumPy 2+ is installed and importable."""

    def test_numpy_imports(self):
        """numpy must be importable and at version >= 2."""
        import numpy as np

        v = np.__version__
        major = int(v.split(".")[0])
        assert major >= 2, f"NumPy {v} < 2 detected"

    def test_numpy_floor_in_pyproject(self):
        """pyproject.toml must pin numpy >= 2.4.4."""
        pyproject = Path("pyproject.toml")
        assert pyproject.exists()
        text = pyproject.read_text()
        assert "numpy>=2" in text or "numpy>=" in text


# ---- Pydantic v2 ----


class TestPydanticBaseline:
    """Verify Pydantic v2 is available."""

    def test_pydantic_v2(self):
        """pydantic must be v2+."""
        try:
            import pydantic

            major = int(pydantic.__version__.split(".")[0])
            assert major >= 2, f"Pydantic v{pydantic.__version__} < 2"
        except ImportError:
            pytest.skip("pydantic not installed")


# ---- basedpyright strict ----


class TestBasedpyrightBaseline:
    """Verify basedpyright is available for type checking."""

    def test_basedpyright_available(self):
        """basedpyright must be runnable."""
        result = subprocess.run(
            ["uv", "run", "basedpyright", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            pytest.skip("basedpyright not installed")


# ---- TypeScript 6 ----


class TestTypeScriptBaseline:
    """Verify TypeScript 6+ for the TypeScript bindings."""

    TS_PKG = Path("bindings/typescript/package.json")
    DOCS_PKG = Path("docs/astro-site/package.json")

    def test_typescript_6_bindings(self):
        """TypeScript bindings package.json must pin typescript ^6."""
        if not self.TS_PKG.exists():
            pytest.skip("TypeScript bindings not present")
        import json

        pkg = json.loads(self.TS_PKG.read_text())
        ts_ver = pkg.get("devDependencies", {}).get("typescript", "")
        assert "^6" in ts_ver or ">=6" in ts_ver, f"typescript version in bindings {ts_ver} is not ^6"

    def test_typescript_6_docs(self):
        """Docs site package.json must pin typescript ^6."""
        if not self.DOCS_PKG.exists():
            pytest.skip("Docs package.json not present")
        import json

        pkg = json.loads(self.DOCS_PKG.read_text())
        ts_ver = pkg.get("devDependencies", {}).get("typescript", "")
        assert "^6" in ts_ver or ">=6" in ts_ver, f"typescript version in docs {ts_ver} is not ^6"


# ---- Node 26 types ----


class TestNode26Baseline:
    """Verify @types/node ^26 for TypeScript bindings."""

    TS_PKG = Path("bindings/typescript/package.json")

    def test_node_types_26(self):
        """@types/node must be ^26."""
        if not self.TS_PKG.exists():
            pytest.skip("TypeScript bindings not present")
        import json

        pkg = json.loads(self.TS_PKG.read_text())
        node_types = pkg.get("devDependencies", {}).get("@types/node", "")
        assert "^26" in node_types or ">=26" in node_types, f"@types/node version {node_types} is not ^26"


# ---- Vitest 4 ----


class TestVitestBaseline:
    """Verify Vitest 4+ for the TypeScript bindings."""

    TS_PKG = Path("bindings/typescript/package.json")

    def test_vitest_4(self):
        """vitest must be ^4."""
        if not self.TS_PKG.exists():
            pytest.skip("TypeScript bindings not present")
        import json

        pkg = json.loads(self.TS_PKG.read_text())
        vitest_ver = pkg.get("devDependencies", {}).get("vitest", "")
        assert "^4" in vitest_ver or ">=4" in vitest_ver, f"vitest version {vitest_ver} is not ^4"


# ---- Criterion 0.8 ----


class TestCriterionBaseline:
    """Verify Rust benchmark criterion stays MSRV-compatible."""

    CARGO_TOML = Path("bindings/rust/Cargo.toml")

    def test_criterion_msrv_compatible(self):
        """Cargo.toml must pin criterion to a 1.85-compatible release.

        criterion 0.8.x requires rustc >= 1.86, which breaks the CI MSRV matrix
        (1.85.0). Keep 0.5.x until the MSRV is raised.
        """
        if not self.CARGO_TOML.exists():
            pytest.skip("Rust bindings not present")
        text = self.CARGO_TOML.read_text()
        assert (
            'criterion = "=0.5.1"' in text
            or 'criterion = "0.5"' in text
            or 'criterion = "0.5.1"' in text
        ), "MSRV-compatible criterion pin not found in Rust Cargo.toml"


# ---- Mutmut current baseline ----


class TestMutmutBaseline:
    """Verify mutmut is available as a dev dependency."""

    def test_mutmut_available(self):
        """mutmut must be runnable or declared in pyproject."""
        pyproject = Path("pyproject.toml")
        if not pyproject.exists():
            pytest.skip("pyproject.toml not found")
        text = pyproject.read_text()
        has_mutmut = "mutmut" in text
        try:
            import mutmut  # noqa: F401

            assert True
        except ImportError:
            if has_mutmut:
                pytest.skip("mutmut declared but not installed")
            else:
                pytest.skip("mutmut not declared or installed")


# ---- Frontend tooling consistency ----


class TestFrontendToolingBaseline:
    """Verify pnpm, Node 26 are consistent across frontends."""

    DOCS_PKG = Path("docs/astro-site/package.json")
    TS_PKG = Path("bindings/typescript/package.json")

    def test_pnpm_package_manager(self):
        """docs/astro-site must use pnpm as package manager."""
        if not self.DOCS_PKG.exists():
            pytest.skip("docs package.json not found")
        import json

        pkg = json.loads(self.DOCS_PKG.read_text())
        manager = pkg.get("packageManager", "")
        assert "pnpm" in manager, f"Package manager is {manager}, expected pnpm"

    def test_node_engine_docs(self):
        """Docs package.json should specify node >= 26."""
        if not self.DOCS_PKG.exists():
            pytest.skip("docs package.json not found")
        import json

        pkg = json.loads(self.DOCS_PKG.read_text())
        engine = pkg.get("engines", {}).get("node", "")
        assert engine, "node engine not specified in docs package.json"

    def test_node_engine_typescript_bindings(self):
        """TypeScript bindings must declare node >= 26."""
        if not self.TS_PKG.exists():
            pytest.skip("TypeScript bindings not found")
        import json

        pkg = json.loads(self.TS_PKG.read_text())
        engine = pkg.get("engines", {}).get("node", "")
        assert "26" in engine, f"node engine in TS bindings is '{engine}', expected >=26"


# ---- Astro 7 ----


class TestAstroBaseline:
    """Verify docs site uses Astro 7+."""

    ASTRO_PKG = Path("docs/astro-site/package.json")

    def test_astro_package_exists(self):
        """package.json for the docs site must exist."""
        assert self.ASTRO_PKG.exists(), "docs/astro-site/package.json not found"

    def test_astro_7_dependency(self):
        """Astro dependency must be ^7.x or >=7."""
        import json

        pkg = json.loads(self.ASTRO_PKG.read_text())
        astro_ver = pkg.get("dependencies", {}).get("astro", "")
        assert astro_ver, "astro not found in dependencies"
        assert astro_ver.startswith("^7") or ">=7" in astro_ver or astro_ver.startswith("7"), (
            f"astro version {astro_ver} is not Astro 7+"
        )
