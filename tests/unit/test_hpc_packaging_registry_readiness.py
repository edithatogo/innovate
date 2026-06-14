"""Checks for the HPC packaging and registry readiness dossier."""

from __future__ import annotations

from pathlib import Path

DOC = Path("docs/source/hpc_packaging_registry_readiness.rst")
INDEX = Path("docs/source/index.rst")


def _doc_text() -> str:
    return DOC.read_text()


def test_hpc_packaging_readiness_dossier_is_in_sphinx_navigation() -> None:
    """The HPC packaging dossier should be reachable from the docs site."""
    index = INDEX.read_text()

    assert DOC.is_file()
    assert "hpc_packaging_registry_readiness" in index
    assert "hpc_submission_workflow" in index


def test_hpc_packaging_readiness_maps_install_surfaces_and_dependencies() -> None:
    """The dossier should enumerate package surfaces, variants, and dependencies."""
    doc = _doc_text()

    for phrase in (
        "Python package surface",
        "Rust crate and native slices",
        "Optional JAX/XLA extras",
        "Language binding surfaces",
        "CPU-only deployment",
        "GPU/XLA deployment",
        "Mixed Rust/Python bridge deployment",
        'variant("+rust"',
        'variant("+jax"',
        'variant("+bindings"',
        'variant("+docs"',
    ):
        assert phrase in doc

    for dependency in (
        "py-numpy",
        "py-scipy",
        "py-pandas",
        "py-pyarrow",
        "py-statsmodels",
        "py-mesa",
        "py-networkx",
        "py-ndlib",
        "py-jitcdde",
        "py-sympy",
        "py-ruptures",
        "py-pymannkendall",
        "py-pytensor",
        "py-typing-extensions",
        "py-jax",
        "py-jaxlib",
        "cargo",
        "rust",
    ):
        assert dependency in doc


def test_hpc_packaging_readiness_captures_spack_and_easybuild_prototypes() -> None:
    """Packaging candidates should include install and smoke-test evidence."""
    doc = _doc_text()

    for phrase in (
        "Spack package candidate",
        "class PyInnovate(PythonPackage):",
        "EasyBuild easyconfig candidate",
        "easyblock = 'PythonPackage'",
        "Slurm and PBS job templates",
        "HPSF and E4S evidence templates",
        "per-target command checklist",
        "module sanity checks",
        'python -c "import innovate; print(innovate.__version__)"',
        "python -m pip check",
        "cargo test --manifest-path bindings/rust/Cargo.toml",
        "julia --project=bindings/julia -e",
        "Rscript -e",
        "npm test --prefix bindings/typescript",
        "spack-batch.log",
        "easybuild-batch.log",
        "spack-pbs.log",
        "easybuild-pbs.log",
    ):
        assert phrase in doc


def test_hpc_registry_claims_are_gated_until_evidence_exists() -> None:
    """HPSF/E4S claims should stay gated by explicit evidence requirements."""
    doc = _doc_text()

    for phrase in (
        "HPSF candidacy",
        "E4S candidacy",
        "evidence/hpsf-review-note.md",
        "evidence/e4s-review-note.md",
        "install, smoke, and batch evidence is now present",
        "package sketches, local evidence, and batch logs are present",
        "performance portability evidence",
        "Slurm or PBS",
        "execution templates for Slurm and PBS scheduler submission",
        "CPU, GPU, and mixed bridge",
        "no HPSF or E4S submission should be made",
        "maintainer handoff note",
        "ready_for_maintainer",
    ):
        assert phrase in doc

    assert "durable HPSF blocker note" not in doc
    assert "durable E4S blocker note" not in doc
    assert "preserve the blocker note" not in doc
