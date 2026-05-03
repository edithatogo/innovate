"""Tests for C# binding documentation and package scaffolding."""

from __future__ import annotations

from pathlib import Path


def test_csharp_binding_documentation_is_present() -> None:
    """The planned C# binding should have an explicit contract document."""
    docs_root = Path("docs/source")

    assert (docs_root / "tutorials/csharp_bindings.rst").is_file()


def test_csharp_binding_docs_define_thin_contract_scope() -> None:
    """The C# binding guidance should align with the shared kernel contract."""
    docs = Path("docs/source/tutorials/csharp_bindings.rst").read_text()

    for phrase in (
        "thin adapter",
        "schema compatibility",
        "KernelRequest",
        "KernelResponse",
        "discover_models",
        "does not reimplement model behavior",
    ):
        assert phrase in docs

    assert ".NET 10 and" in docs
    assert ".NET 11 SDK" in docs
    assert "INNOVATE_PYTHON_COMMAND" in docs


def test_bindings_hub_includes_csharp_plan() -> None:
    """The bindings hub should surface C# as a planned binding target."""
    hub = Path("docs/source/bindings.rst").read_text()

    assert "C#" in hub
    assert "tutorials/csharp_bindings" in hub


def test_csharp_binding_package_scaffold_exists() -> None:
    """The C# binding should have a buildable .NET package scaffold."""
    binding_root = Path("bindings/csharp")

    assert (binding_root / "Innovate.Kernel/Innovate.Kernel.csproj").is_file()
    assert (binding_root / "Innovate.Kernel.Tests/Innovate.Kernel.Tests.csproj").is_file()
    assert (binding_root / "inst/python/kernel_bridge.py").is_file()

    project = (binding_root / "Innovate.Kernel/Innovate.Kernel.csproj").read_text()
    assert "<TargetFrameworks>net10.0;net11.0</TargetFrameworks>" in project
    assert "<PackageId>Innovate.Kernel</PackageId>" in project
