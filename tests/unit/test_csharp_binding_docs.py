"""Tests for C# binding documentation and package scaffolding."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree


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
    assert "NuGet publication" in docs
    assert "contentFiles/any/any/innovate/kernel_bridge.py" in docs


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
    assert "<TargetFrameworks Condition=\"'$(TargetFramework)' == ''\">net10.0;net11.0</TargetFrameworks>" in project
    assert "<PackageId>innovate.cs</PackageId>" in project


def test_csharp_package_metadata_is_nuget_ready() -> None:
    """The C# package should carry registry metadata and packaged bridge content."""
    project_path = Path("bindings/csharp/Innovate.Kernel/Innovate.Kernel.csproj")
    project = ElementTree.parse(project_path)
    properties = {child.tag: child.text for group in project.findall("PropertyGroup") for child in group if child.text}

    assert properties["PackageId"] == "innovate.cs"
    assert properties["PackageProjectUrl"] == "https://github.com/edithatogo/innovate"
    assert properties["RepositoryUrl"] == "https://github.com/edithatogo/innovate"
    assert properties["RepositoryType"] == "git"
    assert properties["PackageLicenseExpression"] == "Apache-2.0"
    assert properties["PackageReadmeFile"] == "README.md"
    assert properties["PackageTags"] == "innovate;health-economics;decision-analysis;kernel;bindings"
    assert properties["PackageReleaseNotes"].startswith("Initial provisional .NET 10 and .NET 11 binding")
    assert properties["PublishRepositoryUrl"] == "true"
    assert properties["EmbedUntrackedSources"] == "true"
    assert properties["IncludeSymbols"] == "true"
    assert properties["SymbolPackageFormat"] == "snupkg"

    none_items = {
        item.attrib["Include"]: item.attrib for group in project.findall("ItemGroup") for item in group.findall("None")
    }
    assert none_items["../README.md"]["Pack"] == "true"
    assert none_items["../README.md"]["PackagePath"] == "/"
    bridge_item = none_items["../inst/python/kernel_bridge.py"]
    assert bridge_item["Pack"] == "true"
    assert bridge_item["PackagePath"] == "contentFiles/any/any/innovate/kernel_bridge.py"
    assert bridge_item["PackageCopyToOutput"] == "true"

    package_refs = {
        item.attrib["Include"]: item.attrib
        for group in project.findall("ItemGroup")
        for item in group.findall("PackageReference")
    }
    assert package_refs["Microsoft.SourceLink.GitHub"]["PrivateAssets"] == "All"
