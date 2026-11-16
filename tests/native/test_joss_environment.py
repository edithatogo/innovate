"""Offline setup-guard tests; synthetic Git/tool fixtures do not qualify a render."""

from __future__ import annotations

import importlib.util
import lzma
import os
import sys
from pathlib import Path

import pytest

PACKET = (
    Path(__file__).resolve().parents[2]
    / "conductor/tracks/research_software_registry_readiness_20260721"
    / "evidence/evidence-acquisition-20261116/render"
)


def load_recipe(name):
    spec = importlib.util.spec_from_file_location("joss_" + name, PACKET / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def builder():
    return load_recipe("build_pdf_environment")


@pytest.fixture
def configure():
    return load_recipe("configure_pdf_environment")


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_builder_refuses_existing_root_without_writes(builder, tmp_path, monkeypatch, kind):
    root = tmp_path / "retained"
    if kind == "directory":
        root.mkdir()
        retained = root / "texlive-2024-final.tlpdb.xz"
    else:
        retained = root
    retained.write_bytes(b"previous evidence")
    monkeypatch.setattr(sys, "argv", ["build", "--root", str(root), "--database", str(tmp_path / "missing")])
    with pytest.raises(SystemExit, match="existing setup root"):
        builder.main()
    assert retained.read_bytes() == b"previous evidence"
    if kind == "directory":
        assert list(root.iterdir()) == [retained]


@pytest.mark.parametrize("database", [b"invalid xz", lzma.compress(b"synthetic unpinned database")])
def test_database_pin_is_checked_before_any_setup_write(builder, tmp_path, monkeypatch, database):
    source = tmp_path / "database.xz"
    source.write_bytes(database)
    root = tmp_path / "new-parent/install"
    monkeypatch.setattr(sys, "argv", ["build", "--root", str(root), "--database", str(source)])
    with pytest.raises(SystemExit, match="pinned SHA-256"):
        builder.main()
    assert not root.parent.exists()
    assert source.read_bytes() == database


@pytest.fixture
def synthetic_font_checkout(configure, tmp_path, monkeypatch):
    checkout = tmp_path / "font-source"
    blobs = {"fonts/A.ttf": b"synthetic font blob", "LICENSE.md": b"synthetic license blob"}
    for name, content in blobs.items():
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def git_oracle(command, *, cwd, text=False):
        assert cwd == checkout
        if command[:5] == ["git", "ls-tree", "-r", "--name-only", "HEAD"]:
            assert text
            return "\n".join(sorted(blobs)) + "\n"
        assert command[:2] == ["git", "show"]
        assert not text
        return blobs[command[2].removeprefix("HEAD:")]

    monkeypatch.setattr(configure.subprocess, "check_output", git_oracle)
    return checkout


def test_clean_copied_font_bytes_match_the_git_oracle(configure, synthetic_font_checkout):
    configure.verify_font_sources(synthetic_font_checkout, ("fonts", "LICENSE.md"))


@pytest.mark.parametrize("relative", ["fonts/A.ttf", "LICENSE.md"])
def test_changed_copied_font_or_license_is_rejected(configure, synthetic_font_checkout, relative):
    (synthetic_font_checkout / relative).write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="differs from Git"):
        configure.verify_font_sources(synthetic_font_checkout, ("fonts", "LICENSE.md"))


@pytest.mark.parametrize("extra", ["fonts/extra.ttf", "fonts/.ignored.ttf"])
def test_untracked_or_ignored_copied_font_is_rejected(configure, synthetic_font_checkout, extra):
    (synthetic_font_checkout / extra).write_bytes(b"untracked synthetic font")
    with pytest.raises(ValueError, match="inventory differs"):
        configure.verify_font_sources(synthetic_font_checkout, ("fonts", "LICENSE.md"))


def test_missing_copied_font_is_rejected(configure, synthetic_font_checkout):
    (synthetic_font_checkout / "fonts/A.ttf").unlink()
    with pytest.raises(ValueError, match="inventory differs"):
        configure.verify_font_sources(synthetic_font_checkout, ("fonts", "LICENSE.md"))


def test_missing_owned_tool_does_not_fall_back_to_path(configure, tmp_path, monkeypatch):
    root = tmp_path / "texlive"
    root.mkdir()
    system = tmp_path / "system-bin"
    system.mkdir()
    (system / "luahbtex").write_bytes(b"synthetic system executable")
    monkeypatch.setenv("PATH", str(system))
    with pytest.raises(ValueError, match="Missing owned TeX tool"):
        configure.owned_tex_tool(root, "luahbtex")


def test_tool_resolving_outside_owned_tree_is_rejected(configure, tmp_path, monkeypatch):
    root = tmp_path / "texlive"
    candidate = root / "bin/x86_64-linux/luahbtex"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"synthetic link placeholder")
    outside = tmp_path / "external-engine"
    outside.write_bytes(b"synthetic external engine")
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        # Model symlink resolution without requiring Windows symlink privileges.
        return outside if path == candidate else original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="outside the owned tree"):
        configure.owned_tex_tool(root, "luahbtex")


def test_owned_tool_must_be_executable(configure, tmp_path, monkeypatch):
    root = tmp_path / "texlive"
    tool = root / "bin/x86_64-linux/luahbtex"
    tool.parent.mkdir(parents=True)
    tool.write_bytes(b"synthetic tool")
    monkeypatch.setattr(os, "access", lambda *_: False)
    with pytest.raises(ValueError, match="not executable"):
        configure.owned_tex_tool(root, "luahbtex")


def test_configure_refuses_existing_output_before_source_reads(configure, tmp_path, monkeypatch):
    output = tmp_path / "retained"
    output.mkdir()
    previous = output / "render-env.json"
    previous.write_bytes(b"previous evidence")
    args = ["configure", "--output", str(output)]
    for name in ("texlive", "inara", "hack", "pandoc"):
        args.extend(["--" + name, str(tmp_path / name)])
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit, match="reuse configuration directory"):
        configure.main()
    assert previous.read_bytes() == b"previous evidence"
    assert list(output.iterdir()) == [previous]


def test_missing_owned_tools_stop_configuration_before_writes(configure, tmp_path, monkeypatch):
    output = tmp_path / "not-created"
    texlive = tmp_path / "texlive"
    texlive.mkdir()
    pins = {
        "inara": "88ee5d656f147d801448ce77ad8375619d383df6",
        "hack": "b47d21f808fbf4784565feccde2213ef2af924b3",
    }

    def pinned_sources(command, *, cwd, text):
        assert command == ["git", "rev-parse", "HEAD"] and text
        return pins[cwd.name]

    monkeypatch.setattr(configure.subprocess, "check_output", pinned_sources)
    monkeypatch.setattr(configure, "verify_font_sources", lambda *_: None)
    args = ["configure", "--output", str(output)]
    for name in ("texlive", "inara", "hack", "pandoc"):
        args.extend(["--" + name, str(tmp_path / name)])
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit, match="Missing owned TeX tool"):
        configure.main()
    assert not output.exists()
