"""Keep retired instructions out of current documentation and preserve their identity.

Synthetic manifests and HTML isolate the retirement gate from native execution.
The Git-history test uses a local temporary repository and performs no network I/O.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_retired_documentation.py"
SPEC = importlib.util.spec_from_file_location("retired_documentation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class RetirementContractTests(unittest.TestCase):
    """Reject resurrected, redirected, untraceable and executable old material."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dist = self.root / "dist"
        self.source = "docs/astro-site/src/content/docs/latest/api/python.md"
        self.route = "/innovate/latest/api/python/"
        self.commit = "a" * 40
        self.entry = {
            "path": self.source,
            "category": "versioned_page",
            "sha256": "b" * 64,
            "git_blob": "c" * 40,
            "history_url": f"https://github.com/edithatogo/innovate/blob/{self.commit}/{self.source}",
            "replacement": "/innovate/api/python/",
            "scope": "Earlier interfaces are retained only in immutable Git history.",
            "route": self.route,
            "title": "Earlier Python API",
            "relationship": "historical_version",
        }
        self.contract = {
            "schema_version": 1,
            "retired_at": "2026-11-16",
            "source_commit": self.commit,
            "policy": "Retire obsolete executable instructions without claiming scientific acceptance.",
            "entries": [self.entry],
        }
        self.write("docs/contracts/content-quality.json", json.dumps({
            "pages": {"index.md": {"kind": "index"}, "api/python.md": {"kind": "reference"}},
            "generated_api": {"manifest": "docs/astro-site/src/content/docs/api/generated/manifest.json"},
        }))
        self.write("docs/astro-site/src/content/docs/api/generated/manifest.json", json.dumps({
            "pages": {"julia.md": {"sha256": "d" * 64}},
        }))
        self.write("docs/astro-site/src/content/docs/index.md", "---\ntitle: Current documentation\n---\n")
        self.write("docs/astro-site/src/content/docs/api/python.md", "---\ntitle: Current Python API\n---\n")
        self.write_notice(self.notice())

    def write(self, relative: str, contents: str) -> Path:
        """Write only inside this test's isolated synthetic repository."""
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents, encoding="utf-8")
        return destination

    def notice(self) -> str:
        """Return an independently authored minimal valid retirement notice."""
        return (
            '<!doctype html><html><head><meta name="robots" content="noindex, follow"></head><body>'
            f'<main><section data-retired-source="{self.source}">'
            '<h1>Retired: Earlier Python API</h1><p>This documentation has been retired.</p>'
            '<a href="/innovate/api/python/">Current guidance</a>'
            f'<a href="{self.entry["history_url"]}">Historical source</a>'
            '</section></main></body></html>'
        )

    def write_notice(self, text: str) -> Path:
        """Write the notice at the independently specified former public URL."""
        return self.write("dist/latest/api/python/index.html", text)

    def test_valid_source_and_rendered_notices_are_not_scientific_acceptance(self) -> None:
        source = checker.audit(self.root, self.contract)
        rendered = checker.audit(self.root, self.contract, self.dist)
        self.assertEqual(source["status"], "passed")
        self.assertEqual(source["rendered_notices_checked"], 0)
        self.assertEqual(rendered["removed_sources"], 1)
        self.assertEqual(rendered["migration_routes"], 1)
        self.assertEqual(rendered["rendered_notices_checked"], 1)
        self.assertIs(rendered["git_history_checked"], False)
        self.assertIs(rendered["scientific_acceptance"], False)

    def test_manifest_identity_and_unknown_fields_fail(self) -> None:
        for field, value in (
            ("schema_version", True), ("schema_version", 2), ("source_commit", "main"),
            ("retired_at", "tomorrow"), ("retired_at", "2026-99-99"),
            ("policy", ""), ("policy", {"claimed": "a policy"}), ("entries", []), ("accepted", True),
        ):
            with self.subTest(field=field, value=value):
                broken = copy.deepcopy(self.contract)
                broken[field] = value
                with self.assertRaises(ValueError):
                    checker.audit(self.root, broken)

    def test_existing_source_cannot_be_called_retired(self) -> None:
        self.write(self.source, "Earlier installation and execution instructions.")
        with self.assertRaisesRegex(ValueError, "returned"):
            checker.audit(self.root, self.contract)

    def test_duplicate_sources_and_routes_fail(self) -> None:
        self.contract["entries"].append(copy.deepcopy(self.entry))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            checker.audit(self.root, self.contract)
        self.contract["entries"][1]["path"] = self.source.replace("python.md", "different.md")
        self.contract["entries"][1]["history_url"] = self.entry["history_url"].replace("python.md", "different.md")
        with self.assertRaises(ValueError):
            checker.audit(self.root, self.contract)

    def test_every_retired_web_source_requires_its_canonical_url(self) -> None:
        for category in ("historical_page", "versioned_page"):
            with self.subTest(category=category, defect="missing route"):
                broken = copy.deepcopy(self.contract)
                entry = broken["entries"][0]
                entry["category"] = category
                for field in ("route", "title", "relationship"):
                    del entry[field]
                with self.assertRaises(ValueError):
                    checker.audit(self.root, broken)
            with self.subTest(category=category, defect="unrelated route"):
                broken = copy.deepcopy(self.contract)
                broken["entries"][0]["category"] = category
                broken["entries"][0]["route"] = "/innovate/unrelated-former-page/"
                with self.assertRaises(ValueError):
                    checker.audit(self.root, broken)

    def test_source_traversal_and_leaf_symlinks_fail(self) -> None:
        for source in ("../outside.md", "/tmp/outside.md", "docs/../outside.md", "docs//outside.md"):
            with self.subTest(source=source):
                broken = copy.deepcopy(self.contract)
                broken["entries"][0]["path"] = source
                with self.assertRaises(ValueError):
                    checker.audit(self.root, broken)
        destination = self.root / self.source
        destination.parent.mkdir(parents=True)
        destination.symlink_to(self.root / "missing-source.md")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            checker.audit(self.root, self.contract)

    def test_mutable_or_mismatched_history_and_invalid_hashes_fail(self) -> None:
        for field, value in (
            ("history_url", self.entry["history_url"].replace(self.commit, "main")),
            ("history_url", self.entry["history_url"].replace("python.md", "different.md")),
            ("history_url", self.entry["history_url"].replace("github.com", "example.org")),
            ("sha256", "not-a-hash"), ("git_blob", "0" * 39),
        ):
            with self.subTest(field=field, value=value):
                broken = copy.deepcopy(self.contract)
                broken["entries"][0][field] = value
                with self.assertRaises(ValueError):
                    checker.audit(self.root, broken)

    def test_replacement_must_be_current_and_route_cannot_collide(self) -> None:
        for field, value in (
            ("replacement", self.route), ("replacement", "/innovate/not-current/"),
            ("replacement", "https://example.org/"), ("replacement", "/innovate/%2e%2e/"),
            ("route", "/innovate/api/python/"),
        ):
            with self.subTest(field=field, value=value):
                broken = copy.deepcopy(self.contract)
                broken["entries"][0][field] = value
                with self.assertRaises(ValueError):
                    checker.audit(self.root, broken)

    def test_rendered_identity_links_search_markers_and_robots_are_required(self) -> None:
        original = self.notice()
        mutations = {
            "source identity": original.replace(self.source + '">', 'wrong-source">'),
            "notice marker": original.replace("This documentation has been retired", "Earlier API"),
            "current link": original.replace('href="/innovate/api/python/"', 'href="/innovate/not-current/"'),
            "history link": original.replace(self.entry["history_url"], "https://example.org/"),
            "search inclusion": original.replace("<main>", "<main data-pagefind-body>"),
            "robots": original.replace("noindex, follow", "index, follow"),
            "duplicate robots": original.replace("</head>", '<meta name="robots" content="noindex, follow"></head>'),
        }
        for defect, html in mutations.items():
            with self.subTest(defect=defect):
                self.write_notice(html)
                with self.assertRaisesRegex(ValueError, "rendered retirement notice"):
                    checker.audit(self.root, self.contract, self.dist)

    def test_retired_pages_cannot_reintroduce_code_or_version_navigation(self) -> None:
        for fragment in (
            "<pre>pip install old-api</pre>",
            "<code>pip install old-api</code>",
            '<starlight-version-select><select aria-label="Version"><option>0.6</option>'
            '</select></starlight-version-select>',
        ):
            with self.subTest(fragment=fragment):
                self.write_notice(self.notice().replace("</section>", fragment + "</section>"))
                with self.assertRaisesRegex(ValueError, "rendered retirement notice"):
                    checker.audit(self.root, self.contract, self.dist)

    def test_missing_or_symlinked_artifacts_cannot_qualify(self) -> None:
        artifact = self.dist / "latest/api/python/index.html"
        artifact.unlink()
        with self.assertRaises(FileNotFoundError):
            checker.audit(self.root, self.contract, self.dist)
        artifact.symlink_to(self.root / "somewhere-else.html")
        with self.assertRaisesRegex(ValueError, "unsafe notice artifact"):
            checker.audit(self.root, self.contract, self.dist)

    def test_unregistered_obsolete_sources_fail_but_static_evidence_remains(self) -> None:
        self.write("docs/source/_static/evidence/receipt.txt", "Retained historical scientific receipt.")
        self.assertEqual(checker.audit(self.root, self.contract)["status"], "passed")
        for relative in (
            "docs/source/new-page.rst", "old_notebooks/old-example.ipynb",
            "docs/astro-site/src/content/docs/latest/unregistered.md",
        ):
            with self.subTest(path=relative):
                restored = self.write(relative, "Obsolete instructions.")
                with self.assertRaises(ValueError):
                    checker.audit(self.root, self.contract)
                restored.unlink()

    def test_real_git_history_binds_both_source_bytes_and_blob_identity(self) -> None:
        raw = b"# The exact retired source\n\nEarlier native interface.\n"
        source = self.write(self.source, raw.decode())
        subprocess.run(["git", "init", "--quiet"], cwd=self.root, check=True, capture_output=True)
        subprocess.run(["git", "add", self.source], cwd=self.root, check=True, capture_output=True)
        subprocess.run(
            ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
             "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Historical fixture"],
            cwd=self.root, check=True, capture_output=True,
        )
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()
        blob = subprocess.check_output(["git", "rev-parse", f"HEAD:{self.source}"], cwd=self.root, text=True).strip()
        source.unlink()
        self.contract["source_commit"] = commit
        self.entry.update({
            "sha256": hashlib.sha256(raw).hexdigest(), "git_blob": blob,
            "history_url": f"https://github.com/edithatogo/innovate/blob/{commit}/{self.source}",
        })
        report = checker.audit(self.root, self.contract, history=True)
        self.assertIs(report["git_history_checked"], True)
        for field, value in (("sha256", "0" * 64), ("git_blob", "0" * 40)):
            with self.subTest(field=field):
                broken = copy.deepcopy(self.contract)
                broken["entries"][0][field] = value
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    checker.audit(self.root, broken, history=True)

    def test_cli_returns_failure_for_bad_notice_and_does_not_repair_it(self) -> None:
        self.write("docs/contracts/retired-documentation.json", json.dumps(self.contract))
        artifact = self.write_notice(self.notice().replace("noindex, follow", "index, follow"))
        original = artifact.read_bytes()
        result = subprocess.run(
            [sys.executable, "-I", str(SCRIPT), "--root", str(self.root), "--dist", str(self.dist), "--json"],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "failed")
        self.assertIs(report["scientific_acceptance"], False)
        self.assertEqual(artifact.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
