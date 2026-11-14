"""The documentation artifact gate must inspect this build, not old receipts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_production_docs.py"
SPEC = importlib.util.spec_from_file_location("production_docs", SCRIPT)
docs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(docs)


class ProductionDocumentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dist = self.root / "dist"
        self.dist.mkdir()
        (self.root / "docs").mkdir()
        (self.root / "docs/page.md").write_text("Current source\n")
        self.contract = {
            "schema_version": 1,
            "origin": "https://edithatogo.github.io",
            "base": "/innovate/",
            "required_routes": ["/", "/guide/"],
            "source_roots": ["docs"],
            "exclude_source_parts": [],
            "max_html_bytes": 2_000_000,
        }
        self.write_page("index.html", "/innovate/", '<a href="/innovate/guide/#result">Guide</a>')
        self.write_page("guide/index.html", "/innovate/guide/", '<h2 id="result">Result</h2>')
        (self.dist / "pagefind").mkdir()
        (self.dist / "pagefind/pagefind.js").write_text("export async function search() {}")
        (self.dist / "pagefind/fragment.pf_fragment").write_bytes(b"indexed content")
        (self.dist / "sitemap-0.xml").write_text(
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            "<url><loc>https://edithatogo.github.io/innovate/</loc></url>"
            "<url><loc>https://edithatogo.github.io/innovate/guide/</loc></url></urlset>"
        )

    def write_page(self, relative, canonical, content):
        path = self.dist / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '<!doctype html><html lang="en"><head><title>Innovate</title>'
            '<meta name="generator" content="Astro v7">'
            '<meta name="description" content="Native diffusion documentation">'
            f'<link rel="canonical" href="https://edithatogo.github.io{canonical}">'
            f"</head><body><main><h1>Innovate</h1>{content}</main></body></html>"
        )

    def check(self):
        return docs.verify_artifact(self.root, self.dist, self.contract)

    def test_real_artifact_passes_and_reports_hashes(self):
        result = self.check()
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["counts"]["html_pages"], 2)
        self.assertTrue(result["source_sha256"])
        self.assertTrue(result["artifact_sha256"])
        self.assertFalse(result["live_deployment_verified"])
        self.assertFalse(result["content_complete"])

    def test_historical_pass_cannot_replace_missing_output(self):
        (self.root / "docs/old-report.json").write_text(json.dumps({"overall_status": "passed"}))
        (self.dist / "guide/index.html").unlink()
        self.assertEqual(self.check()["status"], "failed")

    def test_broken_fragment_fails(self):
        self.write_page("index.html", "/innovate/", '<a href="/innovate/guide/#missing">Guide</a>')
        self.assertIn("broken_fragment", str(self.check()["errors"]))

    def test_double_base_canonical_fails(self):
        self.write_page("guide/index.html", "/innovate/innovate/guide/", "")
        self.assertIn("canonical", str(self.check()["errors"]))

    def test_missing_base_in_internal_link_fails(self):
        self.write_page("index.html", "/innovate/", '<a href="/guide/">Guide</a>')
        self.assertIn("outside_base", str(self.check()["errors"]))

    def test_external_and_mail_links_are_not_network_requests(self):
        self.write_page(
            "index.html",
            "/innovate/",
            '<a href="https://example.org/unavailable">Reference</a><a href="mailto:editor@example.org">Editor</a>',
        )
        self.assertEqual(self.check()["status"], "passed")

    def test_script_only_search_is_not_a_populated_index(self):
        (self.dist / "pagefind/fragment.pf_fragment").unlink()
        self.assertIn("search_index", str(self.check()["errors"]))

    def test_sitemap_cannot_point_at_missing_route(self):
        sitemap = self.dist / "sitemap-0.xml"
        sitemap.write_text(sitemap.read_text().replace("/guide/", "/invented/"))
        self.assertIn("sitemap", str(self.check()["errors"]))

    def test_page_without_heading_or_image_alternative_fails(self):
        path = self.dist / "index.html"
        path.write_text(path.read_text().replace("<h1>Innovate</h1>", '<img src="https://example.org/a.png">'))
        self.assertIn("heading", str(self.check()["errors"]))
        self.assertIn("image_alt", str(self.check()["errors"]))

    def test_source_and_artifact_hashes_change_independently(self):
        first = self.check()
        (self.root / "docs/page.md").write_text("Changed source\n")
        second = self.check()
        self.assertNotEqual(first["source_sha256"], second["source_sha256"])
        self.assertEqual(first["artifact_sha256"], second["artifact_sha256"])

    def test_symlink_outside_artifact_is_rejected(self):
        secret = self.root / "outside.html"
        secret.write_text("outside")
        (self.dist / "escape.html").symlink_to(secret)
        self.assertIn("symlink", str(self.check()["errors"]))

    def test_unknown_schema_is_rejected(self):
        self.contract["schema_version"] = 999
        with self.assertRaises(ValueError):
            self.check()

    def test_invalid_contract_types_are_rejected(self):
        for name, value in (
            ("schema_version", True),
            ("origin", None),
            ("base", []),
            ("required_routes", []),
            ("source_roots", "docs"),
            ("exclude_source_parts", None),
            ("max_html_bytes", True),
            ("max_html_bytes", -1),
        ):
            with self.subTest(name=name, value=value):
                contract = dict(self.contract, **{name: value})
                with self.assertRaises(ValueError):
                    docs.verify_artifact(self.root, self.dist, contract)

    def test_source_traversal_and_absolute_paths_are_rejected(self):
        for path in ("../outside", "/tmp/outside", "docs/../docs", "docs\\outside", "."):
            with self.subTest(path=path):
                self.contract["source_roots"] = [path]
                with self.assertRaises(ValueError):
                    self.check()

    def test_source_root_symlink_is_rejected(self):
        (self.root / "linked-source").symlink_to(self.root / "docs", target_is_directory=True)
        self.contract["source_roots"] = ["linked-source"]
        with self.assertRaises(ValueError):
            self.check()

    def test_source_ancestor_symlink_is_rejected(self):
        (self.root / "linked-source").symlink_to(self.root / "docs", target_is_directory=True)
        self.contract["source_roots"] = ["linked-source/page.md"]
        with self.assertRaises(ValueError):
            self.check()

    def test_artifact_root_symlink_is_rejected(self):
        linked = self.root / "linked-artifact"
        linked.symlink_to(self.dist, target_is_directory=True)
        with self.assertRaises(ValueError):
            docs.verify_artifact(self.root, linked, self.contract)

    def test_invalid_origin_and_base_are_rejected(self):
        for origin in (
            "",
            "https://",
            "https://host/?query",
            "https://host#fragment",
            "https://user@host",
            "http://host",
        ):
            with self.subTest(origin=origin):
                with self.assertRaises(ValueError):
                    docs.verify_artifact(self.root, self.dist, dict(self.contract, origin=origin))
        for base in ("innovate/", "/innovate", "/innovate/../", "/innovate//", "/innovate/?query"):
            with self.subTest(base=base):
                with self.assertRaises(ValueError):
                    docs.verify_artifact(self.root, self.dist, dict(self.contract, base=base))

    def test_multiple_generators_include_astro(self):
        path = self.dist / "index.html"
        path.write_text(path.read_text().replace("</head>", '<meta name="generator" content="Starlight v0.42"></head>'))
        self.assertEqual(self.check()["status"], "passed")

    def test_github_404_file_uses_astro_canonical_route(self):
        self.write_page("404.html", "/innovate/404/", "Page not found")
        self.assertEqual(self.check()["status"], "passed")

    def test_invalid_contract_replaces_previous_pass_receipt(self):
        contract_path = self.root / "contract.json"
        contract_path.write_text("null")
        output = self.root / "receipt.json"
        output.write_text(json.dumps({"status": "passed"}))
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--root",
                str(self.root),
                "--dist",
                str(self.dist),
                "--contract",
                str(contract_path),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        receipt = json.loads(output.read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse(receipt["content_complete"])


if __name__ == "__main__":
    unittest.main()
