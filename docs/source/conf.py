"""Configuration file for the Sphinx documentation builder."""
import sys
from pathlib import Path

# sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "innovate"
copyright_notice = "2025, Doughnut"
author = "Doughnut"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
