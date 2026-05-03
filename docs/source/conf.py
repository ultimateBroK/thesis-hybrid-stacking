"""Sphinx configuration for the thesis project.

Autodoc + Napoleon + Viewcode + MyST for a clean hybrid documentation build.
"""
import os
import sys

# -- Project information -----------------------------------------------------
project = "thesis"
copyright = "2026, Hieu Nguyen"
author = "Hieu Nguyen"
version = "0.1.0"
release = "0.1.0"

# -- Path setup --------------------------------------------------------------
# Allow autodoc to discover thesis.* modules under src/
sys.path.insert(0, os.path.abspath("../../src"))
# Allow local extensions in _ext/
sys.path.insert(0, os.path.abspath("_ext"))

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinxcontrib.mermaid",
    "suppress_same_doc_duplicates",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_ivar = False

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_fence_as_directive = ["mermaid"]

mermaid_cmd = "npx -y @mermaid-js/mermaid-cli"
mermaid_params = [
    "-p",
    os.path.join(os.path.dirname(__file__), "puppeteer-config.json"),
]

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress warnings that are harmless in this project:
#   - autodoc: duplicate object descriptions from class attributes with docstrings
#   - misc.highlighting_failure: mermaid blocks in included markdown guides
#   - myst.xref_missing: cross-references to files outside the Sphinx source tree
suppress_warnings = [
    "misc.highlighting_failure",
    "myst.xref_missing",
    "python",
]

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
}
