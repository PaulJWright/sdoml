# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import os
import sys

# -- Project information -----------------------------------------------------

project = "sdoml"
copyright = "2022, Paul J. Wright"
author = "Paul J. Wright"

# The full version, including alpha/beta/rc tags
from sdoml import __version__  # noqa E402

release = __version__

# -- General configuration ---------------------------------------------------

sys.path.insert(
    0, os.path.abspath("../../sdoml")
)  # Source code dir relative to this file
sys.path.insert(
    0, os.path.abspath("../../")
)  # Source code dir relative to this file

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    # 'sphinx.ext.autosummary',
    "sphinx_rtd_theme",
    # 'sphinx_gallery.gen_gallery',
    "sphinx_copybutton",
    "nbsphinx",
    "nbsphinx_link",
]

#
# https://nbsphinx.readthedocs.io/en/0.6.1/usage.html
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# Napoleon Settings
napoleon_google_docstring = False
napoleon_use_param = False

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "aiapy": ("https://aiapy.readthedocs.io/en/stable/", None),
    "sunpy": ("https://docs.sunpy.org/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
