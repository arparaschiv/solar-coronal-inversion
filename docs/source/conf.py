# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../docs/source'))
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------
project = 'CLEDB solar-coronal-inversion'
copyright = '2025, Alin Paraschiv'
author = 'Alin Paraschiv'

# The full version, including alpha/beta/rc tags
release = 'update-pycelpbuild'


# -- General configuration ---------------------------------------------------
#import sphinx_rtd_theme ##PAR import as in https://sphinx-rtd-theme.readthedocs.io/en/stable/installing.html
#import myst_parser
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["myst_parser", "sphinx_rtd_theme", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
#extensions =[]

# Optional: fine-tune napoleon
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Options for autodoc ---------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_mock_imports = ["multiprocessing","tqdm","astropy","time","numba"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath", "fieldlist", "html_admonition", "html_image", "replacements", "smartquotes", "strikethrough", "substitution", "tasklist"]

myst_heading_anchors = 5
