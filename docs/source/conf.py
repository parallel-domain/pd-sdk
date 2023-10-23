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
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("../../paralleldomain"))

print(os.getcwd())

# -- Project information -----------------------------------------------------

project = "Parallel Domain SDK"
copyright = "2023, Parallel Domain"
author = "Nisse Knudsen, Phillip Thomas"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions: List[str] = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",  # add links to other docs
    "sphinx_mdinclude",
    "sphinx_material",
]


autoclass_content = "both"
autodoc_inherit_docstrings = True
set_type_checking_flag = True
add_module_names = False
autodoc_member_order = "groupwise"
autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_use_param = True

highlight_language = "python3"

# Add any paths that contain templates here, relative to this directory.
templates_path: List[str] = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[str] = ["**/.ipynb_checkpoints"]

source_suffix = [".rst", ".md"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme: str = "sphinx_material"

html_show_sourcelink = True
html_sidebars = {"**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {"collapse_navigation": False}

# Material theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the navigation.
    "nav_title": "Parallel Domain SDK",
    # Set you GA account ID to enable tracking
    # "google_analytics_account": "UA-XXXXX",
    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    "base_url": "https://github.com/parallel-domain/pd-sdk",
    # Set the color and the accent color
    "color_primary": "blue",
    "color_accent": "light-blue",
    # Set the repo location to get a badge with stats
    "repo_url": "https://github.com/parallel-domain/pd-sdk",
    "repo_name": "PD SDK",
    # Visible levels of the global TOC; -1 means unlimited
    "globaltoc_depth": -1,
    # If False, expand all TOC entries
    "globaltoc_collapse": True,
    # If True, show hidden TOC entries
    "globaltoc_includehidden": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path: List[str] = ["_static"]
html_css_files = ["custom.css"]

# -- Options for Linking  ----------------------------------------------------

version_link = f"{sys.version_info.major}.{sys.version_info.minor}"
intersphinx_mapping = {
    "python": (f"https://docs.python.org/{version_link}", None),  # link to used Python version
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "plotly": ("https://plotly.com/python-api-reference", None),
}
