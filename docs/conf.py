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

import pandas_select

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------


project = "pandas-select"
copyright = "2020, Jean-Francois Zinque"
author = "Jean-Francois Zinque"

# The full version, including alpha/beta/rc tags
release = pandas_select.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "IPython.sphinxext.ipython_directive",
    "sphinx_copybutton",
    "sphinx_panels",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
pygments_style = "friendly"
html_theme = "furo"
html_baseurl = "https://pandas-select.readthedocs.io/en/latest/"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]


# Options for the linkcode extension
# ----------------------------------

# Resolve function for the linkcode extension.
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.abspath(".."))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = f"{find_source()}#L%d-L%d"
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    return "https://github.com/jeffzi/pandas-select/blob/master/" + filename


# -- Options for extensions --------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
    "pandera": ("https://pandera.readthedocs.io/en/stable", None),
}

autosummary_generate = True
autodoc_typehints = "none"
add_module_names = False

copybutton_prompt_text = r">>> |\.\.\.\: |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
