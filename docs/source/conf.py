# Configuration file for the Sphinx documentation builder.
import pathlib
import sys
import datetime
import importlib
import inspect
import os
import posixpath

from sphinx.application import Sphinx
from sphinx.writers.html5 import HTML5Translator

# -- Project information -----------------------------------------------------

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

project = 'OntoAligner'
copyright = f'{str(datetime.datetime.now().year)} SciKnowOrg'
author = 'Hamed Babaei Giglou'
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "myst_parser",
    "sphinx_markdown_tables",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx_inline_tabs",
    "sphinxcontrib.mermaid",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to include when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
include_patterns = [
    "../../ontoaligner/**/.py",
    "../../examples/**"
]
# Ensure exclude_patterns doesn't exclude your master document accidentally
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# specify the master doc, otherwise the build at read the docs fails
master_doc = "index"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "press"

html_theme_options = {
    "external_links": [
          ("Github", "https://github.com/sciknoworg/OntoAligner"),
          ("Pypi", "https://pypi.org/project/OntoAligner/")
    ],
    "logo_only": True,
    "light_mode": True,
    "collapse_navigation": False,
    "navigation_depth": 3
}

html_static_path = ["_static"]

html_js_files = [
    'https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js',
    'custom.js'
]

# Add custom CSS
html_css_files = [
    'custom.css',
]
html_show_sourcelink = False
html_context = {
    "display_github": True,
    "github_user": "sciknoworg",
    "github_repo": "OntoAligner",
    "github_version": "master/",
}

html_logo = 'img/logo-ontoaligner.png'
html_favicon = "img/logo-ontoaligner.ico"
autoclass_content = "both"

# Required to get rid of some myst.xref_missing warnings
myst_heading_anchors = 3


# https://github.com/readthedocs/sphinx-autoapi/issues/202#issuecomment-907582382
def linkcode_resolve(domain, info):
    # Non-linkable objects from the starter kit in the tutorial.
    if domain == "js" or info["module"] == "connect4":
        return

    assert domain == "py", "expected only Python objects"

    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])
    obj = inspect.unwrap(obj)

    try:
        file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except TypeError:
        # e.g. object is a typing.Union
        return None
    file = os.path.relpath(file, os.path.abspath(".."))
    if not file.startswith("ontoaligner"):
        # e.g. object is a typing.NewType
        return None
    start, end = lines[1], lines[1] + len(lines[0]) - 1

    return f"https://github.com/sciknoworg/OntoAligner/blob/main/{file}#L{start}-L{end}"


def visit_download_reference(self, node):
    root = "https://github.com/sciknoworg/OntoAligner/tree/main"
    atts = {"class": "reference download", "download": ""}

    if not self.builder.download_support:
        self.context.append("")
    elif "refuri" in node:
        atts["class"] += " external"
        atts["href"] = node["refuri"]
        self.body.append(self.starttag(node, "a", "", **atts))
        self.context.append("</a>")
    elif "reftarget" in node and "refdoc" in node:
        atts["class"] += " external"
        atts["href"] = posixpath.join(root, os.path.dirname(node["refdoc"]), node["reftarget"])
        self.body.append(self.starttag(node, "a", "", **atts))
        self.context.append("</a>")
    else:
        self.context.append("")


HTML5Translator.visit_download_reference = visit_download_reference


def setup(app: Sphinx):
    pass
