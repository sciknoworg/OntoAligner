# Configuration file for the Sphinx documentation builder.
import pathlib
import sys
import datetime
import importlib
from sphinx.application import Sphinx
import inspect
import os
# -- Project information -----------------------------------------------------
#
sys.path.insert(0, pathlib.Path(__file__).parents[0].resolve().as_posix())
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
sys.path.insert(0, pathlib.Path(__file__).parents[1].resolve().as_posix())

def setup(app: Sphinx):
    app.add_css_file('_static/custom.css')
    app.add_js_file('_static/custom.js')


project = 'OntoAligner'
copyright = f'{str(datetime.datetime.now().year)} SciKnowOrg'
author = 'Hamed Babaei Giglou'
release = '0.2.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_markdown_tables",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx_inline_tabs",
    "sphinxcontrib.mermaid",
    "sphinx.ext.mathjax"
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary


autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
# set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
# nbsphinx_allow_errors = True  # Continue through Jupyter errors
# autodoc_typehints = "description" # Sphinx-native method. Not as good as sphinx_autodoc_typehints
# add_module_names = False  # Remove namespaces from class/method signatures

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to include when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
include_patterns = [
    "**",
    "../../ontoaligner/",
    "index.rst",
]
# Ensure exclude_patterns doesn't exclude your master document accidentally
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

source_suffix = '.rst'

# specify the master doc, otherwise the build at read the docs fails
master_doc = "index"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "press"

html_theme_options = {
    "external_links": [
          ("Github", "https://github.com/sciknoworg/OntoAligner"),
          ("Pypi", "https://pypi.org/project/OntoAligner/")
    ],
}

html_static_path = ["_static"]

html_js_files = [
    'https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js',
    'custom.js'
]

html_css_files = [
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css',
    'custom.css',
]

html_show_sourcelink = True
html_context = {
    "display_github": True,
    "github_user": "sciknoworg",
    "github_repo": "OntoAligner",
    "github_version": "main/",
}

html_logo = 'img/logo-ontoaligner.png'
html_favicon = "img/logo-ontoaligner.ico"
autoclass_content = "both"

# Required to get rid of some myst.xref_missing warnings
myst_heading_anchors = 3

html_copy_source = True
def linkcode_resolve(domain, info):
    """
    Resolve a GitHub link for the given domain and info dictionary.
    """
    if domain != "py" or not info["module"]:
        return None

    # Define the GitHub repository URL
    repo_url = "https://github.com/sciknoworg/OntoAligner/blob/main"
    branch = "main"  # Update if using a different branch

    # Retrieve the module and object
    try:
        module = importlib.import_module(info["module"])
    except ImportError:
        return None

    # Try to get the source file and line numbers
    try:
        file_path = inspect.getsourcefile(module)
        source_lines, start_line = inspect.getsourcelines(getattr(module, info["fullname"]))
    except (TypeError, AttributeError, OSError):
        return None

    # Generate the relative file path and GitHub link
    relative_path = os.path.relpath(file_path, start=os.path.dirname(__file__))
    end_line = start_line + len(source_lines) - 1
    return f"{repo_url}/blob/{branch}/{relative_path}#L{start_line}-L{end_line}"

#
# def linkcode_resolve(domain, info):
#     # print("linkcode_resolve:", domain, info)
#     if domain != "py" or not info["module"]:
#         return None
#     try:
#         # Import the module dynamically
#         mod = importlib.import_module(info["module"])
#
#         obj = mod
#         for attr in info["fullname"].split('.'):
#             obj = getattr(obj, attr)
#         obj = inspect.unwrap(obj)
#
#         # Get source file and line numbers
#         file = inspect.getsourcefile(obj)
#
#         lines = inspect.getsourcelines(obj)
#         start_line = lines[1]
#         # print(lines)
#         end_line = start_line + len(lines[0]) - 1
#
#         # Construct GitHub link
#         file_path = os.path.relpath(file, start=os.path.abspath(".."))
#         return f"https://github.com/sciknoworg/OntoAligner/blob/main/{file_path}#L{start_line}-L{end_line}"
#     except Exception as e:
#         print(f"Error in linkcode_resolve: {e}")  # Debugging line
#         return None


# https://github.com/readthedocs/sphinx-autoapi/issues/202#issuecomment-907582382
# def linkcode_resolve(domain, info):
#     # Non-linkable objects from the starter kit in the tutorial.
#     # if domain == "js" or info["module"] == "connect4":
#     if domain != "py" or not info["module"]:
#         return
#
#     # assert domain == "py", "expected only Python objects"
#
#     mod = importlib.import_module(info["module"])
#     if "." in info["fullname"]:
#         objname, attrname = info["fullname"].split(".")
#         obj = getattr(mod, objname)
#         try:
#             # object is a method of a class
#             obj = getattr(obj, attrname)
#         except AttributeError:
#             # object is an attribute of a class
#             return None
#     else:
#         obj = getattr(mod, info["fullname"])
#     obj = inspect.unwrap(obj)
#
#     try:
#         file = inspect.getsourcefile(obj)
#         lines = inspect.getsourcelines(obj)
#     except TypeError:
#         # e.g. object is a typing.Union
#         return None
#     file = os.path.relpath(file, os.path.abspath(".."))
#     if not file.startswith("ontoaligner"):
#         # e.g. object is a typing.NewType
#         return None
#     start, end = lines[1], lines[1] + len(lines[0]) - 1
#
#     return f"https://github.com/sciknoworg/OntoAligner/blob/main/{file}#L{start}-L{end}"


# def visit_download_reference(self, node):
#     root = "https://github.com/sciknoworg/OntoAligner/tree/main"
#     atts = {"class": "reference download", "download": ""}
#     print("WHAT:", node)
#     if not self.builder.download_support:
#         self.context.append("")
#     elif "refuri" in node:
#         atts["class"] += " external"
#         atts["href"] = node["refuri"]
#         self.body.append(self.starttag(node, "a", "", **atts))
#         self.context.append("</a>")
#     elif "reftarget" in node and "refdoc" in node:
#         atts["class"] += " external"
#         atts["href"] = posixpath.join(root, os.path.dirname(node["refdoc"]), node["reftarget"])
#         self.body.append(self.starttag(node, "a", "", **atts))
#         self.context.append("</a>")
#     else:
#         self.context.append("")
#
# def visit_download_reference(self, node):
#     root = "https://github.com/sciknoworg/OntoAligner/tree/main"
#     atts = {"class": "reference download", "download": ""}
#
#     if "refuri" in node:
#         atts["href"] = node["refuri"]
#     elif "reftarget" in node and "refdoc" in node:
#         atts["href"] = posixpath.join(root, node["refdoc"], node["reftarget"])
#     else:
#         return  # Skip if no valid reference
#
#     atts["class"] += " external"
#     self.body.append(self.starttag(node, "a", "", **atts))
#     self.context.append("</a>")
#
# HTML5Translator.visit_download_reference = visit_download_reference
