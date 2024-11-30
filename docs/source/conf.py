# Configuration file for the Sphinx documentation builder.
# -- Project information
import pathlib
import sys
import datetime

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

def setup(app):
    app.add_css_file('_static/custom.css')

# -- Project information -----------------------------------------------------
project = 'OntoAligner'
copyright = f'{str(datetime.datetime.now().year)} SciKnowOrg'
author = 'Hamed Babaei Giglou'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    "sphinx.ext.intersphinx",  # Link to other project's documentation (see mapping below)
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx_autodoc_typehints",  # Automatically document param types (less noise in class signature)
    # 'nbsphinx',  # Integrate Jupyter Notebooks and Sphinx
    # 'IPython.sphinxext.ipython_console_highlighting'
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
nbsphinx_allow_errors = True  # Continue through Jupyter errors
# autodoc_typehints = "description" # Sphinx-native method. Not as good as sphinx_autodoc_typehints
add_module_names = False  # Remove namespaces from class/method signatures

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to include when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# include_patterns = [
#     "**",
#     "../../ontoaligner/**/.py",
#     "../../examples/**",
#     "index.rst",
# ]
# Ensure exclude_patterns doesn't exclude your master document accidentally
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# specify the master doc, otherwise the build at read the docs fails
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "press"

html_theme_options = {
    "external_links": [
          ("Github", "https://github.com/sciknoworg/OntoAligner"),
          ("Pypi", "https://pypi.org/project/OntoAligner/")
    ],
    # 'use_repository_button': False,
    # 'use_version_button': False,
    # 'use_doc_button': True,
    'logo_only': True,
    'light_mode': True,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# https://stackoverflow.com/questions/23211695/modifying-content-width-of-the-sphinx-theme-read-the-docs
html_static_path = ["_static"]

html_js_files = [
    'https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js',
    'flyout.js'
]

# Add custom CSS
html_css_files = [
    'custom.css',
]

html_logo = 'img/logo-ontoaligner.png'
html_favicon = "img/logo-ontoaligner.ico"
