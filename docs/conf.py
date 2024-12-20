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
from unittest.mock import MagicMock


on_rtd = os.environ.get('READTHEDOCS') == 'True' # Check whether we are on Read the Docs
if not on_rtd:
    sys.path.insert(0, os.path.abspath('..'))
    # ^ This line needed to be commented out
    # so that read-the-docs can use
    # the locally built version
    # See https://stackoverflow.com/questions/13238736/how-to-document-cython-function-on-readthedocs
    #
    # Note also the Edit in the above URL describin changes in 2023/2024 
    # when read-the-docs changed its virtualenv build options. This required
    # adding the setuptools and `path` settings in the .readthedocs.yaml file.
    #
else:
    # On read-the-docs
    # Set up mock modules 
    # (See https://read-the-docs.readthedocs.io/en/latest/faq.html#i-get-import-errors-on-libraries-that-depend-on-c-modules)
    #
    class Mock(MagicMock):
        @classmethod 
        def __getattr__(cls,name):
            return MagicMock()
    MOCK_MODULES = ['py3nj'] # py3nj needs Fortran compilation 
    sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
    
from nitrogen import __version__


# -- Project information -----------------------------------------------------

project = 'NITROGEN'
copyright = '2021, Bryan Changala'
author = 'Bryan Changala'
version = __version__
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'sphinx.ext.doctest',
              'matplotlib.sphinxext.plot_directive'
              ]
autoclass_content = 'both'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    'numpy' : ('https://numpy.org/doc/stable/', None), 
    'scipy' : ('https://docs.scipy.org/doc/scipy', None),
    }

# Global doctest setup code
doctest_global_setup = """
import nitrogen as n2
import nitrogen.autodiff.forward as adf
import numpy as np
"""

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'nature'
html_style = 'custom-1.css' # my custom css, makes modifications to nature.css

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
