Installation
============

.. toctree::
   :maxdepth: 2
   
The NITROGEN source code is available at our 
`github repository <https://github.com/bchangala/nitrogen>`_. Stable
releases are packaged and available on the 
`PyPI project page <https://pypi.org/project/nitrogen/>`_,
and can be installed on a command line via ``pip install nitrogen``.

For users new to Python, we recommend setting up a dedicated NITROGEN Python 
environment using Anaconda, Miniconda, etc. NITROGEN is not compatible with 
all Python versions, and specific version support may change with future releases of 
NITROGEN and Python.

For NITROGEN v2.2.3, we recommend using Python v3.7.11.

..  code-block:: console 
    
    > conda create -n nitrogen-env -c conda-forge python=3.7.11 spyder=5.1.5
    > conda activate nitrogen-env
    > pip install nitrogen==2.2.3 
    

(We have found that some dependencies have issues installing unless 
Numpy is installed manually first. Give that a shot. In some cases, Cython
may need to be downgraded to 0.29.21.)