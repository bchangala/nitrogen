"""
nitrogen.dvr
------------

This module provides support for discrete-variable 
representation (DVR) basis functions. The main object
is the :class:`DVR` class.

"""

# Import main module into name-space
from . import dvr
from .dvr import *

# Load submodules
from . import ops  # DVR operators

__all__ = dvr.__all__