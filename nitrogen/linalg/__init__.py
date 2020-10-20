"""
nitrogen.linalg
---------------

Linear algebra routines, including iterative eigensolvers.
This module uses the SciPy :class:`~scipy.sparse.linalg.LinearOperator` class
to represent matrix-vector product routines.

"""

# Import sub-module into name-space
from . import linalg
from .linalg import *  # Import the core namespace

from . import packed   # packed storage submodule

__all__ = linalg.__all__