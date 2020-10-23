"""
nitrogen.coordsys
-----------------

This module implements the CoordSys base class,
which is extended for all NITROGEN coordinate
systems.

"""

from . import coordsys 
from . import simple_builtins

from .coordsys import *
from .simple_builtins import *

__all__ = []
__all__ += coordsys.__all__
__all__ += simple_builtins.__all__