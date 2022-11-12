"""
nitrogen.coordsys
-----------------

This module implements the :class:`CoordSys` base class,
which is extended for all NITROGEN coordinate systems.
See :doc:`/tutorials/coordsys` for a tutorial.


=============================  =================================================
Coordinate systems
================================================================================
:class:`CoordSys`              The coordinate system base class.
:class:`ZMAT`                  Z-matrix coordinates.
:class:`Valence3`              Triatomic valence coordinates.
:class:`CartesianN`            Simple :math:`n`-dimensional Cartesian.
:class:`Polar`                 Polar coordinates.
:class:`Cylindrical`           Cylindrical coordinates.
:class:`Spherical`             Spherical coordinates.
:class:`QTransCoordSys`        Input-transformed coordinate system.
:class:`JacobiChain3N`         An :math:`n`-particle Jacobi chain.
=============================  =================================================

=============================  =================================================
Coordinate transformations
================================================================================
:class:`CoordTrans`            The coordinate transformation base class.
:class:`CompositeCoordTrans`   Composite coordinate transformations.
:class:`LinearTrans`           Linear transformations.
=============================  =================================================

"""

from . import coordsys 
from . import simple_builtins
from . import zmat
from . import jacobi 
from . import frames 

from .coordsys import *
from .simple_builtins import *
from .zmat import *
from .jacobi import *
from .frames import *

__all__ = []
__all__ += coordsys.__all__
__all__ += simple_builtins.__all__
__all__ += zmat.__all__
__all__ += jacobi.__all__
__all__ += frames.__all__