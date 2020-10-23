"""
NITROGEN
========

A Python package for quantum nuclear motion
calculations and rovibronic spectroscopy.

"""

# Import sub-packages and modules into namespace
from . import autodiff
from . import linalg 
from . import dvr 
from . import dfun 
from . import coordsys 
from . import ham
from . import constants




#################################################################
# Define some top-level constants
pi = 3.14159265358979323846264338327950288419716939937510   # pi
deg = pi / 180.0 # 1 degree in radians