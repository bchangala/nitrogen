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
from . import pes

__version__ = '2.0.dev2'

import numpy as np 

#################################################################
# Define some top-level constants
pi = 3.14159265358979323846264338327950288419716939937510   # pi
deg = pi / 180.0 # 1 degree in radians


def X2xyz(X, elements, filename = "out.xyz", comment = None):
    """
    Create an .xyz text file from an array of
    Cartesian positions.

    Parameters
    ----------
    X : (3*N,...) ndarray
        Cartesian position of N atoms.
    elements : list of str
        List of element identifiers
    filename : str, optional
        Output filename. The default is "out.xyz".
    comment : str, optional
        A comment string for the .xyz file. If None,
        a default will be used.

    Returns
    -------
    None.

    """
    
    if np.ndim(X) < 1 :
        raise ValueError("X must have at least 1 dimension.")
    if X.shape[0] % 3 != 0:
        raise ValueError("The first dimension of X must be a multiple of 3.")
    N = X.shape[0] // 3 # number of atoms 
    if N < 1:
        raise ValueError("The number of atoms must be >= 1")
    
    Xp = np.reshape(X, (N,3,-1)) 
    
    ng = Xp.shape[2] # The number of geometries 
    
    # Write .xyz file 
    with open(filename, 'w') as file :
        
        for i in range(ng): # For each geometry
            file.write(f"{N:d}\n") # number of atoms
            
            if comment is None: 
                file.write(f"Geometry {i:d}\n") # comment line
            else :
                file.write(f"{comment:s}\n") # supplied comment 
            
            for j in range(N): # For each atom
                e = elements[j] # element label
                x = Xp[j,0,i] # x
                y = Xp[j,1,i] # y
                z = Xp[j,2,i] # z
                
                file.write(f"{e:s}   {x:.15f} {y:.15f} {z:.15f} \n")
    # Done
    return 
    