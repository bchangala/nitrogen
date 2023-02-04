"""
nitrogen.pes
------------

Potential energy surface utilities and library.

"""

import importlib 
import nitrogen.dfun as dfun 
import nitrogen.linalg as linalg
import nitrogen.constants as constants 
import numpy as np 
import nitrogen.coordsys as coordsys 

import warnings


from . import cfour
from .cfour import *
from . import opt 
from .opt import * 

from . import fit 

__all__ = ['loadpes', 'curvVib']
__all__ += cfour.__all__
__all__ += opt.__all__

def loadpes(pesname):
    """
    Load built-in PES.

    Parameters
    ----------
    pesname : str
        PES name.

    Returns
    -------
    pes : DFun
        The PES as a DFun object.

    """
    
    warnings.warn("loadpes is deprecated. Import directly from nitrogen.pes.library modules",
                  DeprecationWarning)
    
    try : 
        mod = importlib.import_module("nitrogen.pes.library." + pesname)
    except ModuleNotFoundError:
        raise ValueError("PES name is not recognized.")
        
        
    return mod.PES


def curvVib(Q0, pes, cs, masses, mode = 'bodyframe', fidx = 0):
    
    """
    Calculate curvilinear vibrational normal coordinates
    and frequencies at a stationary point.

    Parameters
    ----------
    Q0 : array_like
        The stationary point coordinates.
    pes : DFun
        Potential energy surface function.
    cs : CoordSys
        Coordinate system.
    masses : array_like
        Coordinate system masses.
    mode : {'bodyframe'}
        Coordinate frame mode. 
        'bodyframe' treats the `cs` coordinate system as 
        the body-fixed frame.
        'bodyframe' is the default.
    fidx : integer, optional
        The DFun function index to use. The default is 0

    Returns
    -------
    omega : ndarray
        The harmonic frequencies (times :math:`\hbar`).
        Negative frequencies are returned for imaginary frequencies.
    nctrans : LinearTrans
        The normal coordinate transformation.
        
    Notes
    -----
    The frequency calculation requires the inverse of the 
    molecular moment of inertia tensor. At linear geometries,
    this tensor is singular. Instead, its pseudo-inverse is
    calculated, which gives the correct results at linear
    geometries.
        
    """
    
    Q0 = np.array(Q0)
    
    nQ = pes.nx # Number of coordinates
    
    F = pes.hes(Q0)[fidx] # Calculate PES hessian
    
    g = cs.Q2g(Q0, masses = masses, mode = mode) # Calculate metric tensor 
    
    #
    # Naive inverse breaks for linear geometries
    #
    #G,_ = dfun.sym2invdet(g,0,1)                 # Invert metric tensor
    #G = linalg.packed.symfull(G[0])              # Convert to full value matrix
    #Gvib = G[:nQ,:nQ]                            # Extract vibrational block
    
    g = linalg.packed.symfull(g[0]) # Convert to full matrix 
    
    A = g[:nQ, :nQ] # vibrational block 
    C = g[nQ:, :nQ] # rot-vib block (lower left)
    D = g[nQ:, nQ:] # rotation block (inertia tensor) 
    
    # We need to calculate Gvib, the vibrational
    # block of the inverse of g. At linear geometries
    # g is singular, but Gvib is still well-defined
    # We will calculate it using block matrix inverse
    # and using a Schur complement with a pseudo-inverse
    iGvib = A - C.T @ np.linalg.pinv(D) @ C 
    Gvib = np.linalg.inv(iGvib) 

    GF = Gvib @ F  # Calculate GF matrix

    w,U = np.linalg.eig(GF) # Diagonalize GF matrix

    omega = constants.hbar * np.sqrt(np.abs(w)) # Calculate harmonic energies
    omega[w < 0] = -omega[w < 0] # Imaginary frequencies will be flagged as negative
    
    sort_idx = np.argsort(omega)
    omega = omega[sort_idx]
    U = U[:,sort_idx] 
    
    # Calculate the normal coordinate transformation matrix
    # for the "dimensionless normal coordinates" which are
    # normalized as V = 0.5 * omega * q^2
    #
    T = U.copy()
    for i in range(nQ):
        ui = U[:,i] 
        Fii = ui.T @ F @ ui
        T[:,i] *= np.sqrt( np.abs(omega[i] / Fii) )

    Qpstr = [f"q{i:d}" for i in range(nQ)]
    
    nctrans = coordsys.LinearTrans(T, t = Q0, Qpstr = Qpstr, name = 'Norm. Coord.')
    
    return omega, nctrans