"""
nitrogen.vpt
-----------------

Vibrational perturbation theory and harmonic oscillator methods

"""



import numpy as np 
import nitrogen.constants

from . import ho_core
from .ho_core import *  # Import the core namespace

from . import td_core
from .td_core import *  # Import the core namespace

from . import cfourvib # CFOUR interface routines and parsers

__all__ = ['calc_rectilinear_modes']
__all__ += ho_core.__all__
__all__ += td_core.__all__


def calc_rectilinear_modes(hes, mass, hbar = None, norm = 'dimensionless'):
    """
    Calculate the rectilinear normal modes
    and energies
    
    Parameters
    ----------
    hes : array_like
        The (3*N,3*N) Cartesian Hessian matrix.
    mass : array_like
        The N masses
    hbar : float, optional
        The value of :math:`\\hbar`. If None (default),
        NITROGEN units will be assumed.
    norm : {'dimensionless', 'mass-weighted'}
        The normalization convention of the displacement
        vectors. 
        
    Returns
    -------
    w : (N,) ndarray
        The harmonic frequencies, in energy units.
    R : (3*N,3*N) ndarray
        Each column of `R` is the displacement
        vector for the corresponding normal mode. 
        
    Notes
    -----
    For norm = 'dimensionless', the displacement vectors
    are those of the non-mass-weighted Cartesian coordinates
    with respect to dimensionless, normalized coordinates.
    In these coordinates, the potential energy surface is
    
    ..  math::
        V = \\sum_i \\frac{1}{2} \\omega_i q_i^2
        
    For norm = 'mass-weighted', the displacement vectors
    are those for the mass-weighted Cartesian coordinates
    and equal the eigenvectors of the mass-weighted Hessian, i.e.
    the traditional :math:`\\mathbf{L}` array.
    """
    
    hes = np.array(hes)
    if hes.ndim != 2 or hes.shape[0] != hes.shape[1]:
        raise ValueError("hes has an unexpected shape")
    if hes.shape[0] % 3 != 0:
        raise ValueError("the shape of hes must be a multiple of 3")
        
    N = hes.shape[0] // 3
    if N < 1:
        raise ValueError("there must be at least 1 atom")
    
    m = np.repeat(mass, [3]*N)
    iMrt = np.diag(m**-0.5)
    
    # Mass-scale the Cartesian hessian
    H = iMrt @ hes @ iMrt 
    
    # Diagonalize the mass-weighted hessian
    lam,L = np.linalg.eigh(H) 
    #
    # The eigenvalues equal the square of the angular frequency
    #
    
    if hbar is None:
        hbar = nitrogen.constants.hbar 

    # Calculate the harmonic energies    
    w = hbar * np.sqrt(abs(lam))

    # Calculate the Cartesian displacements
    # for the dimensionless normal coordinates
    #
    T = iMrt @ L
    for i in range(9):
        v = T[:,i] 
        a = v.T @ hes @ v 
        T[:,i] *= np.sqrt(abs(w[i] / a))
    
    if norm == 'dimensionless':
        return w, T 
    elif norm == 'mass-weighted':
        return w, L 
    else:
        raise ValueError('unexpected norm option')
        
     
    
    
    