# -*- coding: utf-8 -*-
"""
Simple model Hamiltonians
"""

__all__ = ['Polar2D']

import numpy as np 
from scipy.sparse.linalg import LinearOperator
import nitrogen.constants

class Polar2D(LinearOperator):
    """
    
    A Hamiltonian for a particle in two dimensions
    with polar coordinates :math:`(r,\\phi)` represented by the
    :class:`~nitrogen.basis.Real2DHOBasis`
    two-dimensional harmonic oscillator basis set. 
    The differential operator is
    
    .. math::
       \\hat{H} = -\\frac{\\hbar^2}{2m} \\left[\\partial_r^2 + 
           \\frac{1}{r}\\partial_r + \\frac{1}{r^2}\\partial_\\phi^2\\right]
           + V(r,\\phi)
           
    with respect to integration as :math:`\int_0^\infty\,r\,dr\, \int_0^{2\pi}\,d\\phi`.
           
    
    """
    
    def __init__(self, Vfun, mass, vmax, R, ell = None, Nr = None, Nphi = None, hbar = None):
        """
        Class initializer.

        Parameters
        ----------
        Vfun : function or DFun
            A function evaluating the potential energy for a (2,...)-shaped
            input array containing :math:`(r,\\phi)` values and returning
            a (...)-shaped output array or a DFun with `nx` = 2.
        mass : float
            The mass.
        vmax : int
            Basis set parameter, see :class:`~nitrogen.basis.Real2DHOBasis`.
        R : float
            Basis set parameter, see :class:`~nitrogen.basis.Real2DHOBasis`.
        ell : int, optional
            Basis set parameter, see :class:`~nitrogen.basis.Real2DHOBasis`.
            The default is None.
        Nr,Nphi : int, optional
            Quadrature parameter, see :class:`~nitrogen.basis.Real2DHOBasis`.
            The default is None.
        hbar : float, optional
            The value of :math:`\\hbar`. If None, the default value in 
            standard NITROGEN units is used (``n2.constants.hbar``).

        """
        
        # Create the 2D HO basis set
        basis = nitrogen.basis.Real2DHOBasis(vmax, R, ell, Nr, Nphi)
        NH = basis.Nb # The number of basis functions 
        
        # Calculate the potential energy surface on the 
        # quadrature grid 
        try: # Attempt DFun interface
            V = Vfun.f(basis.qgrid, deriv = 0)[0,0]
        except:
            V = Vfun(basis.qgrid) # Attempt raw function. Expecting (Nq,) output
            
        # Parse hbar 
        if hbar is None:
            hbar = nitrogen.constants.hbar 
        
        # Pre-construct the KEO operator matrix 
        T = np.zeros((NH,NH))
        for i in range(NH):
            elli = basis.ell[i]
            ni = basis.n[i]
            
            for j in range(NH):
                ellj = basis.ell[j]
                nj = basis.n[j]
                
                # Check \Delta \ell = 0
                if elli != ellj:
                    continue
                
                # Check for diagonal 
                if ni == nj:
                    T[i,j] = -basis.alpha*(2*ni + abs(elli) + 1)
                elif abs(ni-nj) == 1: # Check for off-diagonal
                    n = min(ni,nj)
                    T[i,j] = basis.alpha * np.sqrt((n+1)*(n+abs(elli)+1))
        #
        # Now finish with prefactor        
        T *= -(hbar**2 / (2*mass))
        
        # Define the required LinearOperator attributes
        self.shape = (NH,NH)
        self.dtype = V.dtype 
        
        # Additional attributes
        self.NH = NH        # The size of the Hamiltonian matrix
        self.basis = basis  # The NDBasis object
        self.V = V          # The PES quadrature grid
        self.KEO = T          # The KEO matrix 
        self.mass = mass    # Mass
        self.hbar = hbar    # The value of hbar.
    
    def _matvec(self, x):
        """
        Hamiltonian matrix-vector product routine
        """
        
        x = x.reshape((self.NH,))
        #
        # Compute kinetic energy operator
        #
        tx = self.KEO @ x 
        
        # 
        # Compute potential energy operator
        #
        xquad = self.basis.fbrToQuad(x,axis = 0) # xquad has shape (Nq,)
        vx = self.basis.quadToFbr(self.V * xquad) # vx has shape (NH,)
        
        return tx + vx 