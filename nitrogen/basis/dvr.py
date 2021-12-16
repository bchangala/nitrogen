"""
dvr.py
"""

from .legDVR import _legDVR, _legDVRwfs
from .fourDVR import _fourDVR, _fourDVRwfs
from .hoDVR import _hoDVR,_hoDVRwfs
from .sincDVR import _sincDVR,_sincDVRwfs

import numpy as np 

__all__ = ['GenericDVR','SimpleDVR','Contracted']

from .genericbasis import GriddedBasis 

class GenericDVR(GriddedBasis):
    """
    A super-class for generic 1D DVRs.
    
    Attributes
    ----------
    num : int
        The number of DVR grid points
    grid : ndarray
        The DVR grid points
    D : ndarray
        First derivative operator.
    D2 : ndarray
        Second derivative operator. 
    
    """
    
    def __init__(self, grid, D, D2):
        
        if grid.ndim != 1:
            raise ValueError('grid must be 1-dimensional')
        
        num = len(grid) 
        
        if D.ndim != 2 or D.shape[0] != num or D.shape[1] != num:
            raise ValueError('D must be (num,num) array')
            
        if D2.ndim != 2 or D2.shape[0] != num or D2.shape[1] != num:
            raise ValueError('D2 must be (num,num) array')
        
        # Initialize the generic GriddedBasis
        super().__init__(grid.reshape((1,-1)), len(grid), wgtfun = None)
        
        self.num = len(grid)
        self.grid = grid 
        self.D = D
        self.D2 = D2
        
        return 
    
    def wfs(self, q):
        """
        Evaluate DVR basis wavefunctions.

        Parameters
        ----------
        q : ndarray
            A 1-D array of coordinate values.

        Returns
        -------
        wfs : ndarray
            An array of shape (`q`.size, :attr:`num`) containing
            the values of the DVR wavefunctions evaluated at 
            coordinate values `q`.

        """
        return self._wfs(q) 
        
    def _wfs(self, q):
        
        raise NotImplementedError("")
    
    def matchfun(self, f):
        """
        Calculate DVR coefficients to match a function at 
        DVR grid points.

        Parameters
        ----------
        f : function
            The function to be matched.

        Returns
        -------
        coeff : ndarray
            A (`num`,1) array with the DVR basis function coefficients.

        """
        
        # Calculate the values of the DVR basis functions
        # at their respective grid-points
        wgts = np.diag(self.wfs(self.grid))
        
        coeffs = (f(self.grid) / wgts).reshape((self.num,1))
        
        return coeffs 
    
    def contract(self, u):
        return Contracted(u, self)
    
    #
    # GriddedBasis methods
    #
    def _basis2grid(self, x, axis = 0):
        return x # DVR is already in the grid representation 
    def _grid2basis(self,x, axis = 0):
        return x # DVR is already in the grid representation 
    def _basis2grid_d(self,x,var, axis = 0):
        return self._d_grid(x, var, axis)
    def _grid2basis_d(self, x,var, axis = 0):
        return self._dH_grid(x, var, axis) 
    def _d_grid(self,x,var,axis = 0):
        y = np.tensordot(self.D, x, axes = (1,axis))
        y = np.moveaxis(y, 0, axis)
        return y 
    def _dH_grid(self,x,var,axis = 0):
        y = np.tensordot(self.D.conj().T, x, axes = (1,axis))
        y = np.moveaxis(y, 0, axis)
        return y 


class SimpleDVR(GenericDVR):
    """
    Standard 1D DVRs.
    
    Attributes
    ----------
    start : float
        The DVR grid starting value.
    stop : float
        The DVR grid stopping value.
    basis : {'sinc','ho','fourier', 'legendre'}
        The DVR basis type.
    
    """
    
    def __init__(self, start = 0, stop = 1, num = 10, basis = 'sinc'):
        """
        Create a DVR object.

        Parameters
        ----------
        start : float, optional
            The DVR grid starting value. The default is 0.
        stop : float, optional
            The DVR grid stopping value. The default is 1.
        num : int, optional
            The number of DVR grid points. The default is 10.
        basis : {'sinc','ho','fourier','legendre'}, optional
            The DVR basis type. The default is 'sinc'. See Notes
            for details.
            
        Notes
        -----
        Each `basis` option constructs a one-dimensional DVR over a grid
        defined by `start`, `stop`, and `num`. All basis types have
        a simple :math:`dq` volume element. Descriptions for each follow:
        
        ========================  =========================
        `basis` value             Description
        ========================  =========================
        ``'sinc'``                :math:`\\text{sinc}` basis
        ``'ho'``                  Harmonic oscillator DVR.
        ``'fourier'``             Fourier DVR (a periodic version of ``'sinc'``).
        ``'legendre'``            Legendre polynomial DVR.
        ========================  =========================
        
        

        """
        
        if stop <= start:
            raise ValueError("stop value must be larger than start value")
        if num < 2 :
            raise ValueError("num must be >= 2")
        

        basis = basis.lower()
        
        if basis == 'sinc' : 
            grid, D, D2 = _sincDVR(start, stop, num)
            _wfs_fun = lambda q : _sincDVRwfs(q, start, stop, num)
        elif basis == 'ho' : 
            grid, D, D2 = _hoDVR(start, stop, num)
            _wfs_fun = lambda q : _hoDVRwfs(q, start, stop, num)
        elif basis == 'fourier' : 
            grid, D, D2 = _fourDVR(start, stop, num)
            _wfs_fun = lambda q : _fourDVRwfs(q, start, stop, num)
        elif basis == 'legendre' : 
            grid, D, D2 = _legDVR(start, stop, num)
            _wfs_fun = lambda q : _legDVRwfs(q, start, stop, num)
        else:
            raise ValueError("basis type '{}' not recognized".format(basis))
            
        super().__init__(grid, D, D2)

        self.basis = basis
        self.start = start 
        self.stop = stop 
        
        self._wfs_fun = _wfs_fun
        
    def _wfs(self, q):
        
        return self._wfs_fun(q)
        
    
class Contracted(GenericDVR):
    """
    A contracted basis DVR class (usually for PO-DVRs).
    
    Attributes
    ----------
    prim_dvr : GenericDVR
        The primitive basis
    W : ndarray
        The transformation matrix from the contracted DVR
        to the primitive DVR 
    
    """
    
    def __init__(self, U, prim_dvr):
        """
        

        Parameters
        ----------
        U : (n,m) ndarray
            A unitary operator defining the contracted basis.
            `n` is the size of the primitive basis. 
            `m` is the size of the contracted basis. 
        prim_dvr : GenericDVR
            The primitive DVR


        """
        
        #
        # Construct the coordinate operator in the 
        # contracted basis 
        Q = (U.conj().T * prim_dvr.grid) @ U 
        q,u = np.linalg.eigh(Q)   # u takes a vector from the new DVR 
                                  # to the contracted "fbr"
        
        q = np.reshape(q, (-1,))  # The new DVR grid 
        
        W = U @ u # w takes a vector from the new DVR to the old DVR 
        
        d = W.conj().T @ prim_dvr.D @ W
        d2 = W.conj().T @ prim_dvr.D2 @ W
        
        super().__init__(q, d, d2) 
        
        self.prim_dvr = prim_dvr 
        self.W = W 
    
    def _wfs(self, q):
        
        f = self.prim_dvr.wfs(q) # f is (q.size, primitive num)
    
        return f @ self.W 
    
        