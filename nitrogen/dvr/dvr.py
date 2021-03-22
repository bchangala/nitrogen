
from .legDVR import _legDVR, _legDVRwfs
from .fourDVR import _fourDVR, _fourDVRwfs
from .hoDVR import _hoDVR,_hoDVRwfs
from .sincDVR import _sincDVR,_sincDVRwfs

from .ndbasis import NDBasis, SinCosBasis, LegendreLMCosBasis, RealSphericalHBasis, \
    Real2DHOBasis

import numpy as np 

__all__ = ['DVR', 
           'NDBasis','SinCosBasis', 'LegendreLMCosBasis',
           'RealSphericalHBasis','Real2DHOBasis']

class DVR:
    """
    A 1D DVR basis class
    
    Attributes
    ----------
    start : float
        The DVR grid starting value.
    stop : float
        The DVR grid stopping value.
    num : int
        The number of DVR grid points.
    basis : {'sinc','ho','fourier', 'legendre'}
        The DVR basis type.
    grid : ndarray
        The DVR grid points.
    D : ndarray
        First derivative operator in DVR representation.
    D2 : ndarray
        Second derivative operator in DVR representation.
    
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
            The DVR basis type. The default is 'sinc'.

        """
        
        if stop <= start:
            raise ValueError("stop value must be larger than start value")
        if num < 2 :
            raise ValueError("num must be >= 2")
        
        self.start = start 
        self.stop = stop 
        self.num = num 
        self.basis = basis.lower()
        
        if self.basis == 'sinc' : 
            self.grid, self.D, self.D2 = _sincDVR(start, stop, num)
            self.wfs = lambda q : _sincDVRwfs(q, self.start, self.stop, self.num)
        elif self.basis == 'ho' : 
            self.grid, self.D, self.D2 = _hoDVR(start, stop, num)
            self.wfs = lambda q : _hoDVRwfs(q, self.start, self.stop, self.num)
        elif self.basis == 'fourier' : 
            self.grid, self.D, self.D2 = _fourDVR(start, stop, num)
            self.wfs = lambda q : _fourDVRwfs(q, self.start, self.stop, self.num)
        elif self.basis == 'legendre' : 
            self.grid, self.D, self.D2 = _legDVR(start, stop, num)
            self.wfs = lambda q : _legDVRwfs(q, self.start, self.stop, self.num)
        else:
            raise ValueError("basis type '{}' not recognized".format(basis))
    
    
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
        pass # This should be replaced by an instance method for a given DVR basis type
            


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



    
    