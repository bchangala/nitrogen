
from .legDVR import _legDVR, _legDVRwfs
from .fourDVR import _fourDVR, _fourDVRwfs
from .hoDVR import _hoDVR,_hoDVRwfs
from .sincDVR import _sincDVR,_sincDVRwfs

import numpy as np 

__all__ = ['DVR', 'NDBasis']

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


class NDBasis:
    """
    
    A generic multi-dimensional finite basis representation
    supporting quadrature integration/transformation.
    
    Attributes
    ----------
    nd : int
        The number of dimensions (i.e. coordinates)
    Nb : int 
        The number of basis functions
    Nq : int
        The number of quadrature points
    qgrid : (`nd`,`Nq`) ndarray
        The quadrature grid.
    wgt : (`Nq`,) ndarray
        The quadrature weights.
    bas : (`Nb`,`Nq`) ndarray
        The basis functions evaluated on the quadrature grid.
    basisfun : DFun
        An `Nb`-output-valued DFun of the basis functions 
        with `nd` input variables.
    wgtfun : DFun
        The weight function associated with matrix element
        integrals of the basis.
        
    """    
    
    def __init__(self, basisfun, wgtfun, qgrid, wgt):
        """
        Initialize a generic NDBasis.

        Parameters
        ----------
        basisfun : DFun
            The basis function DFun.
        wgtfun : DFun
            The weight function.
        qgrid : (`nd`, `Nq`) ndarray
            The quadrature points.
        wgt : (`Nq`,) ndarray
            The quadrature weights.
            
        Returns
        -------
        None.

        """
        self.basisfun = basisfun 
        self.wgtfun = wgtfun 
        self.qgrid = qgrid 
        self.wgt = wgt
        
        self.nd = basisfun.nx  # The number of dimensions
        self.Nb = basisfun.nf  # The number of basis functions
        self.Nq = qgrid.shape[1] # The number of quadrature points
        
        
        if qgrid.shape != (self.nd, self.Nq):
            return ValueError("qgrid is the wrong size!")
        if wgtfun.nx != basisfun.nx or wgt.nf != 1 :
            return ValueError("invalid wgtfun")
        if wgt.shape != (self.Nq,):
            return ValueError("wgt is the wrong size!")
        
        # Calculate the `bas` grid
        self.bas = basisfun.val(qgrid)
        
        
    
    def fbrToQuad(self, v, axis = 0):
        """
        Transform an axis from the FBR
        to the quadrature grid representation.

        Parameters
        ----------
        v : (...,`Nb`,...) ndarray
            An array with the `axis` index spanned
            by this basis.

        Returns
        -------
        w : (..., `Nq`, ...) ndarray
            The transformed array.

        """
        
        return self._fbrToQuad(v, axis)
    
    def quadToFbr(self, w, axis = 0):
        """
        Transform an axis from the quadrature representation
        to the FBR.

        Parameters
        ----------
        w : (...,`Nq`,...) ndarray
            An array with the `axis` index spanned
            by this quadrature.

        Returns
        -------
        v : (..., `Nb`, ...) ndarray
            The transformed array.

        """
        
        return self._quadToFbr(w, axis)
    
    def _fbrToQuad(self, v, axis = 0):
        """ The default implemention of
        the FBR to quadrature transformation"""
        
        U = self.bas.T # An (Nq,Nb) array 
        
        # Apply the FBR-to-grid transformation
        # to the axis
        w = np.tensordot(U,v, axes = (1,axis) )
        w = np.moveaxis(w, 0, axis)
        
        # Broadcast the wgt's
        shape = [1] * v.ndim 
        shape[axis] = self.Nq 
        w *= np.sqrt(self.wgt).reshape(tuple(shape))
        
        return w
    
    def _quadToFbr(self, w, axis = 0):
        
        """ The default implemention of
        the quadrature to FBR transformation"""
        
        U = self.bas # An (Nb, Nq) array 
        
        # Broadcast the wgt's
        shape = [1] * w.ndim 
        shape[axis] = self.Nq
        w = w * np.sqrt(self.wgt).reshape(tuple(shape))
        
        # Apply the grid-to-FBR transformation
        # to the axis
        v = np.tensordot(U,w, axes = (1,axis) )
        v = np.moveaxis(v, 0, axis)
        
        return v
    
    