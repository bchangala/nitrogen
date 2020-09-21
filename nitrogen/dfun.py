"""
nitrogen.dfun
-------------

This module implements the class DFun, which 
is used to wrap general differentiable, multi-variate
functions.

"""


import numpy as np
import nitrogen.autodiff.forward as adf


class DFun:
    
    """
    A differentiable multi-variate function wrapper.
    
    Attributes
    ----------
    nf : int
        The number of output variables.
    nx : int
        The number of input variables.
    maxderiv : int
        The maximum supported derivative order. A value
        of -1 indicates arbitrary derivatives are supported.
    
    """
    
    def __init__(self, fx, nf = 1, nx = 1, maxderiv = -1):
        """
        Create a new DFun object.

        Parameters
        ----------
        fx : function
            An instance method implementing the differential
            function with signature ``fx(self, X, deriv = 0, out = None)``.
            See :meth:`DFun.f` for more details.
        nf : int, optional
            The number of output variables of fx. The default is 1.
        nx : int, optional
            The number of input variables of fx. The default is 1.
        maxderiv : int, optional
            The maximum supported derivative of fx(). 
            `maxderiv` = -1 indicates that arbitrary order
            derivatives are supported. The default is -1.

        """
        self._feval = fx
        self.nf = nf
        self.nx = nx
        self.maxderiv = maxderiv
        
    def f(self, X, deriv = 0, out = None):
        """
        Evaluate the differentiable function.

        Parameters
        ----------
        X : ndarray
            An array of ``m`` input vectors. X has shape (:attr:`nx`, ``m``).
        deriv : int, optional
            The requested derivative order. The default is 0.
        out : ndarray, optional
            The buffer to store the output. This is an ndarray
            with shape (`nd`, :attr:`nf`, ``m``), where `nd` is the number
            of derivatives requested sorted in :mod:`~nitrogen.autodiff`
            lexical order. (See Notes.) The default is None. If None,
            then a new output ndarray is created. The data-type of `out`
            is user-determined and -checked.

        Returns
        -------
        out : ndarray
            The result array.
        
        Notes
        -----
        The number of derivatives `nd` equals the binomial coefficient
        (`deriv` + :attr:`nx`, :attr:`nx`).

        """
        # Check the requested derivative order
        if deriv < 0:
            raise ValueError('deriv must be non-negative')
        if deriv > self.maxderiv and self.maxderiv != -1:
            raise ValueError('deriv is larger than maxderiv')
        
        # Check the shape of input X
        n,m = X.shape
        if n != self.nx:
            raise ValueError('X must have shape (nx,m)')
        
        # Create output array is no output buffer passed
        if out is None:
            nd = adf.nck(deriv+self.nx, min(deriv,self.nx))
            out = np.ndarray((nd,self.nf,m))
            
        self._feval(self,X,deriv,out) # Evaluate function
        
        return out
    
    @classmethod 
    def zerofun(cls, nf=1, nx=1):
        """
        Construct a :class:`DFun` object for a 
        zero function.

        Parameters
        ----------
        nf : int, optional
            The number of output values. The default is 1.
        nx : int, optional
            The number of input values. The default is 1.

        Returns
        -------
        DFun
            A :class:`DFun` object for the zero function
            of `nx` variables returning `nf` values.

        """
        return cls(_fzero, nf=nf, nx=nx, maxderiv = -1)
    
    
def _fzero(self, X, deriv = 0, out = None):
    """
    A dummy zero function that can be used to create a :class:`DFun`
    with any values of nf and nx. Note that :func:`_fzero` is
    explicitly written as an instance method.

    Parameters
    ----------
    X : ndarray
        Input array with shape (self.nx, m)
    deriv : int, optional
        Maximum derivative order. The default is 0.
    out : ndarray, optional
        Output location. If None, a new ndarray will
        be created. The default is None.

    Returns
    -------
    out : ndarray
        The result (all zeros) with shape (nd, self.nf, m). 
        nd equals `self.nx` + `deriv` choose `deriv`

    """
    
    if out is None:
        nd = adf.nck(self.nx + deriv, min(self.nx,deriv))
        _,m = X.shape
        out = np.ndarray( (nd, self.nf, m), dtype = X.dtype)
        
    out.fill(0)
    
    return out
