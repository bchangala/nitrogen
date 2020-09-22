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
            function with signature ``fx(self, X, deriv = 0, out = None, var = None)``.
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
        
    def f(self, X, deriv = 0, out = None, var = None):
        """
        Evaluate the differentiable function.

        Parameters
        ----------
        X : ndarray
            An array of ``m`` input vectors. X has shape (:attr:`nx`, ``m``).
        deriv : int, optional
            All derivatives up through order `deriv` are requested. The default is 0.
        out : ndarray, optional
            The buffer to store the output. This is an ndarray
            with shape (:attr:`nf`, `nd`, ``m``), where `nd` is the number
            of derivatives requested sorted in :mod:`~nitrogen.autodiff`
            lexical order. (See Notes.) The default is None. If None,
            then a new output ndarray is created. The data-type of `out`
            is user-determined and -checked.
        var : list of int
            Calculate derivatives only for those input variables whose index
            is included in `var`. Variables are referred to by their
            0-index: 0, ..., :attr:`nx` - 1. Each index may appear at most once.
            The returned derivative array will provide derivatives in lexical
            order based on the ordering in `var`, not the ordering expected
            in `X`. A value of None is equivalent to `var = [0, 1, ..., nx - 1]`.
            
        Returns
        -------
        out : ndarray
            The result array.
        
        Notes
        -----
        The number of derivatives `nd` equals the binomial coefficient
        (`deriv` + `nvar`, `nvar`), where `nvar = len(var)`.
        
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
        
        # Check var
        if var is None:
            nvar = self.nx  # Use all variables. None will be passed to _feval.
        else:
            if np.unique(var).size != len(var):
                raise ValueError('var cannot contain duplicate elements')
            if min(var) < 0 or max(var) >= self.nx:
                raise ValueError('elements of var must be >= 0 and < nx')
            nvar = len(var)
            
        # Create output array if no output buffer passed
        if out is None:
            nd = adf.nck(deriv + nvar, min(deriv,nvar))
            out = np.ndarray((self.nf, nd, m), dtype = X.dtype)
            
        self._feval(self,X,deriv,out,var) # Evaluate function
        
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
    
    
def _fzero(self, X, deriv = 0, out = None, var = None):
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
        The result (all zeros) with shape (self.nf, nd, m). 
        nd equals `self.nx` + `deriv` choose `deriv`

    """
    
    if var is None:
        nvar = self.nx
    else:
        nvar = len(var)
    
    if out is None:
        nd = adf.nck(nvar + deriv, min(nvar,deriv))
        _,m = X.shape
        out = np.ndarray( (self.nf, nd, m), dtype = X.dtype)
        
    out.fill(0)
    
    return out
