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
            An array of input values. `X` has shape (:attr:`nx`, ...).
        deriv : int, optional
            All derivatives up through order `deriv` are requested. The default is 0.
        out : ndarray, optional
            The buffer to store the output. This is an ndarray
            with shape (`nd`, :attr:`nf`, ...), where `nd` is the number
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
            The result array in autodiff format with shape
            (`nd`, :attr:`nf`, ...)
        
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
        n = X.shape[0]
        if n != self.nx:
            raise ValueError('X must have shape (nx,...)')
        
        # Check var
        if var is None:
            nvar = self.nx  # Use all variables. None will be passed to _feval.
        elif len(var) == 0:
            nvar = 0        # An empty list
        else:
            if np.unique(var).size != len(var):
                raise ValueError('var cannot contain duplicate elements')
            if min(var) < 0 or max(var) >= self.nx:
                raise ValueError('elements of var must be >= 0 and < nx')
            nvar = len(var)
            
        # Create output array if no output buffer passed
        if out is None:
            nd = nderiv(deriv, nvar)
            out = np.ndarray((nd, self.nf) + X.shape[1:], dtype = X.dtype)
            
        self._feval(X,deriv,out,var) # Evaluate function
        
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
        Input array with shape (self.nx, ...)
    deriv : int, optional
        Maximum derivative order. The default is 0.
    out : ndarray, optional
        Output location. If None, a new ndarray will
        be created. The default is None.

    Returns
    -------
    out : ndarray
        The result (all zeros) with shape (nd, self.nf, ...). 
        nd equals ``len(var)`` + `deriv` choose `deriv`

    """
    
    if var is None:
        nvar = self.nx
    else:
        nvar = len(var)
    
    if out is None:
        nd = nderiv(deriv, nvar)
        base_shape = X.shape[1:]
        out = np.ndarray( (nd, self.nf) + base_shape, dtype = X.dtype)
        
    out.fill(0)
    
    return out

def nderiv(deriv, nvar):
    """
    The number of derivatives up to order `deriv`,
    inclusively, in `nvar` variables. This equals
    the binomial coefficient (`deriv` + `nvar`, `nvar`).

    Parameters
    ----------
    deriv : int
        The maximum derivative order.
    nvar : int
        The number of independent variables.

    Returns
    -------
    np.uint64
        The number of derivatives.

    """
    n = deriv + nvar 
    k = min(deriv, nvar)
    return adf.ncktab(n,k)[n,k]

def sym2invdet(S, deriv, nvar):
    """
    Calculate the inverse and determinant
    of a symmetric matrix. If S is real,
    then it must be positive definite. If S
    is complex, it must be invertible.

    Parameters
    ----------
    S : ndarray
        The derivative array of a symmetric matrix
        in packed (lower triangle, row-order) storage.
        `S` has a shape of (nd, nS, ...).
        The second dimension is the packed dimension.
    deriv : int
        The maximum derivative order.
    nvar : int
        The number of independent variables.

    Returns
    -------
    iS : ndarray
        The derivative array of the matrix inverse of S
        in packed storage with shape (nd, nS, ...)
    det : ndarray
        The derivative array for det(S), with 
        shape (nd, ...)

    """
    
    if np.ndim(S) < 2 :
        raise ValueError("S must have at least 2 dimensions")
    
    nd = S.shape[0]
    if nd != nderiv(deriv,nvar):
        raise ValueError("The first dimension of S is inconsistent with deriv and nvar")
    
    nS = S.shape[1]
    if nS < 1:
        raise ValueError("S must have a second dimension of at least 1")
    N = adf.n2N(nS) # The rank of the matrix S
    
    # We will carry out the computation using 
    # adarrays / forward ad
    
    # Create an adarray using S
    # This will have new, copied data
    Sad = adf.array(S, deriv, nvar, copyd = True)
    
    # Promote the matrix dimension
    Spacked = adf.ndize1(Sad)
    
    # 1) Compute the Cholesky decomposition
    #    Stored in Spacked
    adf.chol_sp(Spacked, out = Spacked)
    
    # 2) Compute the determinant
    #    This is the product of the squares of the diagonal entries
    k = 0
    for i in range(N):
        if i == 0:
            det = Spacked[k] 
        else:
            det = det * Spacked[k]
        k = k + (i+2)
    det = det * det
    # det now equals the determinant of S
    
    # 3) Compute the inverse of the Cholesky decomposition
    #    This overwrites Spacked
    adf.inv_tp(Spacked, out = Spacked)
    # 4) Compute the inverse of the original matrix
    #    This overwrites Spacked
    adf.ltl_tp(Spacked, out = Spacked)
    
    # 5) Copy this data to the output array
    iS = np.empty(S.shape, dtype = S.dtype)
    for i in range(nS):
        # S[:, i, ...] <-- Spacked[i].d
        np.copyto(iS[:,i], Spacked[i].d)
    
    # S now contains the inverse of the original matrix
    # 
    # finally, return the ln(det) derivative array
    #
    return iS, det.d.copy()