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
        of None indicates arbitrary derivatives are supported.
    zlevel : int
        The zero-level of the function. All derivatives with
        total order greater than `zlevel` are zero. A value
        of None indicates all derivatves may be non-zero.
    
    """
    
    def __init__(self, fx, nf = 1, nx = 1, maxderiv = None, zlevel = None):
        """
        Create a new DFun object.

        Parameters
        ----------
        fx : function
            An instance method implementing the differentiable
            function with signature ``fx(self, X, deriv = 0, out = None, var = None)``.
            See :meth:`DFun.f` for more details.
        nf : int, optional
            The number of output variables of fx. The default is 1.
        nx : int, optional
            The number of input variables of fx. The default is 1.
        maxderiv : int, optional
            The maximum supported derivative of fx(). 
            `maxderiv` = None indicates that arbitrary order
            derivatives are supported. The default is None.
        zlevel : int, optional
            The zero-level of the differentiable function. All derivatives
            with total order greater than `zlevel` are zero. A value
            of None indicates all derivatives may be non-zero. The default is
            None

        """
        self._feval = fx
        self.nf = nf
        self.nx = nx
        self.maxderiv = maxderiv
        self.zlevel = zlevel
        
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
        if self.maxderiv is not None and deriv > self.maxderiv:
            raise ValueError('deriv is larger than maxderiv')
        
        # Check the shape of input X
        n = X.shape[0]
        if n != self.nx:
            raise ValueError(f'Expected input shape ({self.nx:d},...)')
        
        # Check var
        if var is None:
            pass  # Use all variables. None will be passed to _feval.
        elif len(var) == 0:
            pass  # An empty list. This is valid.
        else:
            if np.unique(var).size != len(var):
                raise ValueError('var cannot contain duplicate elements')
            if min(var) < 0 or max(var) >= self.nx:
                raise ValueError('elements of var must be >= 0 and < nx')
                
        return self._feval(X,deriv,out,var) # Evaluate function
    
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
        return cls(_fzero, nf=nf, nx=nx, maxderiv = None, zlevel = None)
    
    def __pow__(self, other):
        """ DFun composition using the ** operator
            
            A ** B returns the composite function
            B(A(x))
            
            Note that ** has right-to-left associativity, i.e.
            A ** B ** C is evaluated as A ** (B ** C) and not
            (A ** B) ** C. 
        """
        return CompositeDFun(other, self)
    
    def __matmul__(self, other):
        """ DFun composition using the @ operator
            
            A @ B returns the composite function
            A(B(x))
            
            Note that @ has left-to-right associativity, i.e.
            A @ B @ C is evaluated as (A @ B) @ C and not
            A @ (B @ C).
        """
        return CompositeDFun(self, other)

class CompositeDFun(DFun):
    
    def __init__(self, A, B):
        """
        Composite function A(B(x))

        Parameters
        ----------
        A : DFun
            The outer function.
        B : DFun
            The inner function.

        """
        if not isinstance(A, DFun) or not isinstance(B, DFun):
            raise ValueError("DFun composition is only supported with other DFun objects")

        if A.nx != B.nf:
            raise ValueError("Incompatible argument count for function composition")
         
        
        # Now create a new DFun
        maxderiv = _composite_maxderiv(A.maxderiv, B.maxderiv)
        zlevel = _composite_zlevel(A.zlevel, B.zlevel)
        
        super().__init__(self._hx, A.nf, B.nx, maxderiv, zlevel)
    
        self.A = A 
        self.B = B 
        
    # Define the derivative array evaluation
    # function for the composition H(x) = A(B(x))
    #
    # Instead of using the Faa di Bruno formula
    # for the derivatives of a composite, we 
    # will simply compute the Taylor series of the
    # composite up to the requested deriv level.
    # It can be shown that the deriv's of the 
    # Taylor series are correct up to the deriv level
    #
    def _hx(self, X, deriv = 0, out = None, var = None):
        
        A = self.A
        B = self.B
        
        # X has shape (B.nx, ...)
        # b has shape (ndb, B.nf, ...)
        b = B.f(X, deriv, out = None, var = var)
        if var is None:
            nvar = B.nx
        else:
            nvar = len(var)
        
        # Calculate the outer function derivatives
        # with respect to all its arguments
        # a has shape (nda, A.nf, ...)
        a = A.f(b[0], deriv, out = None, var = None)
        if A.zlevel is None:
            azlevel = deriv 
        else:
            azlevel = A.zlevel
            
        # Calculate the derivatives of H w.r.t x
        # via a Taylor series
        #
        # H has shape (ndh, A.nf, ...)
        base_shape = X.shape[1:]
        ndh = b.shape[0]
        nfh = A.nf
        if out is None:
            out = np.ndarray((ndh, nfh) + base_shape, dtype = a.dtype)
        out.fill(0)
        #
        # We will need the powers of each of the
        # B.nf intermediate parameters
        bpow = np.ndarray((B.nf,deriv+1), dtype = adf.adarray)
        
        one = np.ones(base_shape, dtype = out.dtype)
        # Determine the zlevel of the derivative
        # arrays of B
        if B.zlevel is None:
            bzlevel = deriv 
        else:
            bzlevel = B.zlevel
        # Zero the value of the B_j derivative arrays
        b[0:1].fill(0)
        # If the b's had a zlevel of 0 (constant)
        # they now have a zlevel of -1 (identically 0)
        if bzlevel == 0:
            bzlevel = -1
            
        # Calculate the powers of the shifted B_j's
        for j in range(B.nf):
            for betaj in range(deriv+1):
                
                if betaj == 0: # B_j ** 0
                    bpow[j,betaj] = adf.const(one, deriv, nvar)
                elif betaj == 1: # B_j ** 1
                    bpow[j,betaj] = adf.array(b[:,j], deriv, nvar,
                                              copyd = False, zlevel = bzlevel) 
                else:
                    bpow[j,betaj] = bpow[j, 1] * bpow[j, betaj-1]
        # bpow now contains adarrays for all required
        # powers of all intermediate arguments B_j
        #
        Aidxtab = adf.idxtab(deriv, B.nf)
        m,_ = Aidxtab.shape
        
        # Initialize the H adf object
        H = np.ndarray((A.nf,) , dtype = adf.adarray)
        for i in range(A.nf):
            H[i] = adf.const(np.zeros(base_shape, dtype = out.dtype), deriv, nvar)
        adfone = adf.const(one, deriv, nvar)
        for k in range(m): # For each derivative of A
            Aidx = Aidxtab[k]
            if np.sum(Aidx) > azlevel:
                # This A derivative and all
                # remaining will be zero
                break
            temp = adfone
            for j in range(B.nf):
                temp = temp * bpow[j, Aidx[j]]
            for i in range(A.nf):
                H[i] = H[i] + a[k,i] * temp
            
        # Copy H data to out
        for i in range(A.nf):
            np.copyto(out[:,i], H[i].d)
        
        return out
    
def _composite_maxderiv(maxA,maxB):
    """
    Determine the maxderiv value of a composite
    DFun A(B(x))

    Parameters
    ----------
    maxA, maxB : int or None
        maxderiv parameter of composite Dfun

    Returns
    -------
    int or None
        maxderiv of composite function

    """
    if maxA is None and maxB is None:
        maxderiv = None
    elif maxA is None:
        maxderiv = maxB
    elif maxB is None:
        maxderiv = maxA
    else:
        maxderiv = min(maxA,maxB)
    
    return maxderiv

def _composite_zlevel(zlevelA, zlevelB):
    """
    Determine the zlevel of a composite
    DFun A(B(x))

    Parameters
    ----------
    zlevelA, zlevelB : int or None
        zlevel parameter of composite DFun

    Returns
    -------
    int or None
        zlevel of composite function

    """
    if zlevelA is None:
        if zlevelB is None:
            zlevel = None
        elif zlevelB < 1: # B is zero or constant
            zlevel = 1 # A(zero or constant) is constant
        else:
            zlevel = None
    elif zlevelA < 1: # If A is zero or constant
        zlevel = zlevelA # then H is zero or constant
    else:
        zlevel = zlevelA * zlevelB # composite polynomials
    
    return zlevel


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

def ndnvar(deriv, var, nX):
    """
    Determine nd and nvar from deriv and var.

    Parameters
    ----------
    deriv : int
        Derivative order
    var : list of int or None
        Requested diff. variables.
    nX : int
        The number of DFun inputs.

    Returns
    -------
    nd : int
        The number of derivatives
    nvar : int
        The number of variables

    """
    
    if var is None:
        var = list(range(nX))
        
    nvar = len(var)
    nd = nderiv(deriv, nvar)
    
    return nd, nvar 
        
def X2adf(X, deriv, var):
    """
    Create adf objects for DFun inputs.

    Parameters
    ----------
    X : ndarray
        (nX, ...) input array
    deriv : int
        Derivative order.
    var : list of int or None
        Requested diff. variables

    Returns
    -------
    x : list of adf
        adf objects for each variable

    """
    
    nX = X.shape[0]
        
    if var is None:
        var = list(range(nX))
    nvar = len(var)
    
    x = []
    for i in range(nX):
        if i in var: # Derivatives requested for this variable
            x.append(adf.sym(X[i], var.index(i), deriv, nvar))
        else: # Derivatives not requested, treat as constant
            x.append(adf.const(X[i], deriv, nvar))
            
    return x
    
def adf2array(Y, out):
    """
    Copy multiple adf objects into a derivative array.

    Parameters
    ----------
    Y : list of adf
        adf objects with data
    out : ndarray
        Output buffer. If None, this will be created.

    Returns
    -------
    out

    """
    
    if len(Y) == 0:
        raise ValueError("Y must not be empty")
    nY = len(Y)
    
    nd = Y[0].nd 
    base_shape = Y[0].d.shape[1:]
    dtype = Y[0].d.dtype
    
    if out is None:
        out = np.ndarray( (nd,nY) + base_shape, dtype = dtype)
        
    for i in range(nY):
        np.copyto(out[:,i] , Y[i].d)
        
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