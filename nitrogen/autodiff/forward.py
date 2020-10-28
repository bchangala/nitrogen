"""
nitrogen.autodiff.forward
-------------------------

This module implements a simple forward accumulation
model for automatic differentiation. Its main object
is the ``adarray`` class.

"""

import numpy as np
import warnings

class adarray:
    
    """
    Forward-type automatic differentiation object.
    
    Attributes
    ----------
    k : int
        The maximum derivative order.
    ni : int 
        The number of independent variables.
    nck : ndarray
        A binomial coefficient table.
    idx : ndarray
        The multi-index table for derivatives of
        `ni` variables to order `k`.
    nd : int
        The number of unique derivatives.
    d : ndarray
        The derivative array, whose first index is the
        generalized derivative index. (See Notes.)
    zlevel : int
        The highest non-zero derivative order. If -1,
        then this adarray is identically zero.
        
        
    Notes
    -----
    The derivative information is stored in the :attr:`d` attribute,
    an ndarray whose first index is a generalized derivative
    index. ``d[0]`` is the the value of the base array and ``d[i]``
    with ``i`` > 0 are the derivatives stored in *lexical* order. 
    This ordering sorts derivatives first by their total degree: zeroth
    derivatives (the value), then first derivatives, then second 
    derivatives, and so on. Within a group of derivatives of a given 
    order, they are sorted by the derivative order with respect to the
    first independent variable, then by the order of the second, and 
    so on. This ordering is the same as that of :attr:`idx`
    
    The values of higher-order derivatives are stored by convention
    with a factor equal to the inverse of the multi-index factorial, i.e.
    a derivative with multi-index ``[2, 0, 1, 3]`` would be stored as the 
    corresponding derivative divided by :math:`2!\\times 0!\\times 1!\\times 3!`
    
    """
    
    def __init__(self,base_shape,k,ni, nck = None, idx = None, dtype = np.float64,
                 d = None, zlevel = None):
        """
        Create a new adarray object.

        Parameters
        ----------
        base_shape : tuple of int
            Shape of base array. adarray.d will have shape
            ``(nd,) + base_shape``. Shape may be ().
        k : int
            Maximum derivative degree. `k` must be greater
            or equal to 0.
        ni : int
            Number of independent variables. `n` must be
            greater than or equal to 1.
        nck : ndarray, optional
            Return value of ncktab(nmax,kmax) with
            ``nmax >= k + ni`` and ``kmax >= min(k, ni)``. 
            If None, this will be calculated.
        idx : ndarray, optional
            Return value of ``idxtab(k,ni)``.
            If None, this will be calculated.
        dtype : data-type, optional
            Data-type of initialized derivative array
        d : ndarray, optional
            A pre-allocated derivative array. If provided,
            `dtype` will be ignored and `d` must have 
            a shape equal to ``(nd,) + base_shape``
        zlevel : int, optional
            The zero level indicator. If None, this
            will be set to `k`. The default is None.
        
        See Also
        --------
        ncktab : Binomial coefficient table
        idxtab : Multi-index table

        """
        
        self.k = k
        self.ni = ni
        
        if nck is None:
            self.nck = ncktab(k + ni, min(k,ni))
        else:
            self.nck = nck # no copy
        
        if idx is None:
            self.idx = idxtab(k,ni)
        else:
            self.idx = idx # no copy
        
        self.nd = self.nck[k + ni, min(ni,k)]
        
        if d is None:
            self.d = np.empty((self.nd,) + base_shape, dtype = dtype)
        else:
            if d.shape != (self.nd,) + base_shape:
                raise ValueError("d does not have the correct shape")
            self.d = d 
            
        if zlevel is None:
            self.zlevel = k 
        else:
            self.zlevel = zlevel 
    
    def copy(self, out = None):
        """
        Copy an adarray object.
        
        Parameters
        ----------
        out : adarray
            Output location. If None, this will be created
            if None. The default is None.
            
        Returns
        -------
        adarray
            An adarray object, with ``d`` attribute
            copied via ``d.copy()``.

        """
        if out is None:
            out = adarray(self.d.shape[1:],self.k,self.ni,
                    self.nck,self.idx,self.d.dtype,self.d.copy())
        else:
            # We assume out has the right shape, data-type, etc.
            np.copyto(out.d, self.d)
            out.zlevel = self.zlevel 
            
        return out
        
    # Define binary operators: +, -, ...
    # ADDITION
    def __add__(self,other):
        """ z = self + other """ 
        if type(other) == type(self):
            z = add(self, other) # adarray add
        else:
            z = self.copy()
            z.d[0] += other # Attempt ndarray iadd
            z.zlevel = max(z.zlevel, 0) # Assume other is non-zero
        return z
    def __radd__(self,other):
        """ z = other + self """
        return self.__add__(other)
    
    # MULTIPLICATION
    def __mul__(self,other):
        """ z = self * other """    
        if type(other) == type(self):
            z = mul(self, other)
        else:
            z = self.copy()
            z.d *= other # Attempt ndarray imul
            # assuming other to be constant.
            # NumPy broadcasting is important here.
            # If other is a scalar, then everything is multiplied
            # by it. If other is an ndarray of the same base_shape
            # as z, then it will be broadcast over each
            # derivative, as desired.
            #
            # zlevel is unchanged by constant multiplication
        return z
    def __rmul__(self,other):
        """ z = other * self """
        return self.__mul__(other)
    
    # DIVISION
    def __truediv__(self, other):
        """ z = self / other """
        if type(other) == type(self):
            z = div(self,other)
        else:
            z = self.copy()
            z.d /= other # ndarray idiv, using broadcasting
            # zlevel is unchanged by constant division
        return z
    def __rtruediv__(self, other):
        """ z = other / self """
        # z = (self**-1.0) * other
        return (powf(self,-1.0)).__mul__(other)
    
    # SUBTRACTION
    def __sub__(self,other):
        """ z = self - other """        
        return self.__add__(-other)
    def __rsub__(self,other):
        """ z = other - self """
        return (-self).__add__(other)
    
    # UNARY OPERATIONS
    def __neg__(self):
        """ z = -self """
        z = self.copy()
        np.negative(z.d, out = z.d) #z.d = -z.d
        # zlevel is unchanged by negation
        return z
    def __pos__(self):
        """ z = +self"""
        return self.copy()

def array(d,k,ni,copyd = False,zlevel = None):
    """
    Create an adarray object from a raw derivative array.

    Parameters
    ----------
    d : ndarray
        The derivative array with shape (nd,...)
    k : int
        The maximum derivative order.
    ni : int
        The number of independent variables.
    copyd : boolean, optional
        If True, a copy of `d` will be made for returned
        adarray. If False, the adarray will use 
        the same reference. The default is False.
    zlevel : int, optional
        The zero-level of `d`. If None, this
        will be set safely to `k`. The default is None.

    Returns
    -------
    adarray

    """   
    
    if type(d) != np.ndarray:
        raise TypeError("d must be an ndarray")
        
    base_shape = d.shape[1:]
    
    if copyd:
        dinit = d.copy()
    else:
        dinit = d # use reference
        
    # Check the length of the first dimension
    if np.ndim(d) < 1:
        return ValueError("d must have at least 1 dimension")
    
    nck = ncktab(k + ni, min(k,ni))
    if d.shape[0] != nck[k+ni,min(k,ni)]:
        return ValueError("The first dimension of d has an incorrect length")
    
    if zlevel is None:    
        zlevel = k
    
    return adarray(base_shape, k, ni, nck, None, d.dtype, dinit, zlevel)
    
def ndize1(x):
    """
    Create a 1D ndarray of adarray objects from a 
    single adarray by promoting the first index
    of the base shape.

    Parameters
    ----------
    x : adarray
        The original adarray.

    Returns
    -------
    ndarray
        The 1D array of adarrays.
        
    Notes
    -----
    The derivative arrays of the new adarray objects
    are views of `x.d`.

    """
    
    if np.ndim(x.d) < 2:
        raise ValueError("The base shape must have at least 1 dimension")
    
    n = x.d.shape[1]    # the size of the new ndarray
    
    X = np.ndarray((n,), dtype = adarray)
    for i in range(n):
        X[i] = array(x.d[:,i], x.k, x.ni, 
                     copyd = False, zlevel = x.zlevel) # keep references
    
    return X    
        

def nck(n,k):
    """
    Calculate the binomical coefficient (`n` choose `k`).
    
    This function uses a simple recursive algorithm. Use of 
    :func:`ncktab` may be significantly faster (i.e. 
    ``ncktab(n,k)[n,k]``.)

    Parameters
    ----------
    n, k : int
        Arguments of the binomial coefficient.
   
    Returns
    -------
    np.uint64
        The binomial coefficient (`n` choose `k`).
        
    Examples
    --------
    >>> nck(4,2)
    6
    
    >>> nck(4,0)
    1
    
    See Also
    --------
    ncktab : Binomial coefficient table

    """
    
    if k > n or k < 0 or n < 0:
        return np.uint64(0)
    elif n == 0 :
        return np.uint64(1)
    elif k == 0 or k == n:
        return np.uint64(1)
    else:
        return nck(n-1,k-1) + nck(n-1,k)

def ncktab(nmax,kmax=None):
    """
    Binomial coefficient table

    Parameters
    ----------
    nmax : int
        Maximum value of n (first argument of coeff).
    kmax : int, optional
        Maximum value of k. If None, its value is
        set to nmax. The default is None.

    Returns
    -------
    tab : ndarray
        2D array of shape (`nmax` + 1, `kmax` + 1) containing
        the binomial coefficients, `tab[n,k]` = 
        (`n` choose `k`). The values of invalid elements
        (i.e. `k` > `n`) are undefined. The data-type is
        ``np.uint64``.
        
    Examples
    --------
    >>> ncktab(3)
    array([[1, 0, 0, 0],
           [1, 1, 0, 0],
           [1, 2, 1, 0],
           [1, 3, 3, 1]], dtype=uint64)

    """
    
    if kmax is None:
        kmax = nmax
        
    if nmax < 0:
        raise ValueError('nmax must be >= 0')
    if kmax < 0:
        raise ValueError('kmax must be >= 0')
        
    tab = np.zeros((nmax+1,kmax+1), dtype = np.uint64)
    
    for n in range(nmax+1):
        for k in range(kmax+1):
            
            if k > n:
                tab[n,k] = 0
            elif n < 1:
                tab[n,k] = 1
            elif k < 1:
                tab[n,k] = 1 
            elif k == n:
                tab[n,k] = 1
            else:
                tab[n,k] = tab[n-1,k-1] + tab[n-1,k]
                
    return tab

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
    return ncktab(n,k)[n,k]

def nckmulti(a,b,nck):
    """
    Calculate multi-index binomial coefficient (`a`, `b`)

    Parameters
    ----------
    a,b : ndarray
        1D multi-indices
    nck : ndarray
        Binomial coefficient table as returned by
        ``ncktab(nmax,kmax)``, with nmax >= max(`a`)
        and kmax >= max(`b`)

    Returns
    -------
    np.float64
        Multi-index binomial coefficient.
        
    See Also
    --------
    ncktab : Binomial coefficient table

    """
    
    ni = a.size # lenth of multi-index
    
    value = np.float64(1.0) # double float
    
    for i in range(ni):
        value *= nck[a[i],b[i]]
        
    return value

def idxtab(k,ni):
    
    """
    Multi-index table for derivatives up to order 
    `k` of `ni` variables.
    
    The number of multi-indices (i.e. the number of 
    unique derivatives) is `nd` = nck(`k` + `ni`, `ni`)

    Parameters
    ----------
    k : int
        Maximum derivative order.
    ni : int
        Number of variables.

    Returns
    -------
    ndarray
        Multi-index table with shape (`nd`, `ni`) and
        data-type ``np.uint32``.
        
    Examples
    --------
    >>> idxtab(2,3)
    array([[0, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [2, 0, 0],
           [1, 1, 0],
           [1, 0, 1],
           [0, 2, 0],
           [0, 1, 1],
           [0, 0, 2]], dtype=uint32)

    """
    
    #nd = nck(k + ni, ni) # The number of derivatives
    nd = ncktab(k+ni,min(k,ni))[k+ni, min(k,ni)] # --this is faster than nck
    idx = np.zeros((nd,ni),dtype=np.uint32)
    
    # The first row is already all zeros -- no derivative
    # Compute next row by incrementing.
    
    for i in range(1,nd):
        
        # idxtab[i,:] equals the increment of
        # idxtab[i-1,:]
        
        # We will treat the second row in the table 
        # as a special case. It is just [1, 0, 0, ...]
        if i == 1:
            idx[i,0] = 1
            continue 
        
        # The current degree is >= 1
        
        # Look for the right-most non-zero degree
        for j in range(ni-1,-1,-1):
            
            if idx[i-1,j] > 0:
                # If it is not at the last position, shift
                # one degree right one position.
                if j < (ni-1):
                    idx[i,:] = idx[i-1,:]
                    idx[i,j] -= 1
                    idx[i,j+1] += 1
                    break
                else: # it is at the last position, j = nd-1
                    # Look for the *next* non-zero position
                    for p in range(ni-2,-1,-1):
                        if idx[i-1,p] > 0:
                            # We found another non-zero entry.
                            # Shift one from this entry to 
                            # the right and add everything
                            # from the last position
                            idx[i,:] = idx[i-1,:]
                            idx[i,ni-1] = 0
                            idx[i,p] -= 1 
                            idx[i,p+1] += 1 + idx[i-1,ni-1]
                            break
                    else: # We did not find another non-zero
                        # this means we have an index like
                        # 0,0,0,...,b
                        # so the next index is
                        # b+1,0,0,...,0
                            idx[i,:] = 0
                            idx[i,0] = 1 + idx[i-1,ni-1]
                    break
        
        
    return idx

def idxposk(a,nck):
    """
    Calculate the relative lexical position of multi-index `a`
    within the block of its degree ``k`` = sum(`a`).

    Parameters
    ----------
    a : ndarray
        1D multi-index
    nck : ndarray
        A binomial coefficent table as returned by
        ``ncktab(nmax,kmax)`` with ``nmax >= ni - 3 + k - a[0]``
        and ``kmax >= min(ni - 2, k - a[0] - 1)``. Simpler
        requirements that satisfy these are ``nmax >= ni + k - 1`` and
        ``kmax >= min(ni, k - 1)``.

    Returns
    -------
    posk : np.uint64
        Relative position of multi-index in its
        block of degree ``k`` = sum(`a`).
    
    Examples
    --------
    >>> idxposk(np.array([0,0,0]),ncktab(3,3))
    0
    
    >>> idxposk(np.array([2,0,1,3]),ncktab(5,2))
    33
        
    See Also 
    --------
    ncktab : Binomial coefficient table

    """
    
    ni = a.size
    
    if ni == 0 or ni == 1:
        posk = np.uint64(0)
    else:
        k = np.sum(a)
        
        if k == 0:
            posk = np.uint64(0)
        else:
            a0 = a[0]
            
            # Because we assume that we will only index `nck` as
            # nck[n,min(k,n-k)], we cannot use the following simple sum
            #
            #  posk = np.sum( nck[(ni-2):(ni-2+(k-a0)), ni-2] ) + idxposk(a[1:], nck)
            #
            # Instead, we can explicitly loop over the sum
            #
            posk = np.uint64(0)
            for p in range(k-a0):
                posk += nck[p + ni - 2, min(p, ni-2)]
            posk += idxposk(a[1:], nck)
    
    return posk
    
def idxpos(a,nck):
    """
    Calculate the absolute lexical position of multi-index `a`.

    Parameters
    ----------
    a : ndarray
        1D multi-index
    nck : ndarray
        A binomial coefficient table as returned by
        ``ncktab(nmax,kmax)``, with ``nmax >= ni + k - 1``
        and ``kmax >= min(ni, k - 1)``, where ``ni = a.size``
        and ``k = sum(a)``.

    Returns
    -------
    pos : np.uint64
        The absolute lexical position of multi-index `a` 
        
    Examples
    --------
    >>> idxpos(np.array([0,0,0]),ncktab(3,3))
    0
    
    >>> idxpos(np.array([2,0,1,3]),ncktab(9,4))
    159
        
    See Also 
    -------- 
    ncktab: Binomial coefficient table

    """
        
    k = np.sum(a) # Degree of multi-index a
    ni = a.size   # Number of variables
    
    if k == 0:
        return np.uint64(0)
    
    else:
    
        offset = nck[ni + k - 1, min(ni,k-1)] # The number of multi-indices with degree
                                              # less than k
        posk = idxposk(a,nck)    # The position of this multi-index within
                                 # the block of multi-indices of the same degree k
        return offset + posk

def mvleibniz(X, Y, k, ni, nck, idx, out=None, Xzlevel = None, Yzlevel = None):
    """
    Multivariate Leibniz formula for derivative arrays.

    Parameters
    ----------
    X,Y : ndarray
        Derivative array factors (e.g. :attr:`adarray.d`). These
        must have matching shapes.
    k : int
        Maximum derivative order of `X` and `Y`
    ni : int
        Number of independent variables for `X` and `Y`
    nck : ndarray
        Binomial coefficient table, as returned
        by ``ncktab(nmax,kmax)`` with ``nmax`` >= `k` + `ni` - 1
        and ``kmax`` >= min(`ni`, `k` - 1) 
    idx : ndarray
        Multi-index table, as returned by ``idxtab(k, ni)``
    out : ndarray, optional
        Output location. If None, a new ndarray is created with the 
        result type of X and Y's data-types
    Xzlevel, Yzlevel : int, optional
        Zero-level for input. If None, this is assumed to be `k`.
        The default is None.

    Returns
    -------
    out : ndarray
        The derivative array for `X` * `Y`
        
    See Also
    --------
    adarray
    ncktab : Binomial coefficient table
    idxtab : Multi-index table
    
    Notes
    -----
    :func:`mvleibniz` is low-level function applied directly to
    derivative arrays (usually :attr:`adarray.d`).
    Typically, multiplication should be applied at high-level
    with the * operator directly with :class:`adarray` objects.
    
    """
    
    # Initialize result Z to zero
    res_type = np.result_type(X.dtype, Y.dtype)
    if out is None:
        out = np.zeros(X.shape, dtype = res_type)
    else:
        if out.dtype != res_type:
            raise TypeError("out data-type is incompatible with X * Y")
        out.fill(0)
        
    if Xzlevel is None:
        Xzlevel = k 
    if Yzlevel is None: 
        Yzlevel = k 
    
    Z = out # Reference only
   
    nd,_ = idx.shape
    
    for iX in range(nd):
        idxX = idx[iX,:]
        kX = np.sum(idxX)
        if kX > Xzlevel:
            break # Skip remaining X derivatives. They are zero.
        
        for iY in range(nd):
            idxY = idx[iY,:]
            kY = np.sum(idxY)
            if kY > Yzlevel:
                break # Skip remaining Y derivatives. They are zero.
            
            kZ = kX + kY
            if kZ > k: # Total degree of this term is too high
                break # Skip remaining Y derivatives
            
            # the product of the X derivative and Y derivatives
            # contributes to the Z derivative with index:
            idxZ = idxX + idxY
            # Convert this multi-index to a 1D index
            iZ = idxpos(idxZ, nck)
                
            # adarrays are normalized to include (inverse) factorial
            # factors for each derivative. This convention removes
            # the binomial coefficient in the generalized Leibniz
            # formula. I.e. we do not need to calculate 
            #    nckmulti(idxZ, idxX, nck)
            #
            
            # Add this term to Z
            # Use of '+=' with ndarrays will not create a new copy
            Z[iZ] += X[iX] * Y[iY]
            
    return Z

def mvchain(df,X,k,ni,nck,idx, out = None, Xzlevel = None):
    """
    Multivariate chain rule Z = f(X) via Taylor series.

    Parameters
    ----------
    df : ndarray
        An array containing the derivatives of 
        single-argument function f through order `k`.
        The shape of `df` is ``(k+1,) + X.shape``
    X : ndarray
        Derivative array in ``adarray.d`` format
    k : int
        Maximum derivative order
    ni : int
        Number of independent variables
    nck : ndarray 
        Binomial coefficient table for `X` satisfying requirements
        for :func:`mvleibniz`.
    idx : ndarray
        Multi-index table for `X` satisfying requirements for
        :func:`mvleibniz`.
    out : ndarray, optional
        Output location. If None, a new ndarray is created with the 
        same data-type as X.        
    Xzlevel : int, optional
        The zlevel of the X derivative array. If None, this is
        assumed to be `k`.

    Returns
    -------
    out : ndarray
        The derivative array for f(`X`)
        
        
    See Also
    --------
    ncktab : Binomial coefficient table
    idxtab : Multi-index table
    mvleibniz : Generalized Leibniz product rule
    adchain : Chain rule for adarray objects
    
    Notes
    -----
    :func:`mvchain` is a low-level function that acts directly
    on derivative arrays (usually :attr:`adarray.d`). In most cases, the
    high-level function :func:`adchain` should be used directly with
    :class:`adarray` objects.
    """
    
    X0 = X[:1].copy() # Value of X 
    X[:1].fill(0) # X now equals "X-X0"
    if Xzlevel is None:
        Xzlevel = k
    
    # Initialize result to zero (and create if necessary)
    res_type = np.result_type(df, X)
    if out is None:
        out = np.ndarray(X.shape, dtype = res_type)
    
    if out.dtype != res_type:
        raise TypeError("out data-type is incompatible with f(X)")
        
    # Initalize result to zero
    out.fill(0)
        
    Z = out # Reference only
    
    for i in range(k+1):
        
        if i == 0:
            # Initialize Xi = X**i = X**0 to 1 (constant)
            Xi = np.zeros(X.shape,dtype = X.dtype)
            #Xi[0] = 1.0
            Xi[:1].fill(1.0)
            Xizlevel = 0 # the zero-level is 0 (constant)
            fact = 1.0 # Initialize factorial to 1
        else:
            Xi = mvleibniz(Xi,X,k,ni,nck,idx) # Xi <-- Xi * (X-X0)
            # The call to mvleibniz needs to create a new output buffer
            # because we cannot overwrite Xi while mvleibniz is operating
            # (this probably could be done more efficiently -- revisit)
            
            # Determine the zlevel of Xi
            if Xizlevel < 0 or Xzlevel < 0:
                Xizlevel = -1 
            else:
                Xizlevel = min(Xizlevel + Xzlevel, k)
            
            fact = fact * i 
            
        if (df[i] != 0 ).any() and Xizlevel >= 0:
            # If both df is non-zero and
            # Xi is non-zero
            Z += (df[i] / fact) * Xi
    
    #X[0] = X0  # Restore the value of X
    np.copyto(X[:1], X0)
    
    return Z


def const(value,k,ni,nck = None, idx = None):
    """
    Create an :class:`adarray` object for a constant scalar or array.

    Parameters
    ----------
    value : array_like
        The constant value. If this is not an ndarray already, it
        will try to be converted.
    k : int
        The maximum derivative degree.
    ni : int
        The number of independent variables
    nck, idx : ndarray, optional
        See :class:`adarray` constructor.

    Returns
    -------
    adarray
        A constant adarray object.
    
    Examples
    --------
    >>> const(1., 2, 2).d
    array([1., 0., 0., 0., 0., 0.])
    
    >>> const(42j, 1, 3).d
    array([0.+42.j, 0. +0.j, 0. +0.j, 0. +0.j])
    
    >>> const([2.0, 3.0], 2, 2).d
    array([[2., 3.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.]])
    """
    
    if type(value) != np.ndarray:
        value = np.array(value)
    
    base_shape = value.shape
    nck = ncktab(k + ni, min(k,ni))
    nd = nck[k + ni, min(k,ni)]
    
    d = np.zeros((nd,)+base_shape, dtype = value.dtype)
    
    # Copy value to d[:1]
    np.copyto(d[:1], value)
    
    # Determine the zlevel
    if (value == 0).all():
        zlevel = -1 # Identically zero
    else:
        zlevel = 0 # A non-zero constant
    
    c = array(d, k, ni, copyd = False, zlevel = zlevel)
        
    return c

def sym(value,i,k,ni,nck = None, idx = None):
    """
    Create an :class:`adarray` for a symbol (i.e. one
    the independent variables with respect to which derivatives
    are being taken).

    Parameters
    ----------
    value : array_like
        The value of the variable.
    i : int
        The variable index, `i` = 0, ... , `ni` - 1
    k, ni: int
        See :class:`adarray` constructor.
    nck,idx : ndarray, optional
        See :class:`adarray` constructor.

    Returns
    -------
    adarray 
        An :class:`adarray` object for variable `i` and value `value`.
        
    Examples
    --------
    >>> sym(2.0, 0, 2, 2).d
    array([2., 1., 0., 0., 0., 0.])
    
    >>> sym(3.0, 1, 2, 2).d
    array([3., 0., 1., 0., 0., 0.])
    
    >>> sym([3.0, 4.0j], 1, 2, 2).d
    array([[3.+0.j, 0.+4.j],
           [0.+0.j, 0.+0.j],
           [1.+0.j, 1.+0.j],
           [0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j]])

    """
    
    if k < 0:
        raise ValueError("k be >= 0")
    if ni < 1:
        raise ValueError("ni must be >= 1 to construct symbol")
    
    if i > ni+1 or i < 0 :
        raise ValueError('Symbol index i must be 0, ..., ni-1')
    
    x = const(value, k, ni, nck, idx)
    # const returns x with zlevel = -1 or 0 depending on value
    
    if k > 0 :
        x.d[(i+1):(i+2)].fill(1)
        x.zlevel = 1 
    # If k == 0, then the zlevel from const can be kept
    
    return x

def empty_like(x, dtype = None):
    """
    Create an *uninitialized* adarray with the same properties as `x`,
    including base array data-type. The zlevel will be maximum.

    Parameters
    ----------
    x : adarray
        Prototype object
        
    dtype : dtype
        Base data-type. If None, then `x.d.dtype` is used

    Returns
    -------
    adarray
        A new adarray with the same properties as `x`
        
    >>> empty_like(const([3.3, 2.1], 2, 2)).d.shape
    (6, 2)

    """
    if dtype is None:
        dtype = x.d.dtype
        
    return adarray(x.d.shape[1:], x.k, x.ni, x.nck, x.idx, dtype)

def add(x, y, out = None):
    """
    Add x + y

    Parameters
    ----------
    x,y : adarray
        Input argument
    out : adarray, optional
        Output location. If None, this 
        will be created. The default is None.

    Returns
    -------
    adarray
        Result.

    """
    
    res_type = np.result_type(x.d, y.d)
    if out is None:
        out = empty_like(x, dtype = res_type)
    
    if res_type != out.d.dtype:
        raise TypeError("output data-type incompatible with x + y")
    
    np.add(x.d, y.d, out = out.d)
    out.zlevel = max(x.zlevel, y.zlevel)
    
    return out

def subtract(x, y, out = None):
    """
    Subtract x - y

    Parameters
    ----------
    x,y : adarray
        Input argument
    out : adarray, optional
        Output location. If None, this 
        will be created. The default is None.

    Returns
    -------
    adarray
        Result.

    """
    
    res_type = np.result_type(x.d, y.d)
    if out is None:
        out = empty_like(x, dtype = res_type)
    
    if res_type != out.d.dtype:
        raise TypeError("output data-type incompatible with x - y")
    
    np.subtract(x.d, y.d, out = out.d)
    out.zlevel = max(x.zlevel, y.zlevel)
    
    return out

def mul(x, y, out = None):
    """
    Multiply x * y
    
    Parameters
    ----------
    x,y : adarray
        Input argument
    out : adarray, optional
        Output buffer. If None, this will be created.
        The default is None.

    Returns
    -------
    adarray
        Result.

    """
    
    res_type = np.result_type(x.d, y.d)
    if out is None:
        out = empty_like(x, dtype = res_type)
        
    if res_type != out.d.dtype:
        raise TypeError("output data-type incompatible with x * y")

    mvleibniz(x.d,y.d,x.k,x.ni,x.nck,x.idx, out = out.d)
    
    if x.zlevel < 0 or y.zlevel < 0 :
        # Either of the factors is identically zero
        # So is the result.
        out.zlevel = -1
    else:
        # Both zlevels are >= 0
        out.zlevel = min(x.zlevel + y.zlevel, x.k)
    
    return out

def div(x, y, out = None):
    """
    Divide x / y

    Parameters
    ----------
    x,y : adarray
        Input argument
    out : adarray, optional
        Output buffer. If None, this will be created.
        The default is None.

    Returns
    -------
    adarray
        Result.

    """
    
    res_type = np.result_type(x.d, y.d)
    if out is None:
        out = empty_like(x, dtype = res_type)
        
    if res_type != out.d.dtype:
        raise TypeError("output data-type incompatible with x / y")

    # Calculate 1 / y
    iy = powf(y, -1.0)
    # Multiply x * (1/y)
    return mul(x,iy, out = out)

def adchain(df,x, out=None):
    
    """
    Calculate f(x) via chain rule.

    Parameters
    ----------
    df : ndarray
        Single-variable derivatives of f(x) w.r.t to its argument
        up to order :attr:`x.k`
        
    x : adarray
        Argument of f(`x`)
        
    out : adarray, optional
        Output location of result. `out` must have the same
        properties as x. If None, a new adarray is allocated and
        returned.

    Returns
    -------
    adarray
        The result f(`x`).

    """
    
    if out is None:
        out = empty_like(x)
    
    mvchain(df, x.d, x.k, x.ni, x.nck, x.idx, out = out.d)
    
    # Determine the zlevel of the result
    # We assume, safely, that df has a maximum zlevel
    if x.zlevel <= 0:
        # x is identically zero or constant
        # The result is also constant
        out.zlevel = 0
    else:
        # x has a zlevel >= 1
        out.zlevel = x.k
    
    return out

def sin(x, out=None):
    """
    Sine for :class:`adarray` objects.

    Parameters
    ----------
    x : adarray
        Input in radians.
        
    out : adarray, optional
        Output location of result. `out` must have the same
        properties as x. If None, a new adarray is allocated and
        returned.

    Returns
    -------
    adarray
        Result.

    Examples
    --------
    >>> x = sym(1.0, 0, 3, 1)
    >>> sin(x).d
    array([ 0.84147098,  0.54030231, -0.42073549, -0.09005038])
    """
    
    xval = x.d[0] # Value array of x
    k = x.k
    
    df = np.ndarray( (k+1,)+x.d.shape[1:], dtype = xval.dtype)    
    
    for i in range(k+1):
        
        if i == 0:
            df[i] = np.sin(xval)
        elif i == 1:
            df[i] = np.cos(xval)
        else:
            df[i] = -df[i-2]
                    
    return adchain(df, x, out = out)
            
def cos(x, out=None):
    """
    Cosine for :class:`adarray` objects.

    Parameters
    ----------
    x : adarray
        Input in radians.
    
    out : adarray, optional
        Output location of result. `out` must have the same
        properties as x. If None, a new adarray is allocated and
        returned.

    Returns
    -------
    adarray
        Result.

    Examples
    --------
    >>> x = sym(2.0, 0, 3, 1)
    >>> cos(x).d
    array([-0.41614684, -0.90929743,  0.20807342,  0.15154957])
    
    """
    
    xval = x.d[0] # Value array of x
    k = x.k
    
    df = np.ndarray( (k+1,)+x.d.shape[1:], dtype = xval.dtype)
    
    for i in range(k+1):
        
        if i == 0:
            df[i] = np.cos(xval)
        elif i == 1:
            df[i] = -np.sin(xval)
        else:
            df[i] = -df[i-2]
                    
    return adchain(df, x, out = out)
      
def exp(x, out = None):
    """
    Exponential for :class:`adarray` objects.

    Parameters
    ----------
    x : adarray
        Input argument.

    out : adarray, optional
        Output location of result. `out` must have the same
        properties as x. If None, a new adarray is allocated and
        returned.

    Returns
    -------
    adarray
        Result.
        
    Examples
    --------
    >>> x = sym(1.5, 0, 3, 1)
    >>> exp(x).d
    array([4.48168907, 4.48168907, 2.24084454, 0.74694818])

    """
    
    xval = x.d[:1] # Value array of x
    k = x.k
    
    df = np.ndarray( (k+1,)+x.d.shape[1:], dtype = xval.dtype)
    
    np.exp(xval, out = df[:1])
    for i in range(1,k+1):
        np.copyto(df[i:(i+1)], df[:1])
            
    return adchain(df, x, out = out)

def log(x, out = None):
    """
    Natural logarithm for :class:`adarray` objects.

    Parameters
    ----------
    x : adarray
        Input argument.

    out : adarray, optional
        Output location of result. `out` must have the same
        properties as x. If None, a new adarray is allocated and
        returned.

    Returns
    -------
    adarray
        Result.

    """
    
    xval = x.d[:1] # Value array of x
    k = x.k
    
    df = np.ndarray( (k+1,)+x.d.shape[1:], dtype = xval.dtype)
    
    # Value of f: ln(x)
    np.log(xval, out = df[:1])
    
    # First derivative of f:
    # 1/x
    if k > 0:
        np.copyto(df[1:2], 1.0/xval)
        
    for i in range(2,k+1):
        # i^th derivative of f:
        # (1/x)**i = 1/x * (1/x)**(i-1)
        #
        np.multiply(df[1:2], df[(i-1):i], out = df[i:(i+1)])
            
    return adchain(df, x, out = out)

def powf(x, p, out = None):
    """
    x**p for general p

    Parameters
    ----------
    x : adarray
        Input argument
    p : float or complex
        Power. 
    out : adarray, optional
        Output location of result.

    Returns
    -------
    adarray
        Result.
        
    Notes
    -----
    powf uses the NumPy :func:`~numpy.float_power` function to 
    compute the value array. It inherits the branch-cut convention
    of this function.
    
    Examples
    --------
    >>> x = sym(1.5, 0, 3, 1)
    >>> powf(x, -2.5).d
    array([ 0.36288737, -0.60481228,  0.70561433, -0.70561433])
    
    """
    
    if np.result_type(x.d.dtype, p) != np.result_type(x.d.dtype):
        raise TypeError("Invalid type combination")
    
    xval = x.d[0] # Value array of x
    k = x.k 
    
    df = np.ndarray( (k+1,) + x.d.shape[1:], dtype = xval.dtype)
    
    df[0] = np.float_power(xval, p)
    coeff = 1.0*p
    for i in range(1, k+1):
        df[i] = coeff * (df[i-1] / xval)
        coeff *= (p-i)
    
    return adchain(df, x, out = out)

def sqrt(x, out = None):
    """
    sqrt(x)

    Parameters
    ----------
    x : adarray 
        Input argument
    out : adarray, optional
        Output buffer. If None, this will be created.
        The default is None.

    Returns
    -------
    adarray
        Result.
        
    Notes
    -----
    The adarray sqrt function uses the NumPy :func:`~numpy.sqrt` function
    as its underlying routine. The branch-cut convention there is inherited.

    """
    
    xval = x.d[0] # Value array of x
    k = x.k 
    
    df = np.ndarray( (k+1,) + x.d.shape[1:], dtype = xval.dtype)
    
    df[0] = np.sqrt(xval) # Uses numpy branch cut
    coeff = +0.5
    for i in range(1, k+1):
        df[i] = coeff * (df[i-1] / xval)
        coeff *= (+0.5 - i)
    
    return adchain(df, x, out = out)


    
def reduceOrder(F, i, k, ni, idx, out = None):
    """
    Reduce the derivative array for F with respect to 
    variable i. The returned derivative array is
    that for the function :math:`\\partial_iF`.

    Parameters
    ----------
    F : ndarray
        The derivative array up to degree `k`
        in `ni` variables.
    i : int
        The variable index (0, ..., `ni` - 1) to
        reduce.
    k : int
        The initial derivative order of F.
    ni : int
        The number of independent variables.
    idx : ndarray
        The return value of idxtab with suitable parameters.
    out : ndarray, optional
        Output buffer. If None, this will be
        created. The default is None.

    Returns
    -------
    out : ndarray

    """

    nd,_ = idx.shape # The number of derivatives
    
    if k <= 0:
        raise ValueError("Cannot reduce a derivative array with k = 0") 
        
    if i < 0 or i >= ni : 
        raise ValueError(f"Cannot reduce w.r.t variable index i = {i:d} with only ni = {ni:d} variables")
        
    # k and ni are now both > 0
    #
    # Calculate the number of derivatives for
    # order k - 1 in ni variables. We already know
    # that nd is the number of deriv for order k
    # in ni variables, so we can use the simple result:
    nd_reduced = (nd * k) // (k + ni)  # This should be an integer result always!
    
    if out is None:
        out = np.ndarray( (nd_reduced,) + F.shape[1:], dtype = F.dtype)
    G = out # reference only
    
    # The index table `idx` for F is already provided,
    # so we will loop through this instead of building
    # a new index table for the reduced derivative array
    # The derivatives we are interested in appear
    # in the same order in the F (unreduced) and G (reduced)
    # derivative arrays.
    # This is to say that if one has two multi-indices of F
    # a = [a0 a1 a2 ... ai ...] and b = [b0 b1 b2 ... bi ...]
    # a appears before b
    # then the corresponding reduced indices
    # a' = [a0 a1 a2 ... ai-1 ... ] and b' = [b0 b1 b2 ... bi-1 ...]
    # appear in the same order: a' before b'
    #
    #
    iG = 0 # running position in the reduced derivative array 
    for iF in range(nd):
        
        idxF = idx[iF,:]  # The multi-index of the original function's derivatives
        
        if idxF[i] < 1:
            # This derivative is not part of the reduced array
            continue 
        # else, idxF[i] >= 1. It contributes to the 
        # derivative element of G corresponding to idxF[i]--
        #
        # Because we store the derivatives with a factor
        # equal to the inverse of the multi-index factorial,
        # we need to correct this in the reduced array
        #
        # The correct conversion factor is
        #  idxF! / idxG! = idxF[i]! / idxG[i]!
        #                = idxF[i]! / (idxF[i] - 1)!
        #                = idxF[i]
        #
        # Now we do this:
        # G[iG] = F[iF] * idxF[i] 
        #
        # Depending on the shape of G, we will split this up
        if len(G.shape) == 1:
            G[iG] = F[iF] * idxF[i] # Scalar multiplication
        else:
            np.multiply(F[iF], idxF[i], out = G[iG]) # use np.multiply to save on temp memory
        
        iG += 1
        
    # Check that nothing went wrong in the book keeping
    assert iG == nd_reduced, "Derivative array reduction was not successful!"
     
    
    return G

def n2N(n):
    """
    Calculate the square matrix rank N
    for a packed storage size n

    Parameters
    ----------
    n : int
        The packed length.

    Returns
    -------
    N : np.uint64
        The matrix rank.

    """
    
    N = np.uint64((np.sqrt(8*n+1)-1)/2)
    
    return N

def chol_sp(H, out = None):
    """
    Cholesky decomposition of a symmetric matrix in packed format.
    If real symmetric, then H should be positive definite.
    If complex symmetric (*not* Hermitian), then H should have
    non-zero pivots.

    Parameters
    ----------
    H : ndarray of adarray
        H is stored in 1D packed format (see :mod:`nitrogen.linalg.packed`)
    out : ndarray of adarray
        Output buffer. If None, this will be created. 
        If out = H, then in-place decomposition is performed

    Returns
    -------
    out : ndarray of adarray
        The lower triangle Cholesky decomposition L in packed storage.
        H = L @ L.T
    
    """
    
    if H.ndim != 1:
        raise ValueError('H must be 1-dimensional')
    
    if out is None:
        out = np.ndarray(H.size, dtype = adarray)
        for i in range(H.size):
            out[i] = empty_like(H[i])
    
    # Copy H to out for in-place routine
    if out is not H:
        for i in range(H.size):
            H[i].copy(out = out[i])
        
    _chol_sp_unblocked(out)
    
    return out
    
def _chol_sp_unblocked(H):
    """
    An unblocked, in-place implementation of Cholesky
    decomposition for symmetric matrices H.

    Parameters
    ----------
    H : ndarray of adarray
        H is a symmetric matrix in 1D packed storage.
        (Lower triangle row-packed)

    Returns
    -------
    ndarray of adarray 
        The in-place result.
        
    Notes
    -----
    This routine uses a standard Cholesky algorithm
    for *real* symmetric matrices.

    """


    # H is a 1d ndarray of adarray objects
    # in packed format
    n = H.size
    N = n2N(n)

    L = np.ndarray((N,N), dtype = adarray)
    # Copy references to adarrays to the lower 
    # triangle of a full "reference" matrix
    # References above the diagonal are undefined.
    k = 0
    for i in range(N):
        for j in range(i+1):
            L[i,j] = H[k]
            k += 1
    
    # Compute the Cholesky decomposition
    # with a simple unblocked algorithm
    tol = 1e-10
    pivmax = np.abs( np.sqrt(L[0,0].d[:1]) ) # for pivot threshold checking
    
    for j in range(N):
        r = L[j,:j]         # The j^th row, left of the diagonal
    
        # L[j,j] <-- sqrt(d - r @ r.T)
        sqrt( (L[j,j] - r @ r.T).copy(), out = L[j,j])
        
        # Check that the pivot is sufficiently large
        if (np.abs(L[j,j].d[:1]) / pivmax < tol).any() :
            warnings.warn("Small diagonal (less than rel. tol. = {:.4E} encountered in Cholesky decomposition".format(tol))
        
        # Store the new maximum pivot
        pivmax = np.maximum(pivmax, np.abs(L[j,j].d[:1]))
        
        # Calculate the column below the diagonal element j
        #B = L[j+1:,:j]      # The block between r and c
        #c = L[j+1:,j]       # The j^th column, below the diagonal
        for i in range(j+1,N):
            Bi = L[i,:j]    # An ndarray
            ci = L[i, j]    # An adarray 
            #L[j+1:,j] = (c - B @ r.T) / L[j,j]
            div( (ci - Bi @ r.T).copy(), L[j,j], out = L[i,j] )
        

    return H
    
def inv_tp(L, out = None):
    """
    Invert a triangular matrix in lower row-packed (or upper column-packed)
    storage.

    Parameters
    ----------
    L : ndarray of adarray
        L is stored in 1D packed format (see :mod:`nitrogen.linalg.packed`)
    out : ndarray of adarray
        Output buffer. If None, this will be created. 
        If out = L, then in-place inversion is performed

    Returns
    -------
    out : ndarray of adarray
        The inverse of the triangular matrix in packed storage.
    
    """
    
    if L.ndim != 1:
        raise ValueError('L must be 1-dimensional')
    
    if out is None:
        out = np.ndarray(L.size, dtype = adarray)
        for i in range(L.size):
            out[i] = empty_like(L[i])
    
    # Copy L to out for in-place routine
    if out is not L:
        for i in range(L.size):
            L[i].copy(out = out[i])
            
    # Now perform in-place on `out`
    _inv_tp_unblocked(out)
    
    return out
    
def _inv_tp_unblocked(L):
    """
    Invert a lower triangular matrix in row-packed storage.

    Parameters
    ----------
    L : ndarray of adarray
        L is a triangular matrix in 1D packed storage.
        (Row-packed for lower, column-packed for upper)

    Returns
    -------
    ndarray of adarray 
        The in-place result.

    """

    n = L.size
    N = n2N(n)
    one = np.uint64(1)
    
    X = np.ndarray((N,N), dtype = adarray)
    # Copy references to adarrays to the lower 
    # triangle of a full "reference" matrix
    # Elements above the diagonal are not defined!
    k = 0
    for i in range(N):
        for j in range(i+1):
            X[i,j] = L[k]
            k += 1
    
    # Compute the triangular inverse
    # with a simple in-place element by element algorithm
    abstol = 1e-15 
    # In-place lower triangle inversion
    for j in range(N - one, -1,-1):
        
        # Compute j^th diagonal element
        if (np.abs(X[j,j].d[:1]) < abstol).any():
            warnings.warn(f"Small diagonal (less than abs. tol. = {abstol:.4E})" \
                          "encounted in triangle inversion")
        
        #X[j,j] = 1.0/X[j,j]
        powf(X[j,j].copy(), -1.0, out = X[j,j])
        
        # Compute j^th column, below diagonal
        for i in range(N - one, j, -1):
            mul( -X[j,j], X[i, j+1:i+1] @ X[j+1:i+1, j], out = X[i,j])
      
    return L

def llt_tp(L, out = None):
    """
    L @ L.T of a lower triangular matrix.

    Parameters
    ----------
    L : ndarray of adarray
        Lower triangular matrix in packed storage.
    out : ndarray of adarray
        Output buffer. If None, this will be created. 
        If out = L, then in-place multiplication is performed

    Returns
    -------
    out : ndarray of adarray
        The result in packed storage.
    
    """
    
    if L.ndim != 1:
        raise ValueError('L must be 1-dimensional')
    
    if out is None:
        out = np.ndarray(L.size, dtype = adarray)
        for i in range(L.size):
            out[i] = empty_like(L[i])
    
    # Copy L to out for in-place routine
    if out is not L:
        for i in range(L.size):
            L[i].copy(out = out[i])
        
    # Perform in-place L @ L.T
    _llt_tp_unblocked(out)
    
    return out

def _llt_tp_unblocked(L):
    """
    An unblocked, in-place routine for multiplying
    L @ L.T where L is a lower triangular matrix
    in packed row-order storage.
    
    This is equivalent to U.T @ U where U is in
    upper triangular packed column-order storage.
    
    The resulting symmetric matrix is returned in
    packed storage.

    Parameters
    ----------
    L : ndarray of adarray
        A lower triangular matrix in 1D packed 
        row-order storage.

    Returns
    -------
    ndarray of adarray 
        The in-place result.
        

    """

    # Calculate matrix dimensions
    n = L.size
    N = n2N(n)
    one = np.uint64(1)

    A = np.ndarray((N,N), dtype = adarray)
    # Copy references to adarrays to the lower 
    # triangle of a full "reference" matrix
    # References above the diagonal are undefined.
    k = 0
    for i in range(N):
        for j in range(i+1):
            A[i,j] = L[k]
            k += 1
    
    # This is similar to a "reverse Cholesky decomposition"
    # so we will work in the opposite direction as that
    
    for j in range(N-one, -1, -1):
        
        r = A[j,:j]         # The j^th row, left of the diagonal
        
        for i in range(N-one, j, -1):
            Bi = A[i,:j]    # An ndarray
            ci = A[i, j]    # An adarray 
            
            # ci <-- Ljj * ci + Bi @ r.T
            ( A[j,j] * ci + Bi @ r.T ).copy(out = A[i,j])
        
        (A[j,j]*A[j,j] + r @ r.T).copy(out = A[j,j])        

    return L

def ltl_tp(L, out = None):
    """
    L.T @ L with a lower triangular matrix.

    Parameters
    ----------
    L : ndarray of adarray
        Lower triangular matrix in packed storage.
    out : ndarray of adarray
        Output buffer. If None, this will be created. 
        If out = L, then in-place multiplication is performed

    Returns
    -------
    out : ndarray of adarray
        The symmetric result in packed storage.
    
    """
    
    if L.ndim != 1:
        raise ValueError('L must be 1-dimensional')
    
    if out is None:
        out = np.ndarray(L.size, dtype = adarray)
        for i in range(L.size):
            out[i] = empty_like(L[i])
    
    # Copy L to out for in-place routine
    if out is not L:
        for i in range(L.size):
            L[i].copy(out = out[i])
        
    # Perform in-place L @ L.T
    _ltl_tp_unblocked(out)
    
    return out

def _ltl_tp_unblocked(L):
    """
    An unblocked, in-place routine for multiplying
    L.T @ L where L is a lower triangular matrix
    in packed row-order storage.
    
    This is equivalent to U @ U.T where U is in
    upper triangular packed column-order storage.
    
    The resulting symmetric matrix is returned in
    packed storage.

    Parameters
    ----------
    L : ndarray of adarray
        A lower triangular matrix in 1D packed 
        row-order storage.

    Returns
    -------
    ndarray of adarray 
        The in-place result.
        

    """

    # Calculate matrix dimensions
    n = L.size
    N = n2N(n)

    A = np.ndarray((N,N), dtype = adarray)
    # Copy references to adarrays to the lower 
    # triangle of a full "reference" matrix
    # References above the diagonal are undefined.
    k = 0
    for i in range(N):
        for j in range(i+1):
            A[i,j] = L[k]
            k += 1
    
    # This is the "converse" of the llt routine
    # for L @ L.T
    for i in range(N):
        for j in range(i+1):
            # Compute A[i,j]
            # This is the dot product of the
            # i^th row of L.T and the
            # j^th column of L
            # 
            # The i^th row of L.T is zero until
            # its i^th element
            # 
            # The j^th column of L is zero until
            # its j^th element
            #
            # So the dot product need only begin
            # at the max(i,j)^th element
            # 
            # By the loop ranges, j is always <= i
            # so max(i,j) = i, and we can begin
            # the dot product with the i^th element
            
            # The first factor is the
            # i^th row of L.T beginning at its i^th element
            # This is the transpose of the i^th column of 
            # L beginning at its i^th element, which is in
            # the lower triangle, so A's reference is OK
            F1 = (A[i:,i]).T
            # The second factor is the j^th column
            # of L beginning at its i^th element, which is
            # also in the lower triangle, so OK
            F2 = A[i:,j]
            
            (F1 @ F2).copy(out = A[i,j])
        
               

    return L