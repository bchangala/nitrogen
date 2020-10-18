"""
nitrogen.autodiff.forward
-------------------------

This module implements a simple forward accumulation
model for automatic differentiation. Its main object
is the ``adarray`` class.

"""

import numpy as np
import nitrogen.linalg.packed as packed
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
    so on. This ordering is the same as that of :attr:idx
    
    The value of higher-order derivatives is stored by convention
    with a factor equal to the inverse of its multi-index factorial, i.e.
    a derivative with multi-index [2, 0, 1, 3] would be stored as the 
    corresponding derivative divided by 2! * 0! * 1! * 3!
    
    """
    
    def __init__(self,base_shape,k,ni, nck = None, idx = None, dtype = np.float64,
                 d = None):
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
            
        # The first index is the new "derivative index"
    
    def copy(self):
        """
        Copy an adarray object.

        Returns
        -------
        adarray
            A new adarray object, with ``d`` attribute
            copied via ``d.copy()``.

        """
        z = adarray(self.d.shape[1:],self.k,self.ni,
                    self.nck,self.idx,self.d.dtype,self.d.copy())
        return z
        
    # Define binary operators: +, -, ...
    # ADDITION
    def __add__(self,other):
        """ z = self + other """
        
        if np.isscalar(other):
            # Addition of scalar constant
            z = self.copy()
            z.d[0] += other  # Add scalar constant only to value (zeroth derivative)
        else:
            if type(other) != type(self):
                # Try to convert `other` to a constant adarray
                other = const(other,self.k,self.ni,self.nck,self.idx)
            
            z = empty_like(self)
            np.add(self.d,other.d, out = z.d) # z <-- self + other
        return z
    def __radd__(self,other):
        """ z = other + self """
        return self.__add__(other)
    
    # MULTIPLICATION
    def __mul__(self,other):
        """ z = self * other """
        if np.isscalar(other):
            # Multiplication by scalar constant
            z = self.copy()
            z.d *= other # Multiply all derivatives by a scalar constant
        else:
            if type(other) != type(self):
                # Try to convert other to a constant adarray
                other = const(other,self.k,self.ni,self.nck,self.idx)
            
            # z = empty_like(self)
            # Use multi-variate Leibniz product rule, z <-- self * other
            # mvleibniz(self.d,other.d,self.k,self.ni,self.nck,self.idx, out = z.d)
            z = mul(self, other)
        return z
    def __rmul__(self,other):
        """ z = other * self """
        return self.__mul__(other)
    
    # DIVISION
    def __truediv__(self, other):
        """ z = self / other """
        if np.isscalar(other):
            # Division by a scalar constant
            z = self.copy()
            z.d /= other
        else:
            if type(other) != type(self):
                # Try to convert other to a constant adarray
                other = const(other, self.k, self.ni, self.nck, self.idx)
            
            z = div(self, other)
        
        return z
    def __rtruediv__(self, other):
        """ z = other / self """
        # z = (self**-1.0) * other
        return (powf(self,-1.0)).__mul__(other)
    
    # SUBTRACTION
    def __sub__(self,other):
        """ z = self - other """
        if np.isscalar(other):
            # Subtraction of scalar constant
            z = self.copy()
            z.d[0] -= other
        else:
            if type(other) != type(self):
                # Try to convert other to a constant adarray
                other = const(other,self.k,self.ni,self.nck,self.idx)
            
            z = empty_like(self)
            np.subtract(self.d,other.d, out = z.d) # z = self - other
        return z
    
    def __rsub__(self,other):
        """ z = other - self """
        if np.isscalar(other):
            # Subtraction from a scalar constant
            z = -self # adarray negate makes a copy
            z.d[0] += other
        else:
            if type(other) != type(self):
                # Try to convert other to a constant adarray
                other = const(other,self.k,self.ni,self.nck,self.idx)
            
            z = empty_like(self)
            np.subtract(other.d,self.d, out = z.d) # z = other - self
        return z
    
    # UNARY OPERATIONS
    def __neg__(self):
        """ z = -self """
        z = self.copy()
        np.negative(z.d, out = z.d) #z.d = -z.d
        return z
    
    def __pos__(self):
        """ z = +self"""
        z = self.copy()
        return z

def array(d,k,ni,copyd = False):
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
    copyd : boolean
        If True, a copy of `d` will be made for returned
        adarray. If False, the adarray will use 
        the same reference. The default is False.

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
        
    return adarray(base_shape, k, ni, None, None, d.dtype, dinit)
    
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
        The !D array of adarrays.

    """
    
    if np.ndim(x.d) < 2:
        raise ValueError("The base shape must have at least 1 dimension")
    
    n = x.d.shape[1]    # the size of the ndarray
    
    X = np.ndarray((n,), dtype = adarray)
    for i in range(n):
        X[i] = array(x.d[:,i], x.k, x.ni, copyd = False) # keep references
    
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

def mvleibniz(X, Y, k, ni, nck, idx, out=None):
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
    if out is None:
        out = np.zeros(X.shape, dtype = np.result_type(X.dtype,Y.dtype))
    else:
        out.fill(0)
    
    Z = out # Reference only
   
    nd,_ = idx.shape
    
    for iX in range(nd):
        idxX = idx[iX,:]
        kX = np.sum(idxX)
        
        for iY in range(nd):
            idxY = idx[iY,:]
            kY = np.sum(idxY)
            
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

def mvchain(df,X,k,ni,nck,idx, out = None):
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
    
    X0 = X[0].copy() # Value of X
    X[0] = 0  # X now equals "X-X0"
    
    # Initialize result to zero (and create if necessary)
    if out is None:
        out = np.zeros(X.shape, dtype = X.dtype)
    else:
        out.fill(0)
    Z = out # Reference only
    
    for i in range(k+1):
        
        if i == 0:
            # Initialize Xi to 1 (constant)
            Xi = np.zeros(X.shape,dtype = X.dtype)
            Xi[0] = 1.0
            fact = 1.0 # Initialize factoral to 1
        else:
            Xi = mvleibniz(Xi,X,k,ni,nck,idx) # Xi <-- Xi * (X-X0)
            # The call to mvleibniz needs to create a new output buffer
            # because we cannot overwrite Xi while mvleibniz is operating
            # (this probably could be done more efficiently -- revisit)
            fact = fact * i 
            
        if (df[i] != 0 ).any():
            Z += (df[i] / fact) * Xi
    
    X[0] = X0  # Restore the value of X
    
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
    
    c = adarray(value.shape,k,ni,nck,idx,value.dtype)
    
    if np.ndim(c.d) == 1:
        c.d[0] = value
        c.d[1:] = 0
    else:
        np.copyto(c.d[0], value)
        c.d[1:].fill(0)

        
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
    
    if i > ni+1 or i < 0 :
        raise ValueError('Symbol index i must be 0, ..., ni-1')
    
    x = const(value, k, ni, nck, idx)
    
    if k > 0 :
        if np.ndim(x.d) == 1:
            x.d[1:] = 0
            x.d[i+1] = 1
        else:
            x.d[1:].fill(0) 
            x.d[i+1].fill(1)
            
    return x

def empty_like(x, dtype = None):
    """
    Create an *uninitialized* adarray with the same properties as `x`,
    including base array data-type.

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
    
    xval = x.d[0] # Value array of x
    k = x.k
    
    df = np.ndarray( (k+1,)+x.d.shape[1:], dtype = xval.dtype)
    
    df[0] = np.exp(xval)
    for i in range(1,k+1):
        df[i] = df[0]
            
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
            out[i] = adarray(H[0].d.shape[1:], H[0].k, H[0].ni, 
                              H[0].nck, H[0].idx, dtype = H[0].d.dtype)
    
    # Copy H to out for in-place routine
    if out is not H:
        for i in range(H.size):
            np.copyto(out[i].d, H[i].d)
        
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
    N = packed.n2N(n)

    L = np.ndarray((N,N), dtype = adarray)
    # Copy references to adarrays to the lower 
    # triangle of a full "reference" matrix
    # References about the diagonal are undefined.
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
        If out = L, then in-place decomposition is performed

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
            out[i] = adarray(L[0].d.shape[1:], L[0].k, L[0].ni, 
                              L[0].nck, L[0].idx, dtype = L[0].d.dtype)
    
    # Copy L to out for in-place routine
    if out is not L:
        for i in range(L.size):
            np.copyto(out[i].d, L[i].d)
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
    N = packed.n2N(n)
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

    