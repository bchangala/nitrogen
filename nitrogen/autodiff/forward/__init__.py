"""
nitrogen.autodiff.forward
-------------------------

This module implements a simple forward accumulation
model for automatic differentiation. Its main object
is the ``adarray`` class.

==============================================   ====================================================
**Constructing adarray objects**                 **Description**
----------------------------------------------   ----------------------------------------------------
:class:`adarray`                                 Class constructor.
:func:`sym`                                      Create a symbol, i.e. an independent variable.
:func:`const`                                    Create a constant.
:func:`const_like`                               Create a constant.
:func:`empty_like`                               Create an uninitialized :class:`adarray`.
==============================================   ====================================================

Mathematical functions implemented include

==============================================   ====================================================
**Arithmetic and powers**                        **Description**
----------------------------------------------   ----------------------------------------------------
:func:`~nitrogen.autodiff.forward.add`           Addition, :math:`x + y`.
:func:`~nitrogen.autodiff.forward.subtract`      Subtraction, :math:`x - y`.
:func:`~nitrogen.autodiff.forward.mul`           Multiplication, :math:`x * y`.
:func:`~nitrogen.autodiff.forward.div`           Division, :math:`x/y`.
:func:`~nitrogen.autodiff.forward.powi`          Integer powers, :math:`x^i`.
:func:`~nitrogen.autodiff.forward.powf`          Real (or complex) powers, :math:`x^p`.
:func:`~nitrogen.autodiff.forward.sqrt`          Square root, :math:`\\sqrt{x}`.
==============================================   ====================================================

==============================================   ====================================================
**Trigometric and hyperbolic**                   **Description**
----------------------------------------------   ----------------------------------------------------
:func:`~nitrogen.autodiff.forward.sin`           Sine, :math:`\\sin(x)`.
:func:`~nitrogen.autodiff.forward.cos`           Subtraction, :math:`\\cos(x)`.
:func:`~nitrogen.autodiff.forward.asin`          Inverse sine, :math:`\\arcsin(x)`.
:func:`~nitrogen.autodiff.forward.acos`          Inverse cosine, :math:`\\arccos(x)`.
:func:`~nitrogen.autodiff.forward.sinh`          Hyperbolic sine, :math:`\\sinh(x)`.
:func:`~nitrogen.autodiff.forward.cosh`          Hyperbolic cosine, :math:`\\cosh(x)`.
:func:`~nitrogen.autodiff.forward.tanh`          Hyperbolic tangent, :math:`\\tanh(x)`.
==============================================   ====================================================

==============================================   ====================================================
**Exponents and logarithms**                     **Description**
----------------------------------------------   ----------------------------------------------------
:func:`~nitrogen.autodiff.forward.exp`           Exponential, :math:`\\exp(x)`.
:func:`~nitrogen.autodiff.forward.log`           Natural logarithm, :math:`\\log(x)`.
==============================================   ====================================================

==============================================   ====================================================
**Linear algebra**                               **Description**
----------------------------------------------   ----------------------------------------------------
:func:`~nitrogen.autodiff.forward.chol_sp`       Cholesky decomposition (symmetric, packed).
:func:`~nitrogen.autodiff.forward.inv_tp`        Triangular matrix inverse (packed).
:func:`~nitrogen.autodiff.forward.llt_tp`        :math:`L L^T` for triangular matrix (packed).
:func:`~nitrogen.autodiff.forward.ltl_tp`        :math:`L^T L` for triangular matrix (packed).
==============================================   ====================================================

Low-level derivative array routines include

==============================================   ====================================================
**Function**                                     **Description**
----------------------------------------------   ----------------------------------------------------
:func:`mvleibniz`                                Leibniz formula for generalized product rule.
:func:`mvchain`                                  Chain rule via Taylor expansion.
:func:`mvrotate`                                 Linear transformation of independent variables.
:func:`mvtranslate`                              Shift a truncated Taylor series.
:func:`mvexpand`                                 Expand derivative array to redundant derivative grid.
:func:`mvexpand_block`                           Expand derivative array to redundant derivative grid.
:func:`mvcompress`                               Compress redundant derivative grid to derivative array.
==============================================   ====================================================

"""

from . import linalg 

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
    zlevels : ndarray
        The zero-levels of individual variables. 
        
        
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
    
    The zero-level indicators `zlevel` and `zlevels` are independent of 
    each other. 
    
    
    The accounting arrays, `nck`, `idx`, and `zlevels` should be considered
    immutable. They might be shared by multiple adarrays. Even `d` might be. In general,
    do not modify adarray attributes directly. 
    
    """
    
    def __init__(self,base_shape,k,ni, nck = None, idx = None, dtype = np.float64,
                 d = None, zlevel = None, zlevels = None):
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
        zlevels : array_like of int, optional
            The zero-levels of each variables. If None,
            each will be set to `k`. The default is None.
        
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
            
        if zlevels is None:
            self.zlevels = np.array([k] * ni) 
        else:
            self.zlevels = np.array(zlevels) 
            
        # In general, zlevel should be
        # between (inclusive) max(zlevels) and sum(zlevels)
        # but that will not be explicitly kept track of.
        #
        # The most exhaustive z-leveling is a boolean
        # mask for each derivative, but I generally think
        # that the combination of the total zlevel
        # and the zlevels for each variables covers 
        # most scenarios. I am trying to reduce overhead
        # for small grid-size evaluations. 
    
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
            
        Notes
        -----
        The `d` and `zlevels` attributes are hard-copied.
        The `nck` and `idx` attributes still share references.

        """
        if out is None:
            out = adarray(self.d.shape[1:],self.k,self.ni,
                    nck = self.nck,
                    idx = self.idx,
                    dtype = self.d.dtype,
                    d = self.d.copy(), # Copy the derivative ndarray itself
                    zlevel = self.zlevel, 
                    zlevels = self.zlevels.copy() )
        else:
            # We assume out has the right shape, data-type, and that
            # nck and idx are already provided
            #
            np.copyto(out.d, self.d)
            out.zlevel = self.zlevel 
            out.zlevels = self.zlevels.copy() 
            
        return out
    
    def view(self):
        """
        Creat an adarray object whose derivative array is 
        a view of this one.
        
        Returns
        -------
        adarray
            An adarray object, with ``d`` attribute
            viewed via ``d.view()``.
            
        Notes
        -----
        The `d` attribute is a view of the original.
        The `zlevels` attribute is just a reference assignment.

        """
        out = adarray(self.d.shape[1:],self.k,self.ni,
                nck = self.nck,
                idx = self.idx,
                dtype = self.d.dtype,
                d = self.d.view(), # View the derivative ndarray
                zlevel = self.zlevel, 
                zlevels = self.zlevels)
        
        return out
        
    def reshape_base(self, new_shape):
        """
        Reshape base array
        
        Parameters
        ----------
        newshape : tuple of ints
        
        Returns
        -------
        adarray
            A view adarray with the deriative array
            referencing the return value of np.reshape
        
        """
        
        nd = self.nd
        
        out = self.view()
        out.d = np.reshape(out.d, (nd,) + new_shape) 

        return out 
    
    def moveaxis_base(self,source,destination):
        """
        Move axes of base array
        
        Parameters
        ----------
        source : int or sequence of int 
            Original positions of axes
        destination : int or sequence of int
            Destination positions of axes
            
        Returns
        -------
        adarray
            A view adarray with the derivative array
            referencing the return value of np.moveaxis
        
        """

        # The axis labels passed in the parameters refer
        # to those of the base shape. Before moveaxis'ing
        # the raw derivative array, the positive entries
        # need to be incremented by 1 to account for the
        # leading derivative axis. The negative entries
        # can remain the same.
        #
        source_new = [ s+1 if s >= 0 else s for s in source ]
        dest_new = [s+1 if s >=0 else s for s in destination]
        
        out = self.view()
        out.d = np.moveaxis(out.d, source_new, dest_new)
        
        return out
    
    def transpose_base(self, axes = None):
        """
        Transpose base array
        
        Parameters
        ----------
        axes : tuple or list of ints, optional
            The same as ndarray transpose. The axis indices
            reference the base shape of the derivative array.
            If None, the default is to reverse base axes.
            
        Returns
        -------
        adarray 
            A view ndarray with the derivative array
            referencing the return value of np.transpose
        
        """
        
        ndim = self.d.ndim - 1 
        
        # Default:
        # axes should be [0,ndim,ndim-1,ndim-2,...,1]
        #
        if axes is None:
            axes = (0,) + tuple(range(ndim,0,-1))
        else:
            # axes is provided. Positive indices need to 
            # be incremented by 1 to account for the 
            # derivative index
            axes = [0] + [ i+1 if i >= 0 else i for i in axes ]
            
            
        out = self.view()
        out.d = np.transpose(out.d, axes=axes)
        
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
            z.zlevels = np.maximum(z.zlevels, 0) 
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
            try:
                z.d *= other # Attempt ndarray imul
            except:
                # Cannot imul, try re-assignment 
                z.d = z.d * other 
            # assuming other to be constant.
            # NumPy broadcasting is important here.
            # If other is a scalar, then everything is multiplied
            # by it. If other is an ndarray of the same base_shape
            # as z, then it will be broadcast over each
            # derivative, as desired.
            #
            # zlevel is unchanged by constant multiplication
            # zlevels is  "  "
            
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
            try: 
                z.d /= other # ndarray idiv, using broadcasting
            except:
                # cannot idiv, try re-assignment 
                z.d = z.d / other 
            # zlevel is unchanged by constant division
            # zlevels " "  " 
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
    
    # MATRIX MULTIPLICATION
    def __matmul__(self,other):
        """ z = self @ other"""
        if type(other) == type(self):
            # adarray @ adarray
            # use ad matmul
            return matmul(self,other)
        else:
            # z = adarray @ constant
            if self.d.ndim <= 1:
                raise ValueError("matmul requires non-scalar base shape")
                
            pre,post = False,False 
            
            if self.d.ndim == 2:
                # Insert 1 in shape 
                # between derivative axis and 
                # vector axis 
                pre = True 
                A = np.expand_dims(self.d, axis = 1)
            else:
                A = self.d 
            
            other = np.array(other)
            if other.ndim == 1 : 
                # Append 1 
                other = np.expand_dims(other, axis = 1) 
                post = True 
            
            # A @ other is now
            # (nd,...,m,n) @ (...,n,k) 
            #
            # Reshape to 
            # (...,nd,m,n) @ (...,1,n,k) 
            # for correct broadcasting
            A = np.moveaxis(A, 0, -3)
            other = np.expand_dims(other, axis = -3) 
            
            result = A @ other 
            
            # result has shape (...,nd,m,k)
            # Move derivative axis back to front
            result = np.moveaxis(result, -3, 0)
            
            # Remove added singletons 
            if pre:
                result = np.squeeze(result, axis = -2)
            if post:
                result = np.squeeze(result, axis = -1) 
            
            z = array(result, self.k, self.ni,
                      zlevel = self.zlevel, zlevels = self.zlevels,
                      nck = self.nck, idx = self.idx)
            
            return z 
    
    def __rmatmul__(self,other):
        """ z = other @ self """
        if type(other) == type(self):
            # This should never be reached, anyway...
            return matmul(other,self) 
        else:
            # z = constant @ adarray 
            if self.d.ndim <= 1:
                raise ValueError("matmul requires non-scalar base shape")
                
            pre,post = False,False 
            
            if self.d.ndim == 2:
                # Insert 1 in shape 
                # between derivative axis and 
                # vector axis 
                post = True 
                A = np.expand_dims(self.d, axis = -1)
            else:
                A = self.d 
            
            other = np.array(other)
            if other.ndim == 1 : 
                # Prepend 1
                other = np.expand_dims(other, axis = 0) 
                pre = True 
            
            # other @ A is now
            # (...,m,n) @ (nd,...,n,k)
            # 
            # Reshape to 
            # (...,1,m,n) @ (...,nd,n,k) 
            # for correct broadcasting
            
            A = np.moveaxis(A, 0, -3)
            other = np.expand_dims(other, axis = -3) 
            
            result = other @ A 
            
            # result has shape (...,nd,m,k)
            # Move derivative axis back to front
            result = np.moveaxis(result, -3, 0)
            
            # Remove added singletons 
            if pre:
                result = np.squeeze(result, axis = -2)
            if post:
                result = np.squeeze(result, axis = -1) 
            
            z = array(result, self.k, self.ni,
                      zlevel = self.zlevel, zlevels = self.zlevels,
                      nck = self.nck, idx = self.idx)
            
            return z 
                
    
    # UNARY OPERATIONS
    def __neg__(self):
        """ z = -self """
        z = self.copy()
        np.negative(z.d, out = z.d) #z.d = -z.d
        # zlevel is unchanged by negation
        # zlevels " " "
        return z
    def __pos__(self):
        """ z = +self"""
        return self.copy()


def array(d,k,ni,copyd = False,zlevel = None, zlevels = None,
          nck = None, idx = None):
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
    zlevels : array_like of int, optional
        The zero-level of each variable. If None,
        this will be set safely to `k` for each. The
        default is None. 
    nck, idx : ndarray, optional
        See :class:`adarray` constructor.

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
        raise ValueError("d must have at least 1 dimension")
    
    if nck is None:
        nck = ncktab(k + ni, min(k,ni))
        
    if d.shape[0] != nck[k+ni,min(k,ni)]:
        raise ValueError("The first dimension of d has an incorrect length")
    
    if zlevel is None:    
        zlevel = k
        
    if zlevels is None:
        zlevels = np.array([k] * ni)
    else:
        zlevels = np.array(zlevels)
    
    return adarray(base_shape, k, ni, nck, idx, d.dtype, dinit, zlevel, zlevels)
    
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
                     copyd = False,     # keep references
                     zlevel = x.zlevel, 
                     zlevels = x.zlevels)
    
    return X    
        

def nck(n,k):
    """
    Calculate the binomial coefficient (`n` choose `k`).
    
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
    >>> adf.nck(4,2)
    6
    
    >>> adf.nck(4,0)
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
    >>> adf.ncktab(3)
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
        
    Examples
    --------
    >>> adf.nderiv(3,2)
    10
    
    >>> adf.nderiv(2,6)
    28

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
    
    Examples
    --------
    >>> adf.nckmulti(np.array([4,2,3]), np.array([2,1,2]), adf.ncktab(4))
    36.0

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
    >>> adf.idxtab(2,3)
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
    >>> adf.idxposk(np.array([0,0,0]),adf.ncktab(3,3))
    0
    
    >>> adf.idxposk(np.array([2,0,1,3]),adf.ncktab(5,2))
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
    >>> adf.idxpos(np.array([0,0,0]),adf.ncktab(3,3))
    0
    
    >>> adf.idxpos(np.array([2,0,1,3]),adf.ncktab(9,4))
    159
        
    See Also 
    -------- 
    ncktab: Binomial coefficient table

    """
    
    #
    # To prevent unwanted int -> float conversions
    # cast everything as uint32. This should cause
    # no overflow issues
    #
    k = np.uint32(np.sum(a)) # Degree of multi-index a
    ni = np.uint32(a.size)   # Number of variables
    one = np.uint32(1)       # 1
    
    if k == 0:
        return np.uint64(0)
    
    else:
        offset = nck[ni + k - one, min(ni,k-one)] # The number of multi-indices with degree
                                                  # less than k
        posk = idxposk(a,nck)    # The position of this multi-index within
                                 # the block of multi-indices of the same degree k
        return offset + posk

def mvexpand(X, k, ni, nck):
    """
    Expand a packed derivative array to
    full symmetric tensors for each derivative degree.

    Parameters
    ----------
    X : ndarray
        A derivative array
    k : int
        Maximum derivative order
    ni : int
        Number of variables
    nck : ndarray
        Binomial table

    Returns
    -------
    partials : list
        A list of arrays for the zeroth, first, second, etc.
        derivatives in full symmetric form.

    """
    
    partials = [] 
    
    idx_low = np.uint64(0)
    for i in range(k+1):
        
        # There are nk derivatives of order i
        nk = nck[i+ni-1, min(i,ni-1)]
        
        idx_high = idx_low + nk
        
        Xk = X[idx_low:idx_high]
        
        partials.append(mvexpand_block(Xk, i, ni, nck))
        
        idx_low = idx_high 
        
    return partials
        
def mvexpand_block(Xk, k, ni, nck):
    """
    Expand the derivative array block of a single total degree

    Parameters
    ----------
    Xk : ndarray
        The block of the derivative array for a single degree
    k : int
        The degree
    ni : int
        The number of variables.
    nck : ndarray
        Binomial table.

    Returns
    -------
    out : ndarray
        The full derivative tensor. The first `k` indices have
        length `ni`. The remaining shape matches the base shape
        of `Xk`.

    """
    
    if k == 0:
        # Zeroth derivative, return value array
        return Xk[0]
    elif k == 1:
        # First derivative, return gradient array
        return Xk[0:ni]
    else:
    
        base_shape = Xk.shape[1:]
        d_shape = tuple([ni] * k) + base_shape 
        
        out = np.zeros((ni**k,) + base_shape, dtype = Xk.dtype) 
        
        grids = np.meshgrid(*[np.arange(ni) for i in range(k)], indexing = 'ij')
        degs = np.stack([g.reshape((-1,)) for g in grids], axis = 1)
        
        N = degs.shape[0] 
        
        one = np.uint64(1)
        
        factorial = [np.math.factorial(i) for i in range(k+1)] 
        
        for i in range(N):
            deg = degs[i,:] 
            idx = np.array([np.count_nonzero(deg == j) for j in range(ni)])
            pos = idxposk(idx, nck) # position in this degree block
            
            c = 1
            for j in range(ni):
                c *= factorial[idx[j]]
                
            np.copyto(out[i:i+1], c * Xk[pos:pos+one])
            
        return out.reshape(d_shape) 

def mvcompress(partials, ni, idx):
    """
    Repack partial derivative arrays to 
    derivative array format 
    
    partials : list
        The derivative tensors for separated by degree
    ni : int
        The number of variables
    idx : ndarray
        The multi-index table
        
    """
    k = len(partials) - 1
    
    nd = idx.shape[0] 
    
    base_shape = np.array(partials[0]).shape 
    
    dtype = np.result_type(partials[0])
    
    X = np.ndarray( (nd,) + base_shape, dtype = dtype)
    
    
    factorial = [np.math.factorial(i) for i in range(k+1)]
    
    for iX in range(nd):
        
        idxX = idx[iX,:]
        
        ki = sum(idxX) # The degree, and the element of partials to index 
        
        if ki == 0:
            # idxX = 0, 0 , 0....
            np.copyto(X[0:1], partials[0])
        else:
            # First or higher derivative
            
            val = partials[ki]  # The full array for the given derivative degree
            # Successively index, and compute factorial
            c = 1
            for i in range(ni): # For each variable
                for j in range(idxX[i]): # For each derivative of this variable
                    val = val[i]
                
                c *= factorial[idxX[i]]
                    
                    
            # val now has shape = base_shape       
            np.copyto(X[iX:iX+1], val / c)
    
    return X 

def mvrotate(X, iT, k, nck, idxnew):
    """
    Rotate the derivative array via a linear transformation. The new 
    coordinates are defined by a matrix :math:`\\mathbf{T}`, i.e.
    :math:`y_i = T_{ij} x_j`.

    Parameters
    ----------
    X : ndarray
        The derivative array with respect to the original coordinates.
    iT : ndarray
        The  **inverse** of the linear transformation matrix. The second
        dimension of `iT` is allowed to be less than `ni`.
    k : int
        The maximum derivative degree.
    nck : ndarray
        Binominal coefficient table for the original number of coordinates.
    idxnew : ndarray
        The multi-index table for the new number of coordinates.
        
    Returns
    -------
    Y : ndarray
        The derivative array with respect to the new coordinates.

    """
    
    ni = iT.shape[0]   # The original number of coordinates
    nnew = iT.shape[1] # The new number of coordinates
    
    # 
    # For now, the linear transformation of
    # derivatives will be carried out on the
    # full, "uncompressed", partial derivative 
    # tensors.
    # 
    # This is not the most efficient, memory-wise
    # (or probably even compute-wise), but it is 
    # simple to get working.
    #
    #
    # Calculate the partial tensors for the original
    # derivatives
    #
    partials = mvexpand(X, k, ni, nck) 
    
    # For each, tensor dot each index with the
    # transpose of the inverse of the transformation
    # array, T.
    #
    new_partials = [partials[0]]
    
    for i in range(1, k+1):
        # Apply iT.T to each of the derivative indices 
        
        temp = partials[i]
        for j in range(i):
            temp = np.tensordot(iT.T, temp, axes = (1,i-1) )
            # Normally, after tensordot, the operated index
            # needs to be moved back into its proper position.
            # However, the symmetry of each of the original and
            # final indices lets us get away without moving
            # it back. Instead, I just always operate on the
            # last derivative index.
            #
        
        new_partials.append(temp.copy())
    
    #
    # Reshape the new partial derivatives into a standard
    # packed derivative array
    #
    return mvcompress(new_partials, nnew, idxnew)
        
    
def mvtranslate(X, D, k, ni, nck, idx, out = None, 
                Xzlevel = None, Xzlevels = None):
    """
    Evaluate a shifted multivariate Taylor series.

    Parameters
    ----------
    X : ndarray
        The derivative array about the initial expansion point.
    D : ndarray
        The expansion point displacement.
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
    Xzlevels : ndarray, optional
        The zlevels of each variable. If None, this is 
        assumed to be `k` for each.

    Returns
    -------
    out : ndarray
        The derivative array about the new expansion point.

    """
    
    base_shape = X.shape[1:]
    D = np.array(D).reshape((ni,) + base_shape)
    
    
    # Initialize result to zero (and create if necessary)
    res_type = np.result_type(X, D)

    if out is None:
        out = np.ndarray(X.shape, dtype = res_type)
    
    if out.dtype != res_type:
        raise TypeError("out data-type is incompatible with [X,D]")
        
    # Initialize result to zero
    out.fill(0)
    
    if Xzlevel is None:
        Xzlevel = k 
    if Xzlevels is None:
        Xzlevels = np.array([k] * ni)    
    
    
    Z = out # Reference only
    
    # Calculate the powers of d
    dpow = np.ones( (k+1,) + D.shape, dtype = D.dtype)
    for i in range(k):
        dpow[i+1] = dpow[i] * D 
        
    # Calculate necessary nchoosek table
    nckc = ncktab(k)
    
    nd,_ = idx.shape
    
    for iD in range(nd):
        idxD = idx[iD,:]   # The powers of d
        
        # Calculate the power of D
        # d0^p0 * d1^p1 * d2^p2 * ...
        #
        Dpow = np.ones(D.shape[1:], dtype = D.dtype)
        for i in range(ni):
            Dpow *= dpow[idxD[i], i]
            
        for iZ in range(nd):
            idxZ = idx[iZ,:] # The result index
            
            idxX = idxD + idxZ # The index of the original derivatives
            kX = np.sum(idxX)
            if kX > Xzlevel:
                break # Skip remaining X derivatives. They are zero
            if (idxX > Xzlevels).any():
                continue # This derivative is zero
            
            # Calculate the multi-index
            # binomial coefficient
            c = 1.0 
            for i in range(ni):
                c *= nckc[idxX[i], min(idxZ[i], idxD[i])]
                
            iX = idxpos(idxX, nck)
            
            Z[iZ] += c * X[iX] * Dpow
            
    
    return Z 
            
def broadcast_shape(sX,sY,mode = 'normal'):
    """
    Calculate the broadcasted shape for different multiplication
    modes.

    Parameters
    ----------
    sX,sY : tuple of int
        The base shapes
    mode : {'normal','matmul'}
        The multiplication mode.

    Returns
    -------
    shape : tuple
        The base shape of the broadcasted result

    """
    
    if mode == 'normal':
        return np.broadcast_shapes(sX,sY)
    elif mode == 'matmul':
        
        if len(sX) == 0 or len(sY) == 0:
            raise ValueError("Cannot matmul scalars")
        pre,post = False,False
        if len(sX) == 1:
            sX = (1,) + sX # Prepend a singleton
            pre = True 
        if len(sY) == 1:
            sY = sY + (1,) # Append a singleton
            post = True 
        
        if sX[-1] != sY[-2]:
            raise ValueError(f"Cannot matmul these shapes : {sX} x {sY}")
        
        newshape = np.broadcast_shapes(sX[:-2], sY[:-2]) + (sX[-2],sY[-1])
        
        if pre :
            # Remove second-to-last
            newshape = newshape[:-2] + (newshape[-1],)
        if post :
            # remove last
            newshape = newshape[:-1] 
        
        return newshape 
        
    else:
        raise ValueError("Unrecognized mode")
        
    return 

def mvleibniz(X, Y, k, ni, nck, idx, out=None, Xzlevel = None, Yzlevel = None,
              Xzlevels = None, Yzlevels = None, mode = 'normal', customfun = None):
    """
    Multivariate Leibniz formula for derivative arrays.

    Parameters
    ----------
    X,Y : ndarray
        Derivative array factors (e.g. :attr:`adarray.d`). These
        must have broadcastable base-shapes.
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
    Xzlevels, Yzlevels : ndarray, optional
        The zero-levels for each variable. If None, this is assumed
        to be `k` for all. The default is None. 
    mode : {'normal','matmul','custom'}, optional
        The multiplication mode. 
    customfun : function
        A function of two arguments.
        
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
    
    Xzlevel and Xzlevels are used independently to determine
    whether certain derivatives are to be skipped. 
    
    """
    
    # Initialize result Z to zero
    res_type = np.result_type(X.dtype, Y.dtype)
    if out is None:
        base_shape = broadcast_shape(X.shape[1:], Y.shape[1:], mode = mode)
        out = np.zeros((X.shape[0],) + base_shape, dtype = res_type)
    else:
        if out.dtype != res_type:
            raise TypeError("out data-type is incompatible with X * Y")
        out.fill(0)
        
    if Xzlevel is None:
        Xzlevel = k 
    if Yzlevel is None: 
        Yzlevel = k 
    
    if Xzlevels is None:
        Xzlevels = np.array([k] * ni)
    if Yzlevels is None:
        Yzlevels = np.array([k] * ni) 
    
    Z = out # Reference only
   
    nd,_ = idx.shape
    
    if mode == 'normal':
        func = np.multiply
    elif mode == 'matmul':
        func = np.matmul
    elif mode == 'custom':
        func = customfun 
    else:
        raise ValueError("multiplication mode unrecognized")
    
    #
    # Sum through all derivatives of X and Y
    # For each pair, accumulate the result 
    # into the appropriate derivative of Z 
    #
    # Derivatives are ordered with lower degrees
    # first.

    for iX in range(nd):
        idxX = idx[iX,:]   # The derivative index of X
        kX = np.sum(idxX)  # The derivative degree
        if kX > Xzlevel:
            break # Skip remaining X derivatives. They are zero.
        if (idxX > Xzlevels).any():
            continue 
        
        for iY in range(nd):
            idxY = idx[iY,:]  # The derivative index of Y
            kY = np.sum(idxY) # The derivative degree 
            if kY > Yzlevel:
                break # Skip remaining Y derivatives. They are zero.
            if (idxY > Yzlevels).any():
                continue 
            
            kZ = kX + kY   # The product's degree
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
            #
            # Numpy broadcasting is applied to this element-wise
            # multiplication 
            Z[iZ] += func(X[iX],Y[iY])
            
    return Z

def mvchain(df,X,k,ni,nck,idx, out = None, Xzlevel = None, Xzlevels = None):
    """
    Multivariate chain rule Z = f(X) via Taylor series.

    Parameters
    ----------
    df : ndarray
        An array containing the derivatives of 
        single-argument function f through order `k`.
        The shape of `df` is ``(k+1,) + X.shape[1:]``
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
    Xzlevels : ndarray, optional
        The zlevels of each variable. If None, this is 
        assumed to be `k` for each.

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
    if Xzlevels is None:
        Xzlevels = np.array([k] * ni) 
    
    # Initialize result to zero (and create if necessary)
    res_type = np.result_type(df, X)
    if out is None:
        out = np.ndarray(X.shape, dtype = res_type)
    
    if out.dtype != res_type:
        raise TypeError("out data-type is incompatible with f(X)")
        
    # Initialize result to zero
    out.fill(0)
        
    Z = out # Reference only
    
    for i in range(k+1):
        
        if i == 0:
            # Initialize Xi = X**i = X**0 to 1 (constant)
            Xi = np.zeros(X.shape,dtype = X.dtype)
            Xi[0] = 1.0
            Xi[:1].fill(1.0)
            Xizlevel = 0 # the zero-level is 0 (constant)
            Xizlevels = np.array([0] * ni)
            fact = 1.0 # Initialize factorial to 1
        else:
            Xi = mvleibniz(Xi,X,k,ni,nck,idx,
                           Xzlevel = Xizlevel, Yzlevel = Xzlevel,
                           Xzlevels = Xizlevels, Yzlevels = Xzlevels) 
            # Xi <-- Xi * (X-X0)
            # The call to mvleibniz needs to create a new output buffer
            # because we cannot overwrite Xi while mvleibniz is operating
            # (this probably could be done more efficiently -- revisit)
            
            # Determine the zlevel of Xi
            if Xizlevel < 0 or Xzlevel < 0: # if either X**i or X is iden. 0
                Xizlevel = -1 # so is the next power
            else:
                Xizlevel = min(Xizlevel + Xzlevel, k)
                
            if (Xizlevels < 0).all() or (Xzlevels < 0).all():
                Xizlevels = np.array([-1] * ni)
            else:
                Xizlevels = np.minimum(Xizlevels + Xzlevels, k) 
            
            fact = fact * i 
            
        if (df[i] != 0 ).any() and Xizlevel >= 0 and (Xizlevels >= 0).all():
            # If both df is non-zero and
            # Xi is non-zero
            Z += (df[i] * (1.0/fact)) * Xi # note broadcast of df[i] over Xi
    
    # Restore the value of X
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
    >>> adf.const(1., 2, 2).d
    array([1., 0., 0., 0., 0., 0.])
    
    >>> adf.const(42j, 1, 3).d
    array([0.+42.j, 0. +0.j, 0. +0.j, 0. +0.j])
    
    >>> adf.const([2.0, 3.0], 2, 2).d
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
    if nck is None:
        nck = ncktab(k + ni, min(k,ni))
    nd = nck[k + ni, min(k,ni)]
    
    d = np.zeros((nd,)+base_shape, dtype = value.dtype)
    
    # Copy value to d[:1]
    np.copyto(d[:1], value)
    
    # Determine the zlevel
    if (value == 0).all():
        zlevel = -1 # Identically zero
        zlevels = np.array([-1] * ni) 
    else:
        zlevel = 0 # A non-zero constant
        zlevels = np.array([0] * ni) 
    
    c = array(d, k, ni, copyd = False, zlevel = zlevel, zlevels = zlevels,
              nck = nck, idx = idx)
        
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
    >>> adf.sym(2.0, 0, 2, 2).d
    array([2., 1., 0., 0., 0., 0.])
    
    >>> adf.sym(3.0, 1, 2, 2).d
    array([3., 0., 1., 0., 0., 0.])
    
    >>> adf.sym([3.0, 4.0j], 1, 2, 2).d
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
        
        x.zlevels = np.array([0] * ni)
        x.zlevels[i] = 1 
        
    # If k == 0, then the zlevel/zlevels from const can be kept
    
    return x

def empty_like(x, dtype = None, baseshape = None):
    """
    Create an *uninitialized* adarray with the same properties as `x`,
    including base array data-type. The zlevel will be maximum.

    Parameters
    ----------
    x : adarray
        Prototype object
        
    dtype : dtype, optional
        Base data-type. If None, then `x.d.dtype` is used.
        
    baseshape : tuple, optional
        The base shape. If None, then the base shape of
            `x` is used.

    Returns
    -------
    adarray
        A new adarray with the same properties as `x`
        
    >>> empty_like(const([3.3, 2.1], 2, 2)).d.shape
    (6, 2)

    """
    if dtype is None:
        dtype = x.d.dtype
    if baseshape is None:
        baseshape = x.d.shape[1:] 
        
    return adarray(baseshape, x.k, x.ni, x.nck, x.idx, dtype)

def const_like(value, x, dtype = None):
    """
    Create a constant adarray initialized to
    `value` with similar properties to `x`.

    Parameters
    ----------
    value : scalar or array_like
        Value.
    x : adarray
        Prototype.
    dtype : dtype, optional
        Data-type. If None, then `x.d.dtype` is used.

    Returns
    -------
    adarray

    """
    
    if dtype is None:
        dtype = x.d.dtype 
        
    base_shape = x.d.shape[1:]
    if np.isscalar(value):
        value = np.full(base_shape, value, dtype=dtype)
    
    return const(value, x.k, x.ni, x.nck, x.idx)

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
        
    Examples
    --------
    >>> x = adf.sym(1.0, 0, 2, 2)
    >>> y = adf.sym(3.0, 1, 2, 2)
    >>> adf.add(x,y).d
    array([4., 1., 1., 0., 0., 0.])

    """
    
    res_type = np.result_type(x.d, y.d)
    if out is None:
        base_shape = np.broadcast_shapes(x.d.shape[1:], y.d.shape[1:])
        out = empty_like(x, dtype = res_type, baseshape=base_shape)
    
    if res_type != out.d.dtype:
        raise TypeError("output data-type incompatible with x + y")
    
    # Perform addition
    # To broadcast correctly over the base_shape,
    # we use a view with the derivative axis moved
    # to the back. 
    np.add(np.moveaxis(x.d,0,-1), 
           np.moveaxis(y.d,0,-1),
           out = np.moveaxis(out.d,0,-1))
    out.zlevel = max(x.zlevel, y.zlevel)
    out.zlevels = np.maximum(x.zlevels, y.zlevels) 
    
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
        
    Examples
    --------
    >>> x = adf.sym(1.0, 0, 2, 2)
    >>> y = adf.sym(3.0, 1, 2, 2)
    >>> adf.subtract(x,y).d
    array([-2.,  1., -1.,  0.,  0.,  0.])

    """
    
    res_type = np.result_type(x.d, y.d)
    if out is None:
        base_shape = np.broadcast_shapes(x.d.shape[1:], y.d.shape[1:])
        out = empty_like(x, dtype = res_type, baseshape=base_shape)
    
    if res_type != out.d.dtype:
        raise TypeError("output data-type incompatible with x - y")
    
    np.subtract(np.moveaxis(x.d,0,-1), 
                np.moveaxis(y.d,0,-1),
                out = np.moveaxis(out.d,0,-1))
    out.zlevel = max(x.zlevel, y.zlevel)
    out.zlevels = np.maximum(x.zlevels, y.zlevels) 
    
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
        
    Examples
    --------
    >>> x = adf.sym(1.0, 0, 2, 2)
    >>> y = adf.sym(3.0, 1, 2, 2)
    >>> adf.mul(x,y).d
    array([3., 3., 1., 0., 1., 0.])

    """
    
    res_type = np.result_type(x.d, y.d)
    if out is None:
        base_shape = np.broadcast_shapes(x.d.shape[1:], y.d.shape[1:])
        out = empty_like(x, dtype = res_type, baseshape=base_shape)
        
    if res_type != out.d.dtype:
        raise TypeError("output data-type incompatible with x * y")

    mvleibniz(x.d,y.d,x.k,x.ni,x.nck,x.idx, out = out.d,
              Xzlevel = x.zlevel, Yzlevel = y.zlevel,
              Xzlevels = x.zlevels, Yzlevels = y.zlevels)
    
    if x.zlevel < 0 or y.zlevel < 0 :
        # Either of the factors is identically zero
        # So is the result.
        out.zlevel = -1
    else:
        # Both zlevels are >= 0
        out.zlevel = min(x.zlevel + y.zlevel, x.k)
    
    if (x.zlevels < 0).any() or (y.zlevels < 0).any():
        out.zlevels = np.array([-1] * x.ni) 
    else:
        out.zlevels = np.minimum(x.zlevels + y.zlevels, x.k) 
    
    return out

def matmul(x, y, out = None):
    """
    Matrix multiply x @ y
    
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
        
    Notes
    -----
    The usual NumPy matmul broadcasting rules apply
    to the base shapes of the derivative arrays.
    """
    
    res_type = np.result_type(x.d, y.d)
    if out is None:
        base_shape = broadcast_shape(x.d.shape[1:], y.d.shape[1:], mode = 'matmul')
        out = empty_like(x, dtype = res_type, baseshape=base_shape)
        
    if res_type != out.d.dtype:
        raise TypeError("output data-type incompatible with x * y")

    mvleibniz(x.d,y.d,x.k,x.ni,x.nck,x.idx, out = out.d,
              Xzlevel = x.zlevel, Yzlevel = y.zlevel,
              Xzlevels = x.zlevels, Yzlevels = y.zlevels,
              mode = 'matmul')
    
    if x.zlevel < 0 or y.zlevel < 0 :
        # Either of the factors is identically zero
        # So is the result.
        out.zlevel = -1
    else:
        # Both zlevels are >= 0
        out.zlevel = min(x.zlevel + y.zlevel, x.k)
    
    if (x.zlevels < 0).any() or (y.zlevels < 0).any():
        out.zlevels = np.array([-1] * x.ni) 
    else:
        out.zlevels = np.minimum(x.zlevels + y.zlevels, x.k) 
    
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
        
    Examples
    --------
    >>> x = adf.sym(1.0, 0, 2, 2)
    >>> y = adf.sym(3.0, 1, 2, 2)
    >>> adf.div(x,y).d
    array([ 0.33333333,  0.33333333, -0.11111111,  0.        , -0.11111111,
            0.03703704])

    """
    
    res_type = np.result_type(x.d, y.d)
    if out is None:
        base_shape = np.broadcast_shapes(x.d.shape[1:], y.d.shape[1:])
        out = empty_like(x, dtype = res_type, baseshape=base_shape)
        
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
    
    mvchain(df, x.d, x.k, x.ni, x.nck, x.idx, out = out.d,
            Xzlevel = x.zlevel, Xzlevels = x.zlevels)
    
    # Determine the zlevel of the result
    # We assume, safely, that df has a maximum zlevel
    if x.zlevel <= 0:
        # x is identically zero or constant
        # The result is also constant
        out.zlevel = 0
    else:
        # x has a zlevel >= 1
        out.zlevel = x.k
        
    # For zlevels, first assume full dependence
    out.zlevels = np.array([x.k]*x.ni)
    #
    # then for any variable i that x does not 
    # depend on (x.zlevels[i] <= 0), neither
    # will the output depend on it
    #
    out.zlevels[x.zlevels <= 0] = 0
    
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
    >>> x = adf.sym(1.0, 0, 3, 1)
    >>> adf.sin(x).d
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
    >>> x = adf.sym(2.0, 0, 3, 1)
    >>> adf.cos(x).d
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

def asin(x, out = None):
    """
    Arcsine for :class:`adarray` objects.

    Parameters
    ----------
    x : adarray
        Input, :math:`x \\in [-1,1]`
    out : adarray, optional
        Output location of result. `out` must have the same
        properties as x. If None, a new adarray is allocated and
        returned.

    Returns
    -------
    adarray
        Result. The real part lies in :math:`[-\\pi/2, \\pi/2]`.
        
    Examples
    --------
    >>> x = adf.sym(0.35, 0, 3, 1)
    >>> adf.asin(x).d
    array([0.3575711 , 1.06752103, 0.21289593, 0.28767379])

    """
    
    xval = x.d[:1] # Value array of x 
    k = x.k 
    
    df = np.ndarray( (k+1,)+x.d.shape[1:], dtype = xval.dtype)
    
    df[0] = np.arcsin(xval) 
    
    #
    # The first derivative of arcsin(x) is 1/sqrt(1-x^2)
    #
    # We'll construct this now 
    #
    y = sym(xval, 0, k, 1)
    z = powf(1.0 - y*y, -0.5) 
    # z.d contains the **scaled** derivatives of 1/sqrt(1-x^2) w.r.t. x
    #
    scale = 1.0 
    for i in range(1, k+1):
        df[i] = z.d[i-1] * scale 
        scale *= i 
        
    return adchain(df, x, out = out) 

def acos(x, out = None):
    """
    Arccosine for :class:`adarray` objects.

    Parameters
    ----------
    x : adarray
        Input, :math:`x \\in [-1,1]`
    out : adarray, optional
        Output location of result. `out` must have the same
        properties as x. If None, a new adarray is allocated and
        returned.

    Returns
    -------
    adarray
        Result. The real part lies in :math:`[0,\\pi]`.
        
    Examples
    --------
    >>> x = adf.sym(0.35, 0, 3, 1)
    >>> adf.acos(x).d
    array([ 1.21322522, -1.06752103, -0.21289593, -0.28767379])

    """
    
    #
    # arccos(x) = pi/2 - arcsin(x)
    #
    
    out = asin(x, out = out)  # Put arcsin(x) into place
    # We are assuming that asin returns with 
    # zlevel/zlevels all >= 0
    out.d *= -1               # Multiply by -1
    out.d[0] += np.pi/2       # Add pi/2 to value
    
    return out
      
# Aliases for acos and asin
def arcsin(x, out = None):
    return asin(x, out)
def arccos(x, out = None):
    return acos(x, out)

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
    >>> x = adf.sym(1.5, 0, 3, 1)
    >>> adf.exp(x).d
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
        
    Examples
    --------
    >>> x = adf.sym(3.0, 0, 3, 1)
    >>> adf.log(x).d
    array([ 1.09861229,  0.33333333, -0.05555556,  0.01234568])
    
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
        np.multiply((-i + 1) * df[1:2], df[(i-1):i], out = df[i:(i+1)])
            
    return adchain(df, x, out = out)

def sinh(x, out = None):
    """
    Hyperbolic sine for :class:`adarray` objects.

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
    >>> x = adf.sym(1.5, 0, 3, 1)
    >>> adf.sinh(x).d
    array([2.12927946, 2.35240962, 1.06463973, 0.39206827])

    """
    
    xval = x.d[:1] # Value array of x
    k = x.k
    
    df = np.ndarray( (k+1,)+x.d.shape[1:], dtype = xval.dtype)
    
    np.sinh(xval, out = df[:1])
    
    # The derivative of sinh is cosh
    # and the derivative of cosh is sinh
    if k >= 1:
        np.cosh(xval, out = df[1:2])
    
    for i in range(2,k+1):
        np.copyto(df[i:(i+1)], df[(i-2):(i-1)])
            
    return adchain(df, x, out = out)

def cosh(x, out = None):
    """
    Hyperbolic cosine for :class:`adarray` objects.

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
    >>> x = adf.sym(1.5, 0, 3, 1)
    >>> adf.cosh(x).d
    array([2.35240962, 2.12927946, 1.17620481, 0.35487991])

    """
    
    xval = x.d[:1] # Value array of x
    k = x.k
    
    df = np.ndarray( (k+1,)+x.d.shape[1:], dtype = xval.dtype)
    
    np.cosh(xval, out = df[:1])
    
    # The derivative of sinh is cosh
    # and the derivative of cosh is sinh
    if k >= 1:
        np.sinh(xval, out = df[1:2])
    
    for i in range(2,k+1):
        np.copyto(df[i:(i+1)], df[(i-2):(i-1)])
            
    return adchain(df, x, out = out)

def tanh(x, out = None):
    """
    Hyperbolic tangent for :class:`adarray` objects.

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
    >>> x = adf.sym(1.5, 0, 3, 1)
    >>> adf.tanh(x).d
    array([ 0.90514825,  0.18070664, -0.1635663 ,  0.0878162 ])

    """
    
    sh = sinh(x)
    ch = cosh(x)
    
    # tanh = sinh / cosh
    
    return div(sh, ch, out = out)


def powi(x, i, out = None):
    """
    x**i for integer i
    
    Parameters
    ----------
    x : adarray
        Base argument.
    i : integer
        Integer exponent.
    out : adarray, optional 
        Output location of result.
        
    Returns
    -------
    adarray 
        Result.
    
    Notes
    -----
    If `i` == 0, then powi returns a constant 1 for 
    any value of `x`. For negative `i`, `x` is inverted
    and then the positive power is applied to 1/`x`.
    
    Examples
    --------
    >>> x = adf.sym(1.5, 0, 3, 1)
    >>> adf.powf(x, 3).d
    array([3.375, 6.75 , 4.5  , 1.   ])
    
    """
    
    if out is None:
        out = empty_like(x)
    
    # If exponent is negative,
    # then invert argument and
    # and sign of exponent
    if i < 0:
        x = powf(x, -1.0) # 1 / x
        i = -i
    
    # Check for identity, x**1 = x
    #
    if i == 1: # Just a copy
        np.copyto(out.d, x.d)
        out.zlevel = x.zlevel
        out.zlevels = x.zlevels
        return out
    
    # Otherwise,
    # initialize out to constant 1.
    res = const_like(1.0, x)
    
    # Do right-to-left binary
    # exponentiation
    a = x
    while True:
        if i % 2 == 1:
            res = res * a 
            
        i = i // 2  # floor division
        if i == 0:
            break 
        a = a * a
    
    np.copyto(out.d, res.d)
    out.zlevel = res.zlevel
    out.zlevels = res.zlevels 
        
    return out

def powf(x, p, out = None):
    """
    x**p for general p

    Parameters
    ----------
    x : adarray
        Base argument.
    p : float or complex
        Exponent.
    out : adarray, optional
        Output location of result.

    Returns
    -------
    adarray
        Result.
        
    Notes
    -----
    powf uses the NumPy 
    `float_power() <https://numpy.org/doc/stable/reference/generated/numpy.float_power.html#numpy.float_power>`_ 
    function to 
    compute the value array. It inherits the branch-cut convention
    of this function.
    
    Examples
    --------
    >>> x = adf.sym(1.5, 0, 3, 1)
    >>> adf.powf(x, -2.5).d
    array([ 0.36288737, -0.60481228,  0.70561433, -0.70561433])
    
    """
    
    if np.result_type(x.d.dtype, p) != np.result_type(x.d.dtype):
        raise TypeError("Invalid type combination")
    
    xval = x.d[0] # Value array of x
    k = x.k 
    
    df = np.ndarray( (k+1,) + x.d.shape[1:], dtype = xval.dtype)
    
    df[0] = np.float_power(xval, p)
    for i in range(1, k+1):
        df[i] = (p-i+1) * (df[i-1] / xval)
    
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
    The adarray sqrt function uses the NumPy 
    `sqrt() <https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html#numpy.sqrt>`_ 
    function as its underlying routine. Its branch-cut convention is inherited.
    
    Examples
    --------
    >>> x = adf.sym(2.5, 0, 3, 1)
    >>> adf.sqrt(x).d
    array([ 1.58113883,  0.31622777, -0.03162278,  0.00632456])
    """
    
    xval = x.d[0] # Value array of x
    k = x.k 
    
    df = np.ndarray( (k+1,) + x.d.shape[1:], dtype = xval.dtype)
    
    df[0] = np.sqrt(xval) # Uses numpy branch cut
    
    if k >= 1:
        ixval = 1.0 / xval # The inverse value
    for i in range(1, k+1):
        df[i] = (1.5 - i) * (df[i-1] * ixval)
    
    return adchain(df, x, out = out)


    
def reduceOrder(F, i, k, ni, idx, out = None):
    """
    Reduce the derivative array for F with respect to 
    variable i. The returned derivative array is
    that for the function :math:`\\partial_i F`.

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
        The derivative array with shape (`nd_reduced`, ...)
        corresponding to the new function :math:`\\partial_i F`

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

def reduceOrderTwice(F, i, j, k, ni, idx, out = None):
    """
    Reduce the derivative array for F with respect to 
    variables i and j. The returned derivative array is
    that for the function :math:`\\partial_i \\partial_j F`.
    (Note that this function is an unscaled derivative.)

    Parameters
    ----------
    F : ndarray
        The derivative array up to degree `k`
        in `ni` variables.
    i,j : int
        The variable index (0, ..., `ni` - 1) to
        reduce. `i` may equal `j`.
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
        The derivative array with shape (`nd_reduced`, ...)
        corresponding to the new function :math:`\\partial_i \\partial_j F`

    """

    nd,_ = idx.shape # The number of derivatives
    
    if k <= 1:
        raise ValueError("Cannot reduce a derivative array with k = 0 or 1") 
        
    if i < 0 or i >= ni : 
        raise ValueError(f"Cannot reduce w.r.t variable index i = {i:d} with only ni = {ni:d} variables")
    if j < 0 or j >= ni : 
        raise ValueError(f"Cannot reduce w.r.t variable index j = {j:d} with only ni = {ni:d} variables")
        
    # k is > 1 and ni is > 0 
    #
    # Calculate the number of derivatives for
    # order k - 2 in ni variables. We already know
    # that nd is the number of deriv for order k
    # in ni variables, so we can use the simple result:
    nd_reduced = (nd * k * (k-1) ) // ( (k + ni) * (k + ni - 1) )
    # This should be an integer result always!
    
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
    # a' = [a0 a1 a2 ... ai-1 ... aj-1 ... ] and b' = [b0 b1 b2 ... bi-1 ... bj-1 ...]
    # appear in the same order: a' before b'
    # (regardless of whether i == j)
    #
    iG = 0 # running position in the reduced derivative array 
    for iF in range(nd):
        
        idxF = idx[iF,:]  # The multi-index of the original function's derivatives
        
        
        # Check whether the reduced deriative 
        # exists. If i == j, then the derivative order
        # for variable i must be >= 2. If i != j, 
        # then both must be >= 1
        #
        if i == j and idxF[i] < 2 :
            continue 
        if i !=j and (idxF[i] < 1 or idxF[j] < 1):
            continue 
        
        # The reduced derivative exists.
        #
        # Because we store the derivatives with a factor
        # equal to the inverse of the multi-index factorial,
        # we need to correct this in the reduced array
        #
        # The correct conversion factor is idxF! / idxG! 
        #
        # If i != j, then this ratio is just idxF[i] * idxF[j].
        #
        # If i == j, then this ratio is idxF[i] * (idxF[i] - 1)
        #
        # Now we do this:
        # G[iG] = F[iF] * ratio
        #
        if i == j:
            ratio = idxF[i] * (idxF[i] - 1) 
        else:
            ratio = idxF[i] * idxF[j] 
            
        # Depending on the shape of G, we will split this up
        if len(G.shape) == 1:
            G[iG] = F[iF] * ratio # Scalar multiplication
        else:
            np.multiply(F[iF], ratio, out = G[iG]) # use np.multiply to save on temp memory
        
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
        
    Examples
    --------
    >>> adf.n2N(21)
    6

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


def cost(k,ni, quiet = False):
    """
    Estimate the cost of adarray operations.

    Parameters
    ----------
    k : int
        The total degree, :math:`k \\geq 0`
    ni : int
        The number of variables, :math:`n_i \\geq 1`
    quiet : bool, optional
        Suppress printed output. The default is False.

    Returns
    -------
    None

    """
    
    nck = ncktab(k + 2*ni)
    k = np.uint32(k) 
    two = np.uint32(2)
    one = np.uint32(1) 
    
    nd = nck[k+ni,k] # The number of derivatives
    
    # First, calculate the memory cost. This equals
    # the number of derivatives stored
    mem_cost = nd
    
    # Estimate the number of floating point operations
    # required for some simple arithmetic. (Assuming real numbers.)
    #
    # Addition requires the addition of each derivative separately
    add_cost = nd
    #
    # Multiplication requires the generalized Leibniz formula,
    # "mvleibniz". This has
    # sum_(kp=0)^k  (kp + n - 1 choose kp) (k-kp+ni choose k-kp)
    #      =  (k + 2n choose k) 
    # terms. Each term requires one multiplication and one addition
    # (to a given derivative of the result) for a total of 2 FLOPs 
    # per term
    mul_cost = two * nck[k + 2*ni, k]
    
    #
    # Now calculate the cost of a call to mvchain
    # This assumes df is already calculated.
    # 
    # The total includes
    #  (k+1) * mul_cost (for calculating powers) and
    #  (k+1) * (2*nd + 1) for FLOPs scaling by df and summing result.
    mvc_cost = (k + one) * (mul_cost + two*nd + one) 
    #
    # Any actual usage of mvchain requires the calculation
    # of df, which will be variable depending on the 
    # function itself. Often these go as ~ k primitive multiplies,
    # which is typically small compared to the remaining cost.
    #
    fun_cost = mvc_cost + k + one  
    if not quiet:
        print(f"Memory scaling         = {mem_cost:d}")
        print(f"Addition FLOPs         = {add_cost:d}")
        print(f"Multiplication FLOPs   = {mul_cost:d}")
        print(f"Chain rule FLOPs       = {mvc_cost:d}")
        print(f"Typical function eval. = {fun_cost:d}")
     
    
    return

def block2(arrays):
    """
    Assemble an array from 2-D nested list of sub-arrays
    
    Parameters
    ----------
    arrays : list of list of adarray
        The blocks 
    
    Returns
    -------
    adarray
        The assembled array
        
    Notes
    -----
    This performs numpy.block on the base arrays of the 
    adarray objects
    
    """
    
    nd = arrays[0][0].nd # The number of derivatives 
    
    
    for k in range(nd):
        # Assemble the block matrix for this derivative 
        
        blocks = [ [X.d[k] for X in row] for row in arrays] 
        # 
        # The base array for this derivative
        Md = np.block(blocks) 
        
        if k==0: 
            # Create the output array 
            M = empty_like(arrays[0][0], baseshape=Md.shape)
        
        # Copy Md to the appropriate derivative sub-array
        np.copyto(M.d[k], Md)
    
    
    return M 
            
def block4(arrays):
    """
    Assemble an array from 4-D nested list of sub-arrays
    
    Parameters
    ----------
    arrays : nested list of adarray
        The blocks 
    
    Returns
    -------
    adarray
        The assembled array
        
    Notes
    -----
    This performs numpy.block on the base arrays of the 
    adarray objects
    
    """
    
    nd = arrays[0][0][0][0].nd # The number of derivatives 
    
    
    for k in range(nd):
        # Assemble the block matrix for this derivative 
        
        blocks = [[[[block.d[k] for block in list3] for list3 in list2] for list2 in list1] for list1 in arrays]
        # 
        # The base array for this derivative
        Md = np.block(blocks) 
        
        if k==0: 
            # Create the output array 
            M = empty_like(arrays[0][0][0][0], baseshape=Md.shape)
        
        # Copy Md to the appropriate derivative sub-array
        np.copyto(M.d[k], Md)
    
    
    return M 


def tensordot(A,B,axes=2):
    """
    Perform a tensor dot product of base axes
    
    Parameters
    ----------
    A,B : adarray
        Tensors to contract
    
    axes : int or (2,) array_like
        The contraction axes. See numpy.tensordot
        
    Returns
    -------
    adarray
        The tensor dot result
        
    """
    
    A_base = A.d.shape[1:] # The base shape of A
    B_base = B.d.shape[1:] # The base shape of B 
    
    idx = A.idx # The index table 
    nck = A.nck # The binomial coefficient table 
    
    if np.isscalar(axes): # A scalar
         # The last N axes of A and the first
         # N axes of B will be contracted. The resulting
         # shape is whatever is left over
         
         C_base = A_base[:-axes] + B_base[axes:]
    else: 
        if len(axes) != 2:
            raise ValueError("axes must be a (2,) array_like")
        if len(axes[0]) != len(axes[1]):
            raise ValueError("Each element of axes must be the same length")
        
        A_con = np.zeros((len(A_base),) , dtype = bool)
        B_con = np.zeros((len(B_base),) , dtype = bool)
        A_con[list(axes[0])] = True  # marked contracted indices as True
        B_con[list(axes[1])] = True 
        
        A_leftover = tuple(np.array(A_base)[~A_con]) # leftover dimensions
        B_leftover = tuple(np.array(B_base)[~B_con])
        
        C_base = A_leftover + B_leftover
    
    res_type = np.result_type(A.d, B.d)
    
    # Allocate the result derivative array 
    out = empty_like(A, dtype = res_type, baseshape=C_base)
    
    tensordot_fun = lambda X, Y : np.tensordot(X,Y, axes = axes)
    
    mvleibniz(A.d, B.d, A.k, A.ni, nck, idx, out = out.d,
              Xzlevel = A.zlevel, Yzlevel = B.zlevel,
              Xzlevels = A.zlevels, Yzlevels = B.zlevels, 
              mode = 'custom', customfun = tensordot_fun)
    
    
    # Do zlevel logic
    
    if A.zlevel < 0 or B.zlevel < 0 :
        # Either of the factors is identically zero
        # So is the result.
        out.zlevel = -1
    else:
        # Both zlevels are >= 0
        out.zlevel = min(A.zlevel + B.zlevel, A.k)
    
    if (A.zlevels < 0).any() or (B.zlevels < 0).any():
        out.zlevels = np.array([-1] * A.ni) 
    else:
        out.zlevels = np.minimum(A.zlevels + B.zlevels, A.k) 
        
    return out 
        
        
        
def calc_product_table(k, ni):
    """
    Calculate the sorted direct product table for derivative 
    array product.

    Parameters
    ----------
    k : integer
        The maximum derivative order.
    ni : integer
        The number of variables.

    Returns
    -------
    table : (3,tablesize) ndarray
        The sorted direct product table. The elements equal 1-D indices 
        of standard derivative array lexical ordering.
    
    Notes
    -----
    
    The generalized Leibniz product for scaled derivative arrays is
    
    .. math::
        
        Z^{(\\gamma)} = \\sum_{\\alpha \\leq \\gamma} X^{(\\alpha)} Y^{(\\beta = \\gamma - \\alpha)}
        
    The direct product table pre-computes all multi-index triplets
    :math:`(\\gamma, \\alpha, \\beta = \\gamma - \\alpha)`
    in terms of their one-dimensional derivative array index:
    
    ``Z[table[0,i]] <-- X[table[1,i]] * Y[table[2,i]]``
    
    This function returns a *sorted* table. A sorted
    table satisfies these conditions:
        
        1) The one-dimensional indices are ordered such that if
           :math:`\\alpha < \\beta`, then :math:`idx(\\alpha) < idx(\\beta)`.
           (The standard derivative array lexical ordering is sorted by 
           :math:`\\vert \\alpha \\vert`, which guarantees it is also *sorted*
           in this formal sense.)
        2) ``table`` is sorted by ascending order of ``table[0]``.
        3) For equal elements of ``table[0]``, the table is sorted by ``table[1]``.
    
    This ordering of the direct product table is useful for derivative array
    routines that construct derivative arrays recursively.
    
    """
    
    nck = ncktab(k+ni)   # Binomial coefficients
    idx = idxtab(k, ni)  # The derivative indices 
    
    nd = idx.shape[0] # The number of derivatives 
    
    tZ = [] 
    tX = [] 
    tY = [] 
    
    for iX in range(nd):
        kX = sum(idx[iX])
        
        for iY in range(nd):
            kY = sum(idx[iY]) 
            
            if kX + kY > k :
                continue # product order out of range 
            
            idxZ = idx[iX] + idx[iY] # The product index 
            iZ = idxpos(idxZ, nck) 
            
            tZ.append(iZ)
            tX.append(iX)
            tY.append(iY) 
    
    table = np.array([tZ,tX,tY], dtype = np.int32) 
    
    #
    # As constructed, the table is sorted by iX then iY
    # 
    # We need to re-sort by iZ. If we use a *stable* sort,
    # then for a given value of iZ, the rows will remain
    # sorted by iX.
    #
    I = np.argsort(table[0], kind = 'stable')
    table = table[:,I]
    
    return table


def _mul_ad_table(dZ,dX,dY,product_table, reduce = False):
    """
    Compute Leibniz product using product table.

    Parameters
    ----------
    dZ : (nd, ...) ndarray
        Derivative array output.
    dX,dY : (nd, ...) ndarray
        Derivative array input.
    product_table : (3,nt)
        The sorted product table 
    reduce : bool
        If True, add result to dZ 

    Returns
    -------
    None.

    """
    
    nt = product_table.shape[1] 
    
    if not reduce: 
        dZ.fill(0)
        
    for i in range(nt):
        iZ,iX,iY = product_table[:,i]
        
        # np.multiply(dX[iX:(iX+1)], dY[iY:(iY+1)], out = dZ[iZ:(iZ+1)])
        dZ[iZ] += dX[iX] * dY[iY] 
    
    return 