"""
nitrogen.dfun
-------------

This module implements the class DFun, which 
is used to wrap general differentiable, multi-variate
functions.

"""


import numpy as np
import nitrogen.autodiff.forward as adf
import scipy.optimize as spopt


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
            function with signature ``fx(X, deriv = 0, out = None, var = None)``.
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
            None.

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
    
    def val(self, X, out = None):
        """
        Wrapper for diff. function value (zeroth derivative)

        Parameters
        ----------
        X : ndarray
            An (:attr:`nx`, ...) array of input values.
        out : ndarray, optional
            The (:attr:`nf`, ...) buffer to store the output. If None,
            then a new output ndarray is created. The default is None.

        Returns
        -------
        ndarray : out

        """
    
        d = self.f(X, deriv = 0) # Calculate derivative array
        
        if out is None: 
            out = np.ndarray( (self.nf,) + X.shape[1:], dtype = d.dtype)
            
        np.copyto(out, d[0])     # Copy zeroth derivative
        
        return out

    def jac(self, X, out = None, var = None):
        """
        Wrapper for diff. function Jacobian (first derivatives)

        Parameters
        ----------
        X : ndarray
            An (:attr:`nx`, ...) array of input values.
        out : ndarray, optional
            The (:attr:`nf`, `nvar`, ...) buffer to store the output. If None,
            then a new output ndarray is created. The default is None.
        var : list of int
            Variable list (see `var` in :func:`DFun.f`).
            
        Returns
        -------
        ndarray : out
            ``out[i,j]`` is the derivative of output value ``i`` with
            respsect to variable ``var[j]``

        """
        
        d = self.f(X, deriv = 1, out = None, var = var)
        
        if out is None:
            out = np.ndarray( (self.nf, d.shape[0]-1) + X.shape[1:], dtype = d.dtype)
        
        for i in range(self.nf):
            np.copyto(out[i,:],d[1:,i])
            
        return out
    
    def hes(self, X, out = None, var = None):
        """
        Wrapper for diff. function Hessian (second derivatives)

        Parameters
        ----------
        X : ndarray
            An (:attr:`nx`, ...) array of input values.
        out : ndarray, optional
            The (:attr:`nf`, `nvar`,`nvar`, ...) buffer to store the output. If None,
            then a new output ndarray is created. The default is None.
        var : list of int
            Variable list (see `var` in :func:`DFun.f`).
            
        Returns
        -------
        ndarray : out
            ``out[k,i,j]`` is the second derivative of
            output value ``k`` with respect to variables ``var[i]`` and
            ``var[j]``.

        """
        
        d = self.f(X, deriv = 2, out = None, var = var)
        
        if var is None:
            nvar = self.nx 
        else:
            nvar = len(var)
        
        if out is None:
            out = np.ndarray( (self.nf, nvar, nvar) + X.shape[1:], dtype = d.dtype)
        
        k = nvar + 1
        for i in range(nvar):
            for j in range(i,nvar):
                # (i,j) derivative
                
                if i == j:
                    np.copyto(out[:,i,j], 2.0 * d[k,:])
                else:
                    np.copyto(out[:,i,j], d[k,:])
                    np.copyto(out[:,j,i], d[k,:])
                    
                k = k + 1
        
            
        return out
    
    def optimize(self, X0, fidx = 0, var = None, mode = 'min'):
        """
        Optimize an output value of a diff. function.

        Parameters
        ----------
        X0 : ndarray
            (`nx`,) array containing the initial guess
        fidx : int 
            The DFun function index to optimize. The default is 0.
        var : list of int, optional
            The input variables to optimize. If None, all variables
            will be optimized. The default is None.
        mode : {'min'}, optional
            The optimization mode. 'min' determines the function
            minimum. The default is 'min'.

        Returns
        -------
        Xopt : ndarray
            The optimized input values.
        fopt : float
            The optimized function value.

        """
        
        if var is None:
            var = [i for i in range(self.nx)]
        
        def fun(x):
            X = X0.copy() 
            X[var] = x 
            return self.val(X)[fidx]
        
        if self.maxderiv is None or self.maxderiv >= 1:
            def jac(x):
                X = X0.copy()
                X[var] = x 
                return self.jac(X,var=var)[fidx]
        else:
            jac = None
            
        # if self.maxderiv is None or self.maxderiv >= 2:
        #     def hess(x):
        #         X = X0.copy()
        #         X[var] = x 
        #         return self.hes(X,var=var)[fidx]
        # else: 
        #     hess = None
        
        if mode == 'min':
            res = spopt.minimize(fun, X0[var], method = 'BFGS', 
                                 jac = jac) #, hess = hess)
        
            Xopt = X0.copy()
            Xopt[var] = res.x 
            fopt = res.fun 
        else:
            raise ValueError(f'Unexpected mode type "{mode:s}"')
            
        return Xopt,fopt 
    
    def jacderiv(self, X, deriv = 0, out = None, var = None):
        """
        Calculate the Jacobian *and* its derivatives

        Parameters
        ----------
        X : ndarray
            An (:attr:`nx`, ...) array of input values.
        deriv : int 
            The derivative order of the Jacobian function.
        out : ndarray, optional
            Output buffer. If None, this will be created.
        var : list of int
            Variable list (see `var` in :func:`DFun.f`).
            
        Returns
        -------
        ndarray 
            An array of shape (`nd`, `nvar`, `nf`, ...) 
            where `nvar` is the number of variables requested
            by `var`. 

        """
        
        # Calculate derivates of f to order deriv + 1
        F = self.f(X, deriv = deriv + 1, out = None, var = var)
        
        if var is None:
            var = [i for i in range(self.nx)]
        
        # Calculate nd for the reduced function
        nd, nvar = ndnvar(deriv, var, self.nx)
        
        if out is None:
            out = np.ndarray( (nd, nvar, self.nf) + X.shape[1:], dtype = F.dtype)    
        
        # Extract Jacobian (and its derivatives) from the
        # derivative array of f ("order reduction")
        
        idxtab = adf.idxtab(deriv+1, nvar)
        
        for k in range(self.nf):
            # For the k^th output value
            
            for i in range(nvar):
                # For the i^th requested deriv. variable
                
                adf.reduceOrder(F[:,k], i, deriv + 1, nvar, idxtab, 
                                out = out[:, i, k])
            
        return out
    
class CompositeDFun(DFun):
    
    """
    A composite DFun for C(x) = A(B(x))
    
    Attributes
    ----------
    A : DFun
        The outer function.
    B : DFun
        The inner function.
    
    """
    
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
                H[i] = H[i] + (temp * a[k,i])
                # Note the order of temp*a[k,i]
                # The other way around will use
                # a[k,i] 's __mul__. If a[k,i] is
                # still an ndarray, then it
                # will just broadcast temp
                # to an ndarray -- not what we want
                # 
            
        # Copy H data to out
        for i in range(A.nf):
            np.copyto(out[:,i], H[i].d)
        
        return out

class FixedInputDFun(DFun):
    
    """ 
    Fixed-input differential function
    
    """
    
    def __init__(self, dfunction, values):
        """

        Parameters
        ----------
        dfunction : DFun
            Requested inputs to DFun will
            be held fixed.
        values : list
            A list of length `dfunction.nX` 
            with the fixed input values. A
            list element of None keeps the
            corresponding input active.

        """
        
        if len(values) != dfunction.nx:
            raise ValueError("len(values) must equal dfunction.nx")
        
        new2old = []
        for i in range(len(values)):
            if values[i] is None: # keep active
                new2old.append(i)
        if len(new2old) < 1:
            raise ValueError("There must remain at least one active input")
                
        super().__init__(self._fixedinput, dfunction.nf,
                         len(new2old), dfunction.maxderiv,
                         dfunction.zlevel)
        
        self.values = values
        self.new2old = new2old 
        self.oldfunction = dfunction
        
    
    def _fixedinput(self, X, deriv = 0, out = None, var = None):
        """
        Compute derivative array for a DFun
        with fixed inputs.

        """
        
        # Check for use-all case
        if var is None:
            var = [i for i in range(self.nx)]
        
        # Determine the var argument for the original
        # dfunction
        old_var = [self.new2old[i] for i in var]
        # The original number of inputs
        old_nx = len(self.values)
        base_shape = X.shape[1:]
        
        # Construct the input array for the original dfunction
        oldX = np.ndarray((old_nx,) + base_shape, dtype = X.dtype)
        k = 0
        for i in range(len(self.values)):
            if self.values[i] is None: # Active
                np.copyto(oldX[i:i+1], X[k:k+1])
                k = k + 1
            else: # fixed value
                oldX[i:i+1].fill(self.values[i])
        
        return self.oldfunction.f(oldX, deriv, out, old_var)

    def __repr__(self):
        return f"FixedInputDFun({self.dfunction!r}, {self.values!r})"
        
class FiniteDFun(DFun):
    """
    Finite difference derivatives
    
    Attributes
    ----------
    steps : (nx,) ndarray
        Finite difference step sizes for each argument.
    
    """
    
    def __init__(self, fval, nx, steps = 1e-3, isvector = False, nf = 1):
        """
        Initialize a FiniteDFun instance.

        Parameters
        ----------
        fval : function
            A function ``fval(X)`` that accepts an
            an (`nx`, ...) ndarray input and returns
            an (...) ndarray output (if `isvector` is False) 
            or an (`nf`, ...) ndarray output (if `isvector` is True).
        nx : integer 
            The number of input values.
        steps : scalar or array_like of float
            Finite difference step size. If a single scalar,
            then a uniform step size is used for all 
            input arguments. An array_like list of length `nx` can 
            be used to specify different step sizes for 
            each input argument. The default is 1e-3.
        isvector : bool
            Indicates fval is vector-valued.
        nf : integer
            The number of output values. The default is 1.
            If `isvector` is False, `nf` is ignored.

        """
        
        if not isvector:
            nf = 1 # ignore passed nf value 
        
        super().__init__(self._findiff, nf=nf, nx=nx,
                         maxderiv = 2, zlevel = None)
        
        self._fval = fval 
        self.isvector = isvector
        
        if np.isscalar(steps):
            self.steps = np.full((nx,), steps)
        else:
            if len(steps) != nx:
                raise ValueError("len(steps) must equal nx")
            self.steps = np.array(steps)
        
        
        
    def _findiff(self, X, deriv = 0, out = None, var = None):
        """
        FiniteDFun _feval function

        """
        if var is None:
            var = [i for i in range(self.nx)]
        # deriv check performed by wrapper
        nvar = len(var)
        nd = nderiv(deriv, nvar)
        
        if out is None:
            out = np.ndarray( (nd,self.nf) + X.shape[1:], dtype = X.dtype)
            
        # Define a lambda function to make sure
        # that fval is always in (nf, ...) shape
        if not self.isvector:
            fval = lambda X : (self._fval(X)).reshape((1,) + X.shape[1:])
        else:
            fval = lambda X : self._fval(X)
            
        # Compute value at evaluation point
        f0 = fval(X) 
        np.copyto(out[0:1], f0) # Copy to output
        
        if deriv >= 1:
            
            Xp = X.copy() # Initialize X+
            Xm = X.copy() #   "   "    X-
            
            idx2 = nvar + 1
            for i in range(nvar): # for each requested variables
                
                # Compute derivative for variable idx = var[i]
                idx = var[i] 
                delta = self.steps[idx]  # step size
                
                Xp[idx:(idx+1)] += delta # forward step
                Xm[idx:(idx+1)] -= delta # backward step
                
                fp = fval(Xp)   # forward value
                fm = fval(Xm)   # backward value
                
                fi = (fp - fm) / (2.0 * delta) # central difference
                np.copyto(out[(i+1):(i+2)], fi) # Copy to derivative array
                
                if deriv >= 2: # Diagonal second derivative 
                    # Include the 1/2! = 0.5 permutation factor
                    fii = 0.5 * (fp - 2.0*f0 + fm) / (delta*delta)
                    np.copyto(out[idx2:(idx2+1)], fii)
                    idx2 = idx2 + (nvar-i)
                
                # Re-initialize Xp and Xm
                np.copyto(Xp[idx:(idx+1)], X[idx:(idx+1)])
                np.copyto(Xm[idx:(idx+1)], X[idx:(idx+1)])
                
        if deriv >= 2: # Mixed second derivatives
            
            
            Xpp = X.copy() # Initialize X++
            Xpm = X.copy() # Initialize X+-
            Xmp = X.copy() # Initialize X-+
            Xmm = X.copy() # Initialize X--
            
            k = nvar
            for i in range(nvar):
                for j in range(i,nvar):
                    
                    k = k + 1 
                    
                    if i == j:
                        continue # (i,i) derivative ... already done
                        
                    idx1 = var[i] 
                    idx2 = var[j] 
                    
                    d1 = self.steps[idx1]
                    d2 = self.steps[idx2] 
                    
                    Xpp[idx1] += d1; Xpp[idx2] += d2
                    Xpm[idx1] += d1; Xpm[idx2] -= d2
                    Xmp[idx1] -= d1; Xmp[idx2] += d2
                    Xmm[idx1] -= d1; Xmm[idx2] -= d2
                    
                    fpp = fval(Xpp)
                    fpm = fval(Xpm)
                    fmp = fval(Xmp)
                    fmm = fval(Xmm)
                    
                    # Central mixed derivative
                    fij = (fpp - fmp - fpm + fmm) / (4.0 * d1 * d2)
                    np.copyto(out[k:(k+1)], fij)
                    
                    Xpp = X.copy() # Initialize X++
                    Xpm = X.copy() # Initialize X+-
                    Xmp = X.copy() # Initialize X-+
                    Xmm = X.copy() # Initialize X--       
                    
        return out

    def __repr__(self):
        
        if self.isvector:
            rep = f"""FiniteDFun({self._fval!r}, {self.nx!r}, steps = {self.steps!r},
                              isvector = {self.isvector!r}, nf = {self.nf!r})"""
        else:
            rep = f"FiniteDFun({self._fval!r}, {self.nx!r}, steps = {self.steps!r})"
            
        return rep
    
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