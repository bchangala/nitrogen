"""
nitrogen.dfun
-------------

This module implements the class DFun, which 
is used to interface general differentiable, multi-variate
functions. See :doc:`tutorials/dfun` for an in-depth tutorial.


===========================   ===================================
**Create DFun objects from functions**
-----------------------------------------------------------------
:class:`DFun`                 The :class:`DFun` constructor
:class:`FiniteDFun`           Finite-difference derivatives
---------------------------   -----------------------------------
**Modify single DFun objects**
-----------------------------------------------------------------
:class:`FixedInputDFun`       Fix input value(s).
:class:`PermutedDFun`         Permute input and output ordering.
:class:`ArrangedDFun`         Rearrange and duplicate outputs.
---------------------------   -----------------------------------
**Combine multiple DFun objects**
-----------------------------------------------------------------
:class:`CompositeDFun`        Function composition.
:class:`MergedDFun`           Concatenate outputs.
:class:`SimpleProduct`        Separable direct product.
:class:`SelectedProduct`      Separable non-direct product.
===========================   ===================================

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
            of None indicates all derivatives may be non-zero, while
            -1 indicates an identically zero function and 0 a constant function.
            The default is None.

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
        X : array_like
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
        X = np.array(X) 
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
    
    def vj(self, X, var = None):
        """
        Wrapper for diff. function value and Jacobian

        Parameters
        ----------
        X : ndarray
            An (:attr:`nx`, ...) array of input values.
        var : list of int
            Variable list (see `var` in :func:`DFun.f`).
            
        Returns
        -------
        val : ndarray
            The (nf,...) value.
        jac : ndarray
            The (nf,nv,...) Jacobian

        """
        
        if var is None:
            nvar = self.nx 
        else:
            nvar = len(var) 
            
        base_shape = X.shape[1:]
        
        d = self.f(X, deriv = 1, var = var) # Calculate 0th, 1st, and 2nd derivs
        
        #
        # d has shape (nd, nf, ...)
        #
        
        # Copy 0th derivatives to value
        val = d[0,:].copy() 
        
        # Copy 1st derivatives to jacobian
        # Notice swapping order of nd/nf from ndarray to jacobian
        #
        jac = np.ndarray( (self.nf, nvar,) + base_shape, dtype = d.dtype)
        for i in range(self.nf):
            np.copyto(jac[i,:], d[1:(nvar+1),i])
     
        return val, jac
    
    def vjh(self, X, var = None):
        """
        Wrapper for diff. function value, Jacobian, and Hessian

        Parameters
        ----------
        X : ndarray
            An (:attr:`nx`, ...) array of input values.
        var : list of int
            Variable list (see `var` in :func:`DFun.f`).
            
        Returns
        -------
        val : ndarray
            The (nf,...) value.
        jac : ndarray
            The (nf,nv,...) Jacobian
        hes : ndarray
            The (nf,nv,nv,...) Hessian

        """
        
        if var is None:
            nvar = self.nx 
        else:
            nvar = len(var) 
            
        base_shape = X.shape[1:]
        
        d = self.f(X, deriv = 2, var = var) # Calculate 0th, 1st, and 2nd derivs
        
        #
        # d has shape (nd, nf, ...)
        #
        
        # Copy 0th derivatives to value
        val = d[0,:].copy() 
        
        # Copy 1st derivatives to jacobian
        # Notice swapping order of nd/nf from ndarray to jacobian
        #
        jac = np.ndarray( (self.nf, nvar,) + base_shape, dtype = d.dtype)
        for i in range(self.nf):
            np.copyto(jac[i,:], d[1:(nvar+1),i])
        
        # Copy 2nd derivatives to hessian
        hes = np.ndarray( (self.nf, nvar, nvar) + base_shape, dtype = d.dtype)
        
        idx = nvar + 1 
        for i in range(nvar):
            for j in range(i,nvar):
                # (i,j) derivative
                
                if i == j:
                    np.copyto(hes[:,i,j], 2.0 * d[idx,:])
                else:
                    np.copyto(hes[:,i,j], d[idx,:])
                    np.copyto(hes[:,j,i], d[idx,:])
                    
                idx = idx + 1
        
        return val, jac, hes 
        
    
    def optimize(self, X0, fidx = 0, var = None, mode = 'min', disp = False):
        """
        Optimize an output value of a diff. function.

        Parameters
        ----------
        X0 : array_like
            (`nx`,) array containing the initial guess
        fidx : int 
            The DFun function index to optimize. The default is 0.
        var : list of int, optional
            The input variables to optimize. If None, all variables
            will be optimized. The default is None.
        mode : {'min'}, optional
            The optimization mode. 'min' determines the function
            minimum. The default is 'min'.
        disp : bool
            If True, messages will be displayed.

        Returns
        -------
        Xopt : ndarray
            The optimized input values.
        fopt : float
            The optimized function value.

        """
        
        X0 = np.array(X0)
        
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
                                 jac = jac, options = {'disp':disp} ) #, hess = hess)
        
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
        
        # Calculate derivatives of f to order deriv + 1
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
    
    def _parse_out_var(self, X, deriv, out, var):
        """ Parse a out and var, which may be None"""
        if var is None:
            var = [i for i in range(self.nx)]
        if out is None:
            nd, nvar = ndnvar(deriv, var, self.nx)
            out = np.ndarray( (nd, self.nf) + X.shape[1:], dtype = X.dtype)
        
        return out, var
        
        
    
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
    
    def __repr__(self):
        return f"CompositeDFun({self.A!r}, {self.B!r})"

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
    
    
class MergedDFun(DFun):
    """
    Concatenate the outputs of multiple DFuns
    into a single, merged DFun.
    
    """
    
    def __init__(self, dfuns):
        """
        Create a merged DFun with the outputs of
        multiples DFuns.
           
        Parameters
        ----------
        dfuns : list of DFun
            The component DFun's.
           
        """
        
        if len(dfuns) == 0:
            raise ValueError("dfuns list must have at least one element")
        
        nx = dfuns[0].nx 
        # Check that each DFun has the same number of inputs
        for df in dfuns:
            if df.nx != nx:
                raise ValueError("All DFun's must have the same number of inputs.")
        
        nf = 0
        # The new number of outputs is the sum of
        # DFun's outputs
        for df in dfuns:
            nf += df.nf 
        
        # Determine the maxderiv and zlevel of the
        # merged functions
        maxderiv = None 
        zlevel = -1
        for df in dfuns:
            maxderiv = _merged_maxderiv(df.maxderiv, maxderiv)
            zlevel = _merged_zlevel(df.zlevel, zlevel) 
        
        super().__init__(self._fmerged, nf, nx, maxderiv, zlevel)
        
        self.dfuns = dfuns 
        
        return 
    
    def _fmerged(self, X, deriv = 0, out = None, var = None):
        """
        Merged DFun derivative array evaluation function.
        
        """
        # Set up output
        nd,nvar = ndnvar(deriv, var, self.nx)
        if out is None:
            base_shape = X.shape[1:]
            out = np.ndarray( (nd, self.nf) + base_shape, dtype = X.dtype)
            
        # Place the derivative array outputs 
        # of each DFun into the combined output
        # array, `out`, in order
        
        offset = 0
        for df in self.dfuns:
            df.f(X, deriv = deriv, out = out[:,offset:(offset+df.nf)], var = var)
            offset += df.nf
           
        return out
    
    def __repr__(self):
        return f"MergedDFun({self.dfuns!r})"
           
class PermutedDFun(DFun):
    """
    Permute the input or output order of a DFun
    
    Attributes
    ----------
    
    df : DFun
        The original DFun
    in_order : ndarray
        The new input order 
    out_order : ndarray
        The new output order
    
    """
    
    def __init__(self, df, in_order = None, out_order = None):
        """
        

        Parameters
        ----------
        df : DFun
            A DFun object.
        in_order : array_like of int, optional
            The new input order. If None (default), the original
            input order is used. For example, [2, 0, 1] moves
            the third input first, the first input second, and the
            second input last.
        out_order : array_like of int, optional
            The new output order. If None (default), the original
            output order is used.        

        """
        
        nx = df.nx 
        nf = df.nf 
        
        if in_order is None:
            in_order = np.arange(nx)
        else:
            in_order = np.array(in_order)
            
        if out_order is None:
            out_order = np.arange(nf)
        else:
            out_order = np.array(out_order)
            
        if not np.all(np.sort(in_order) == np.arange(nx)):
            raise ValueError('invalid in_order')
        if not np.all(np.sort(out_order) == np.arange(nf)):
            raise ValueError('invalid out_order') 
        
        
        super().__init__(self._fpermute, nf, nx, df.maxderiv, df.zlevel)
        
        self.df = df
        self.in_order = in_order
        self.out_order = out_order 
        
        
        # create the reverse direction in_order 
        reverse_in = [None for i in range(nx)]
        for i in range(nx):
            reverse_in[in_order[i]] = i
        
        self._reverse_in = np.array(reverse_in)
        
        return

    def _fpermute(self, X, deriv = 0, out = None, var = None):
        
        #
        # X is in the input variables in the new order
        # Permute them back to the old order first
        #
        X_old = X[self._reverse_in]
        
        #
        # var is the requested variables refering to new
        # input labels. Translate this to old input labels
        #
        if var is None:
            var = [i for i in range(self.nx)]
        
        var_old = [self.in_order[v]  for v in var]
        
        # 
        # Evaluate original dfun 
        #
        out_old = self.df.f(X_old, deriv = deriv, out = None, var = var_old) 
        
        #
        # Finally, re-order the output values **in-place**
        # 
        
        # Set up output
        nd,nvar = ndnvar(deriv, var, self.nx)
        if out is None:
            base_shape = X.shape[1:]
            out = np.ndarray( (nd, self.nf) + base_shape, dtype = X.dtype)
            
        # Copy old output to new order 
        for i in range(self.nf):
            iold = self.out_order[i]
            np.copyto(out[:,i], out_old[:,iold])
            
        return out 
    
    def __repr__(self):
        return f"PermutedDFun({self.df!r}, in_order = {self.in_order!r}, out_order = {self.out_order!r})"

class ArrangedDFun(DFun):
    """
    Select a subset or duplicates of output functions.
    
    Attributes
    ----------
    
    df : DFun
        The original DFun
    select : ndarray
        The output functions in terms of the indices
        of the original output functions.
    
    """
    
    def __init__(self, df, select):
        """
        
        Parameters
        ----------
        df : DFun
            A DFun object.
        select : array_like of int
            The new output functions. Repetitions are allowed.

        """
        
        nx = df.nx 
        
        select = np.array(select, copy = True)
        nf = len(select)
            
        if np.any(select < 0) or np.any(select >= df.nf):
            raise ValueError("select has an element out of bounds")
        
        super().__init__(self._farrange, nf, nx, df.maxderiv, df.zlevel)
        
        self.df = df
        self.select = select 
        
        return

    def _farrange(self, X, deriv = 0, out = None, var = None):
        
        #
        # Evaluate the complete set of functions
        #
        out_all = self.df.f(X, deriv = deriv, out = None, var = var) 
        
        # Set up output
        nd,nvar = ndnvar(deriv, var, self.nx)
        if out is None:
            base_shape = X.shape[1:]
            out = np.ndarray( (nd, self.nf) + base_shape, dtype = X.dtype)
            
        # Copy old output to new order 
        for i in range(self.nf):
            iold = self.select[i]
            np.copyto(out[:,i], out_all[:,iold])
            
        return out 
    
    def __repr__(self):
        return f"ArrangedDFun({self.df!r}, {self.select!r})"


class SimpleProduct(DFun):
    """
    
    Create a product function from 
    mutually independent factors.
    
    .. math:
       
       f(x,y,\cdots) = g(x)h(y) \cdots
       
    Multi-dimensional factors may be supported in the future.
    
    """
    
    def __init__(self, factors):
        """
    
        Parameters
        ----------
        factors : list of DFun or None
            The factor for each input variable. Elements 
            of None will be interpreted as unity.
            
        Notes
        -----
        If all elements are None, then 1 output variable
        (equal to one) will be assumed. 

        """
        
        nx = len(factors) # the number of factors 
        
        # Set the default parameters if all None
        nf = 1 
        maxderiv = None 
        zlevel = 0  # Constant function (i.e. unity)
        
        for i in range(nx):
            if factors[i] is not None: # There is some DFun presented
                nf = factors[i].nf 
                maxderiv = factors[i].maxderiv 
                zlevel = factors[i].zlevel
                break 
        
        # Now check all factors for agreement
        for i in range(nx):
            if factors[i] is not None:
                if factors[i].nx != 1:
                    raise ValueError("All factors must be 1-dimensional. This may"
                                     " change in the future.")
                if factors[i].nf != nf:
                    raise ValueError("Factors do not have the same nf")
                
                maxderiv = _min_maxderiv(maxderiv, factors[i].maxderiv)
                
                if factors[i].zlevel is not None and factors[i].zlevel < 0 :
                    zlevel = -1
                elif zlevel is not None and zlevel < 0 :
                    zlevel = -1 # remains zero 
                else:
                    zlevel = _sum_None(zlevel, factors[i].zlevel) 
                
        super().__init__(self._f_simple_prod, nf = nf, nx = nx,
                         maxderiv = maxderiv, zlevel = zlevel) 
        self.factors = factors 
    
    def _f_simple_prod(self, X, deriv = 0, out = None, var = None):
        #
        #
        # Calculate derivative array for a product of
        # independent functions
        #  f(x1) * g(x2) * h(x3) * ...
        #
        factors = self.factors # The factors f, g, h, ...
        
        nd, nvar = ndnvar(deriv, var, self.nx) 
        
        out, var = self._parse_out_var(X, deriv, out, var)
        
        dv = []
        for v in var:
            if factors[v] is None:
                # A unity factor
                d = np.zeros((deriv+1, self.nf) + X.shape[1:], dtype = X.dtype)
                d[0].fill(1.0) 
            else:
                # An explicit factor
                d = factors[v].f(X[v:(v+1)], deriv) # (deriv+1, nf, ...)
            dv.append(d)
        #
        # dv contains the 1-D derivatives of each factor in var respectively
        # We need to combine these derivatives, and then scalar multiply
        # by the values of non-var factors.
        #
        idxtab = adf.idxtab(deriv, nvar) 
        for i in range(nd):
            idx = idxtab[i]
            out[i,:] = 1.0 
            for j in range(nvar):
                out[i,:] *= dv[j][idx[j]] 
        
        for i in range(self.nx):
            if i not in var:
                if factors[i] is None:
                    pass # A factor of unity
                else:
                    val_i = factors[i].f(X[i:(i+1)], deriv = 0) # (1, nf, ...) 
                    out *= val_i # broadcast onto (nd, nf, ...) with (1, nf, ...)
        
        return out

class SelectedProduct(DFun):
    
    """
    Given multiple sets of functions, create
    a simple product using only selected combinatoins
    of factors.
    
    """

    def __init__(self, factors, select):
        """
        Parameters
        ----------
        factors : list of DFun
            The factor for each input variable.
        select : (n,len(factors)) array_like
            The factor function selection indices.
            
        Notes
        -----
        Only 1D factors are currently supported. This may change 
        in the future.

        """
        
        nx = len(factors) # assume 1-D factors. this may change in the future
        select = np.array(select).copy()  
        nf = select.shape[0] # the number of selection products
        if select.shape[1] != nx:
            raise ValueError("select.shape[1] must equal the number of factors")
        
        maxderiv = None 
        zlevel = 0  
        
        # Check all factors
        for i in range(nx):
            if factors[i].nx != 1:
                raise ValueError("All factors must be 1-dimensional. This may"
                                 " change in the future.")
            if select[:,i].max() > (factors[i].nf - 1):
                raise ValueError(f"factors[{i:d}] has too functions to select")
                
            maxderiv = _min_maxderiv(maxderiv, factors[i].maxderiv)
            
            if factors[i].zlevel is not None and factors[i].zlevel < 0 :
                zlevel = -1
            elif zlevel is not None and zlevel < 0 :
                zlevel = -1 # remains zero 
            else:
                zlevel = _sum_None(zlevel, factors[i].zlevel) 
                
        super().__init__(self._f_select_prod, nf = nf, nx = nx,
                         maxderiv = maxderiv, zlevel = zlevel) 
        
        self.factors = factors 
        self.select = select 
        
    def _f_select_prod(self, X, deriv = 0, out = None, var = None):
        #
        #
        # Calculate derivative array for a selected product of
        # independent functions
        #  f(x1) * g(x2) * h(x3) * ...
        #
        factors = self.factors # The factors f, g, h, ...
        select = self.select   # The selected products 
        
        nd, nvar = ndnvar(deriv, var, self.nx) 
        out, var = self._parse_out_var(X, deriv, out, var)
        
        dv = []
        for v in var:
            # (deriv+1, nf, ...)
            dv.append(factors[v].f(X[v:(v+1)], deriv))
            
        # dv contains the 1-D derivatives of each factor in var respectively
        # We need to combine these derivatives, and then scalar multiply
        # by the values of non-var factors.
        #
        idxtab = adf.idxtab(deriv, nvar) 
        
        out.fill(1.0) # initialize all derivative products to 1
        for i in range(nd):
            idx = idxtab[i]
            
            for k in range(self.nf):
                for j in range(nvar):
                    #
                    # For the j**th var, take the required
                    # derivative of its respective factor
                    #
                    out[i,k] *= dv[j][idx[j], select[k,var[j]]] 
        
        # Now scale by value of non-var variables
        for i in range(self.nx):
            if i not in var:
                val_i = factors[i].f(X[i:(i+1)], deriv = 0) # (1, nf, ...) 
                
                for k in range(self.nf):
                    out[:,k] *= val_i[:,select[k,i]]
                    # broadcast onto (nd, ...) with (1, ...)
        
        return out 
    
class PolyFactor(DFun):
    
    def __init__(self, terms, nx = None):
        """
        Parameters
        ----------
        terms : list of lists
            Each element of terms is a list corresponding
            to one term in a summation. Each term is itself
            a list of 2 elements. The first element is a 
            list of factors and the second is a scalar coefficient.
            For example, the term element [[0,0,1], 0.5] corresponds
            to x0*x0*x1 * 0.5.
        nx : int, optional
            The number of coordinates. If None (default), `nx` is assumed
            to the be maximum 
        """
        
        nterms = len(terms) # May be zero 
        
        max_degree = 0 
        nvar = 1 
        for i in range(nterms):
            term = terms[i] 
            factors = term[0]
            max_degree = max(len(factors), max_degree) 
            nvar = max(nvar, 1 + max(factors + [0]))
            
        if nx is None: # Default
            nx = nvar 
        
        
        super().__init__(self._fpoly, nf = 1, nx = nx, maxderiv = None,
                         zlevel = max_degree)
        self.terms = terms 
        
    def _fpoly(self, X, deriv = 0, out = None, var = None):
        
        x = X2adf(X, deriv, var) # x[0], x[1], x[2] are adf objects for each variable
        
        res = 0*x[0] 
        
        for term in self.terms :
            
            factors = term[0] # the list of factors (by variable label)
            coeff = term[1]   # the coefficient 
            
            if len(factors) == 0: # constant
                res += coeff # res <-- res + coeff
            else:
                temp = x[factors[0]] # temp <-- first factor 
                for i in range(1,len(factors)):
                    temp = temp * x[factors[i]] # product of all factors 
                
                res += coeff * temp 
        
        # Return result
        return adf2array([res],out)
        
class PolyPower(DFun):
    
    def __init__(self, terms):
        """
        Parameters
        ----------
        terms : list of lists
            Each element of terms is a list corresponding
            to one term in a summation. Each term is itself
            a list of 2 elements. The first element is a 
            list of integer powers and the second is a scalar coefficient.
            For example, the term element [[0,3,1], 0.5] corresponds
            to x0**0 * x1**3 * x2**1 * 0.5.
        """
        
        nterms = len(terms) # May not be zero 
        
        if nterms == 0:
            raise ValueError('At least one term must be included')
        
        max_degree = 0 
        
        for i in range(nterms):
            term = terms[i] 
            pows = term[0]
            max_degree = max(max(pows), max_degree)
    
        nx = len(terms[0][0]) 
        
        
        super().__init__(self._fpolypow, nf = 1, nx = nx, maxderiv = None,
                         zlevel = max_degree)
        self.terms = terms 
        
    def _fpolypow(self, X, deriv = 0, out = None, var = None):
        
        x = X2adf(X, deriv, var) # x[0], x[1], x[2] are adf objects for each variable
        
        # Pre-compute the powers of each variable 
        xpow = [] 
        for i in range(self.nx):
            xipow = [ adf.const_like(1.0, x[i])]
            for j in range(self.zlevel): # zlevel = max_degree 
                xipow.append(xipow[-1] * x[i]) 
            xpow.append(xipow)
        # xpow[i][j] = x[i] ** j 
            
        
        res = 0*x[0] 
        
        for term in self.terms :
            
            pows = term[0]    # the list of powers
            coeff = term[1]   # the coefficient 
            
            temp = adf.const_like(1.0, x[0]) 
            for i in range(len(pows)):
                temp = temp * xpow[i][pows[i]] 
            
            res += coeff * temp 
        
        # Return result
        return adf2array([res],out)   

class PowerExpansion(DFun):
    
    """
    Power series expansion about a given point. This is usually more efficient
    than similar functions :class:`PolyPower` and :class:`PolyFactor`
    because it uses derivative array translation instead of an explicit
    sum over terms.
    
    Attributes
    ----------
    d : (nd,nf) ndarray
        The defining derivative array about the expansion point
    x0 : (nx,) ndarray
        The expansion point. 
    
    """
    
    def __init__(self, d, x0):
        """
        Create a power series expansion 
        
        ..  math::
            
            f(\\mathbf{x}) = \\sum_\\alpha d^{(\\alpha)}(\\mathbf{x}-
                                                        \\mathbf{x}_0)^\\alpha
                
        Parameters
        ----------
        d : array_like
            The derivative array(s) at the expansion point.
        x0 : array_like
            The expansion point.
            
        """
        
        d = np.array(d) 
        
        if d.ndim == 1:
            d = d.reshape((-1,1))
        
        # d now has shape (nd, nf)
        
        nd = d.shape[0]
        nf = d.shape[1] 
        
        x0 = np.array(x0)
        nx = len(x0)
        
        # Figure out the order of the expansion
        #
        order = 0
        while True:
            if nd == nderiv(order, nx):
                # found
                break
            elif nd < nderiv(order, nx):
                raise ValueError('the shape of d is inconsistent with x0')
            else:
                order += 1 
        # order now equals the expansion order,
        # which is equal to the zlevel of the DFun
        
        
        super().__init__(self._fexpansion, nf = nf, nx = nx, maxderiv = None,
                         zlevel = order)
        
        self.d = d 
        self.x0 = x0 
        self.order = self.zlevel 
        self.idxD = adf.idxtab(self.order, self.nx)
        self.nckD = adf.ncktab(self.order + self.nx + 1) 
        
        return 
    
    def _fexpansion(self, X, deriv = 0, out = None, var = None):

        # Calculate the displacement from the 
        # expansion origin
        disp = np.empty_like(X)
        for i in range(self.nx):
            disp[i] = X[i] - self.x0[i] 
        
        # Now calculate the powers of the displacement
        # [power, variable, ...]
        disp_pow = np.ones((self.order+1,) + X.shape, dtype = X.dtype)
        for k in range(self.order):
            disp_pow[k+1] = disp_pow[k] * disp 

        out, var = self._parse_out_var(X, deriv, out, var)
        out.fill(0)
        
        nvar = len(var)
        idx = adf.idxtab(deriv, nvar) # Index table for output derivative array
        nd = nderiv(deriv, nvar)
        
        for iD in range(self.idxD.shape[0]):
            
            idxD = self.idxD[iD] # The index of the powers of d
        
            Dpow = np.ones(disp_pow.shape[2:], dtype = disp_pow.dtype)
            for i in range(self.nx):
                Dpow *= disp_pow[idxD[i], i]
                
            for iZ in range(nd):
                idxZ = idx[iZ,:] # The result index in the requested
                                 # `var` order
                
                idx_orig = np.arange(self.nx) * 0
                for vid in range(nvar):
                    idx_orig[var[vid]] = idxZ[vid]
                # idx_orig is the result index in the complete variable order
                
                idxX = idxD + idx_orig # The index of the expansion coefficient
                                       # that contributes to result
                kX = np.sum(idxX)
                if kX > self.order:
                    break # Skip remaining 
                
                # Calculate the multi-index
                # binomial coefficient
                c = 1.0 
                for i in range(self.nx):
                    c *= self.nckD[idxX[i], min(idx_orig[i], idxD[i])]
                    
                iX = adf.idxpos(idxX, self.nckD)
                
                out[iZ,:] += c * self.d[iX,:] * Dpow
        
        return out

def _min_maxderiv(maxA,maxB):
    return _composite_maxderiv(maxA,maxB)
def _max_zlevel(zlevelA,zlevelB):
    return _merged_zlevel(zlevelA, zlevelB)
def _product_None(a,b):
    if a is None or b is None:
        return None 
    else:
        return a * b
def _sum_None(a,b):
    if a is None or b is None:
        return None 
    else:
        return a + b

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

def _merged_maxderiv(maxA,maxB):
    """
    Calculate the maxderiv of a merged DFun.

    Parameters
    ----------
    maxA, maxB : int or None
        maxderiv of merged DFun.

    Returns
    -------
    int or None
        The maxderiv of the merger.

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

def _merged_zlevel(zlevelA, zlevelB):
    """
    Determine the zlevel of a merged
    DFun.

    Parameters
    ----------
    zlevelA, zlevelB : int or None
        zlevel parameter of merged DFuns.

    Returns
    -------
    int or None
        zlevel of merged function

    """
    
    # If zlevel is None, then the merged
    # outputs also have no zlevel.
    if zlevelA is None or zlevelB is None:
        zlevel = None
    else:
        # If neither is none, i.e. both
        # have some finite zlevel, then the
        # merged zlevel is the greater of the 
        # two.
        zlevel = max(zlevelA, zlevelB)

    
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

def sym2invdet(S, deriv, nvar, logdet = False):
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
    logdet : boolean, optional
        If True, the natural logarithm of the determinant
        is returned instead. The default is False.

    Returns
    -------
    iS : ndarray
        The derivative array of the matrix inverse of S
        in packed storage with shape (nd, nS, ...)
    det : ndarray
        The derivative array for det(S), with 
        shape (nd, ...). (This equals ln(det(S)) if
        `logdet` is True.)

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
    
    ############################################
    # 2) Compute the determinant
    #    This is the product of the squares of the diagonal entries
    if logdet == False: # Calculate the normal determinant (default)
        k = 0
        for i in range(N):
            if i == 0:
                det = Spacked[k] 
            else:
                det = det * Spacked[k]
            k = k + (i+2)
        det = det * det
        # det now equals the determinant of S
        
    else: # logdet == True, calculate the logarithm of the determinant instead 
        # 
        # The determinant is the product of the squares
        # of the diagonal entries. The logarithm of the 
        # determinant is thus twice the sum of
        # logarithms of the diagonal entries.
        k = 0
        for i in range(N):
            if i == 0:
                det = adf.log(Spacked[k])
            else:
                det = det + adf.log(Spacked[k])
            k = k + (i+2) 
        det = 2.0 * det 
    #
    ############################################
    
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
    
    # iS now contains the inverse of the original matrix
    # and det the determinant (or log of det)
    #
    return iS, det.d.copy()

