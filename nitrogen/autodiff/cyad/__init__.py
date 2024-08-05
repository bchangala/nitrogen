"""
nitrogen.autodiff.cyad
======================

Cython implementation of simple forward
differentiation. 

"""

import nitrogen.dfun
import numpy as np 
import nitrogen.autodiff.forward as adf 

class ForwardDFun(nitrogen.dfun.DFun):
    """
    Attributes
    ----------
    ny : integer
        The number of intermediate variables
    fun : function 
        The forward differentiation function 
    input_fun: DFun
        The input function.
        
    Notes
    -----
    
    The forward differentiation function is a (Python or Cython) function with 
    signature ``fun(double [:,:,:] dY, double [:,:,:] dF, int order, int [:,:] table)``
    with the parameters defined as follows,
    
    double [nd,ny,n] dY
        The derivatives of the `ny` intermediate inputs.
        
    double [nd,nf,n] dF
        The output buffer of the `nf` output values.
        
    int order 
        Derivative order 
        
    int [3,tablesize] table
        The derivative product table
        
    
    See Also
    --------
    nitrogen.autodiff.forward.calc_product_table
    
    """
    
    
    def __init__(self, fun, nf, ny, input_fun = None):
        """
        Create a ForwardDFun
        
        Parameters
        ----------
        
        fun : function
            The forward differentiation function.
        nf : integer
            The number of output values of `fun`. 
        ny : integer
            The number of input values of `fun`.
        input_fun : DFun, optional
            The input composition function, ny <-- nx. The default is None.

        """
        
        #
        # `fun` provides a forward diff. evaluation for a 
        # nf <-- ny function.
        #
        # input_fun is a DFun for ny <-- nx 
        #
        # if input_fun is None, then it should just
        # be identity, ny = nx <-- nx 
        #
        if input_fun is None:
            input_fun = nitrogen.dfun.IdentityDFun(ny)
        
        super().__init__(self._ffun, nf, input_fun.nx, 
                         input_fun.maxderiv, None)
        
        self.fun = fun 
        self.input_fun = input_fun 
        self.ny = ny # The number of intermediate variables 
    
    
    def _ffun(self, X, deriv = 0, out = None, var = None):
        
        # Evaluate the input function 
        Y = self.input_fun.f(X, deriv = deriv, var = var) # (nd, ny, ...) 
        
        nder,nvar = nitrogen.dfun.ndnvar(deriv, var, self.nx)
        
        nder = Y.shape[0] 
        base_shape = Y.shape[2:]

        # Reshape base shape to one dimension
        Y = np.reshape(Y, (nder, self.ny, -1))
        
        
        if out is None: 
            #
            # Create intermediate output, with 1D base shape
            #
            f = np.empty_like(Y, shape = (nder, self.nf, Y.shape[2]) )
        else:
            f = np.reshape(out, shape = (nder, self.f, Y.shape[2]))
        
        # Calculate the product table 
        table = adf.calc_product_table(deriv, nvar) 
        
        # Call the forward function 
        self.fun(Y, f, deriv, table)
        
        # Reshape output 
        F = np.reshape(f, (nder,1) + base_shape)
        # (`out` should still reference same data)
    
        return F