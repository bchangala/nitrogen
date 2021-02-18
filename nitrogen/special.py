"""
nitrogen.special
-------------

Special math and basis functions 
as differentiable DFun objects.

"""

import nitrogen.dfun as dfun
import numpy as np 


class SinCosDFun(dfun.DFun):
    
    """
    A real sine/cosine basis set
    
    f_m(phi) = 1/sqrt(pi) * sin(|m| * phi) ... m < 0
             = 1/sqrt(2*pi)                ... m = 0
             = 1/sqrt(pi) * cos( m  * phi) ... m > 0
             
    Attributes
    ----------
    
    m : ndarray
        The m-index of each basis function (i.e. each output value).
             
    """
    
    def __init__(self, m):
        """
        Create sine-cosine basis functions.

        Parameters
        ----------
        m : scalar or 1-D array_like
            If scalar, the 2|`m`| + 1 basis functions with
            index <= |`m`| will be included. If array_like,
            then `m` lists all m-indices to be included.

        """
        
        # Parse the `m` parameter
        # and generate a list of m-indices
        #
        if np.isscalar(m):
            m = np.arange(-abs(m), abs(m)+1)
        else:
            m = np.array(m)
        
        super().__init__(self._fphi, nf = len(m), nx = 1,
                         maxderiv = None, zlevel = None)
        
        self.m = m 
        
        return 
    
    
    def _fphi(self, X, deriv = 0, out = None, var = None):
        """ evaluation function """
        
        nd,nvar = dfun.ndnvar(deriv, var, self.nx)
        if out is None:
            base_shape = X.shape[1:]
            out = np.ndarray( (nd, self.nf) + base_shape, dtype = X.dtype)
            
        pi = np.pi
        phi = X 
        for k in range(nd):
            #
            # The k^th derivative order
            #
            for i in range(self.nf):
                m = self.m[i] # The m-index of the basis function
                if m < 0:
                    # 1/sqrt(pi) * sin(|m| * phi) 
                    if k % 4 == 0:
                        y = +abs(m)**k * np.sin(abs(m) * phi)
                    elif k % 4 == 1:
                        y = +abs(m)**k * np.cos(abs(m) * phi)
                    elif k % 4 == 2:
                        y = -abs(m)**k * np.sin(abs(m) * phi)
                    else: # k % 4 == 3
                        y = -abs(m)**k * np.cos(abs(m) * phi)
                    np.copyto(out[k:(k+1),i],y / np.sqrt(pi))
                              
                elif m == 0:
                    # 1/sqrt(2*pi)
                    if k == 0: # Zeroth derivative
                        out[0:1,i].fill(1.0 / np.sqrt(2*pi))
                    if k > 0 : # All higher derivatives
                        out[k:(k+1),i].fill(0.0)
                        
                elif m > 0 :
                    # 1/sqrt(pi) * cos(m * phi) 
                    if k % 4 == 0:
                        y = +m**k * np.cos(m * phi)
                    elif k % 4 == 1:
                        y = -m**k * np.sin(m * phi)
                    elif k % 4 == 2:
                        y = -m**k * np.cos(m * phi)
                    else: # k % 4 == 3
                        y = +m**k * np.sin(m * phi)
                    np.copyto(out[k:(k+1),i],y / np.sqrt(pi))        
        
        return out 
            