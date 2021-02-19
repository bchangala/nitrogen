"""
nitrogen.special
-------------

Special math and basis functions 
as differentiable DFun objects.

==============   ====================================================
Function         Description
--------------   ----------------------------------------------------
SinCosDFun       Real sine-cosine basis
LegendreLMCos    Associated Legendre functions with cosine argument

"""

import nitrogen.dfun as dfun
import nitrogen.autodiff.forward as adf
import numpy as np 
from scipy.special import factorial, factorial2


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


class LegendreLMCos(dfun.DFun):  
    """
    Associated Legendre polynomials of a 
    given order, m, with 
    cos(theta) argument for theta in [0,pi]
    
    F_lm(theta) = N^m_l * P_l,|m|(cos(theta)),
    m = 0, 1, 2, 3, ...
    l = |m|, |m|+1,|m|+2,..., lmax
    (negative m is defined, but just equal to m = |m|)
    
    where the normalization coefficient, N^m_l, is
      
      N^m_l = [2/(2*l+1)  * (l+|m|)!/(l-|m|)!] ** (-1/2)
    
    
    Attributes
    ----------
    
    l : ndarray
        A list of the l-index of each basis function
    m : int 
        The associated Legendre function order index.
    """
    
    def __init__(self, m, lmax):
        """
        Create associated Legendre basis DFuns

        Parameters
        ----------
        m : int
            The Legendre order.
        lmax : int
            The maximum value of l. This must
            be greater than or equal to |m|.
        """
        
        if lmax < abs(m):
            return ValueError("lmax must be >= |m|")
        
        l = np.arange(abs(m), lmax+1)
        
        super().__init__(self._ftheta, nf = len(l), nx = 1,
                         maxderiv = None, zlevel = None)
        
        self.l = l
        self.m = m 
        
        return 
        
    def _ftheta(self, X, deriv = 0, out = None, var = None):
        """
        basis evaluation function 
        
        Use recursion relations for associated Legendre
        polynomials.
    
        """    
        nd,nvar = dfun.ndnvar(deriv, var, self.nx)
        if out is None:
            base_shape = X.shape[1:]
            out = np.ndarray( (nd, self.nf) + base_shape, dtype = X.dtype)
        
        # Create adf object for theta
        theta = dfun.X2adf(X, deriv, var)[0]
        m = abs(self.m) # |m| 
        
        F = []
        # Calculate Legendre polynomials 
        #
        # Start with F_l=|m|, m
        #  = N * P_l,l     with l = |m|
        #  = N * (-1)**l * (2l-1)!! * sin(theta)**l
        sinm = adf.powi(adf.sin(theta), m)
        Nmm = (2/(2*m+1) * factorial(2*m)) ** (-0.5)
        c = (Nmm * (-1)**m * factorial2(2*m-1))
        Fmm = c * sinm
        F.append(Fmm)
        cos = adf.cos(theta)
        #
        if self.nf > 1:
            # The second function is related to the first via
            # 
            #                N^|m|_|m|+1
            #  F^|m|_|m|+1 = ------------ cos(theta) * (2|m|+1) F^|m|_|m|
            #                 N^|m|,|m|
            #
            # The ratio of N's is just
            # sqrt(2|m|+3) / (2|m| + 1)
            #
            Fmmp1 = np.sqrt(2*m+3) * (cos * Fmm) 
            F.append(Fmmp1)
        for l in self.l[2:]:
            # 
            # Continue with general recursion relation
            #
            #           2l-1             N^m_l                 l+m-1 N^m_l
            # F^m_l =  ------ cos(theta) -------- F^m_l-1   -  ----- ------- F^m_l-2
            #           l - m            N^m_l-1               l-m   N^m_l-2
            #
            # first N ratio:
            N1 = np.sqrt((2*l+1)/(2*l-1) * (l-m)/(l+m))
            # second N ratio: 
            N2 = np.sqrt((2*l+1)/(2*l-3) * ((l-m)/(l+m)) * ((l-m-1)/(l+m-1)))
            Fl = ((2*l-1) / (l-m) * N1) * (cos * F[-1]) - ((l+m-1)/(l-m)*N2) * F[-2]
            
            F.append(Fl)
            
        # Convert adf objects to a single
        # derivative array
        dfun.adf2array(F, out)
        
        return out 
    
class Sin(dfun.DFun):
    
    """
    The function sin(phi). 
    (Used for Legendre Basis weight function)
             
    """
    
    def __init__(self):
        """
        """
        super().__init__(self._sin, nf = 1, nx = 1,
                         maxderiv = None, zlevel = None)
        return 
    
    def _sin(self, X, deriv = 0, out = None, var = None):
        """ evaluation function """
        
        nd,nvar = dfun.ndnvar(deriv, var, self.nx)
        if out is None:
            base_shape = X.shape[1:]
            out = np.ndarray( (nd, self.nf) + base_shape, dtype = X.dtype)
            
        phi = X 
        for k in range(nd):
            #
            # The k^th derivative order
            #
            if k % 4 == 0: 
                y =  np.sin(phi)
            elif k % 4 == 1:
                y =  np.cos(phi)
            elif k % 4 == 2:
                y = -np.sin(phi)
            else: # k % 4 == 3
                y = -np.cos(phi)   
                
            np.copyto(out[k,0:1], y)
            
        return out 