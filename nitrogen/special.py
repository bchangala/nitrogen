"""
nitrogen.special
----------------

Special math and basis functions 
as differentiable DFun objects.

==============   ====================================================
Function         Description
--------------   ----------------------------------------------------
SinCosDFun       Real sine-cosine basis
LegendreLMCos    Associated Legendre functions with cosine argument
==============   ====================================================

"""

import nitrogen.dfun as dfun
import nitrogen.autodiff.forward as adf
import numpy as np 
from scipy.special import factorial, factorial2


class SinCosDFun(dfun.DFun):
    
    """
    A real sine/cosine basis set,
    
    .. math::

       f_m(\\phi) &= 1/\sqrt{\pi} \sin(|m|\phi) &\ldots m < 0\\\\
       &= 1/\sqrt{2\pi}              &\ldots m = 0\\\\
       &= 1/\sqrt{\pi} \cos(m \phi)  &\ldots m > 0 
             
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
            If scalar, the :math:`2|m|+1` basis functions with
            index :math:`\leq |m|` will be included. If array_like,
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
    given order, :math:`m`, with 
    :math:`\cos \\theta` argument for 
    :math:`\\theta \in [0,\pi]`.
    
    .. math::
       F_\ell^m(\\theta) = N^m_l  P_l^{|m|}(\cos \\theta),
       
    with :math:`m = 0, 1, 2, \ldots` and 
    :math:`\ell = |m|, |m|+1, \ldots, \ell_\\text{max}`.
    (Negative :math:`m` is defined, but just equal to :math:`m = |m|`).
    
    The normalization coefficient is
    
    .. math::
        
       N_\ell^m = \\left( \\frac{2}{2\ell+1}  \\frac{(l+|m|)!}{(l-|m|)!} \\right) ^{-1/2}
    
    
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
            The maximum value of :math:`\ell`. This must
            be greater than or equal to :math:`|m|`.
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
        sinm = adf.powi(adf.sin(theta), m)
        cos = adf.cos(theta)
        
        # Calculate Legendre polynomials 
        # with generic algebra
        F = _leg_gen(sinm, cos, m, np.max(self.l))
            
        # Convert adf objects to a single
        # derivative array
        dfun.adf2array(F, out)
        
        return out 
    
def _leg_gen(sinm,cos,m,lmax):
    
    """ Calculate associated Legendre polynomials
    of order m for abs(m) <= l <= lmax with generic algebra.
    
    Parameters
    ----------
    
    sinm : ndarray, adarray, or other algebraic object
        sin(theta)**abs(m)
    cos : ndarray, adarray, or other algebraic object
        cos(theta)
    m : int
        The Legendre order.
    lmax : int
        The maximum :math:`\ell` index.
        
    Returns
    -------
    F : list
        The associated Legendre polynomials as the 
        result type of `sinm` and `cos`.
    
    """
    F = []
    
    m = abs(m) 
    # Start with F_l=|m|, m
    #  = N * P_l,l     with l = |m|
    #  = N * (-1)**l * (2l-1)!! * sin(theta)**l
    Nmm = (2/(2*m+1) * factorial(2*m)) ** (-0.5)
    c = (Nmm * (-1)**m * factorial2(2*m-1))
    Fmm = c * sinm
    if lmax >= m:
        F.append(Fmm)
    
    #
    if lmax >= m + 1:
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
    for l in range(m+2, lmax+1):
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
        
    return F 
    
class Sin(dfun.DFun):
    
    """
    The function sin(theta). 
    (Used for Legendre Basis weight function)
             
    """
    
    def __init__(self, nx = 1):
        """
        
        Parameters
        ----------
        nx : int, optional
            The total number of input variables. The
            argument of sine is always the first.
            The rest are dummies. The default is 1.
            
        """
        super().__init__(self._sin, nf = 1, nx = nx,
                         maxderiv = None, zlevel = None)
        return 
    
    def _sin(self, X, deriv = 0, out = None, var = None):
        """ evaluation function """
        
        nd,nvar = dfun.ndnvar(deriv, var, self.nx)
        if out is None:
            base_shape = X.shape[1:]
            out = np.ndarray( (nd, self.nf) + base_shape, dtype = X.dtype)
            
        theta = X[0]
        for k in range(nd):
            #
            # The k^th derivative order
            #
            if k % 4 == 0: 
                y =  np.sin(theta)
            elif k % 4 == 1:
                y =  np.cos(theta)
            elif k % 4 == 2:
                y = -np.sin(theta)
            else: # k % 4 == 3
                y = -np.cos(theta)   
                
            np.copyto(out[k,0:1], y)
            
        return out 
    
class RealSphericalH(dfun.DFun):
    
    """
    Real-valued spherical harmonics,
    
    .. math::
       \Phi_\ell^m(\\theta, \\phi) = F_\ell^m(\\theta) f_m (\\phi)
    
    For definitions of the associated Legendre polynomials
    :math:`F_\ell^m` and sine-cosine functions :math:`f_m`, see
    :class:`~nitrogen.special.LegendreLMCos` and 
    :class:`~nitrogen.special.SinCosDFun`.
    
    Attributes
    ----------
    l : ndarray
        The :math:`\ell` quantum numbers.
    m : ndarray
        The :math:`m` quantum numbers.
    
    """
    
    def __init__(self, m, lmax):
        """
        Create a real spherical harmonic basis.
        
        Parameters
        ----------
        m : scalar or 1-D array_like
            The projection quantum number. 
            If scalar, then all :math:`m` with 
            :math:`|m| \leq` `m` will be included. If array_like,
            then `m` lists all :math:`m` to be included.
        lmax : int
            The maximum value of :math:`\ell`, the angular momentum
            (or azimuthal) quantum number.

        """
        
        # Parse m-list
        if np.isscalar(m):
            m = np.arange(-abs(m), abs(m)+1)
        else:
            m = np.array(m)
        
        # Require every m quantum number to have at least
        # one basis function, i.e. lmax must be >= max(abs(m))
        if lmax < np.max(abs(m)):
            raise ValueError("At least one m-order is empty. Increase lmax")
        
        mlist = []
        llist = []
        for M in m:
            for L in range(abs(M),lmax+1):
                mlist.append(M)
                llist.append(L) 
        
        nf = len(mlist) # = len(llist)
        if nf == 0:
            raise ValueError("Invalid m or l quantum number constraints.")
        
        
        super().__init__(self._Philm, nf = nf, nx = 2,
                         maxderiv = None, zlevel = None)
        
        # Save the quantum number lists 
        # as ndarray's
        self.l = np.array(llist) 
        self.m = np.array(mlist) 
        
        # Set up DFun's for the theta basis and the phi 
        # basis separable factors 
        self.phi_basis = SinCosDFun(m) # only supplies functions for index M in `m`
        theta_bases = []
        for M in m: # Legendre basis for each M in `m`
            theta_bases.append(LegendreLMCos(M,lmax))
        self.theta_bases = theta_bases 
        
        
        return 
    
    def _Philm(self, X, deriv = 0, out = None, var = None):
        
        # Setup
        nd,nvar = dfun.ndnvar(deriv, var, self.nx)
        if out is None:
            base_shape = X.shape[1:]
            out = np.ndarray( (nd, self.nf) + base_shape, dtype = X.dtype)
        
        # Make adf objects for theta and phi 
        x = dfun.X2adf(X, deriv, var)
        theta = x[0]
        phi = x[1] 
        
        #########################################
        #
        # Calculate F and f factors first
        #
        sinth = adf.sin(theta)
        costh = adf.cos(theta)
        lmax = np.max(self.l)
        # Calculate the associated Legendre factors
        Flm = [] 
        amuni = np.unique(abs(self.m)) # list of unique |m|
        for aM in amuni:
            # Order aM = |M|
            sinM = adf.powi(sinth, aM)
            # Calculate F^M up to l <= lmax
            Flm.append(_leg_gen(sinM, costh, aM, lmax))
        #Flm is now a nested list 
        
        # Calculate the phi factors 
        smuni = np.unique(self.m) # list of unique signed m
        fm = []
        for sM in smuni:
            if sM == 0:
                fm.append(adf.const_like(1/np.sqrt(2*np.pi), phi))
            elif sM < 0:
                fm.append(1/np.sqrt(np.pi) * adf.sin(abs(sM) * phi))
            else: #sM > 0
                fm.append(1/np.sqrt(np.pi) * adf.cos(sM * phi))
        #
        ##############################################
        
        ###########################################
        # Now calculate the list of real spherical harmonics 
        Phi = []
        for i in range(self.nf):
            l = self.l[i]
            m = self.m[i] # signed m
            
            # Gather the associated Legendre polynomial (theta factor)
            aM_idx = np.where(amuni == abs(m))[0][0]
            l_idx = l - abs(m)
            F_i = Flm[aM_idx][l_idx] 
            # Gather the sine/cosine (phi factor)
            sM_idx = np.where(smuni == m)[0][0]
            f_i = fm[sM_idx]
            
            # Add their product to the list of functions
            Phi.append(F_i * f_i) 
        #
        ###########################################
            
        # Convert the list to a single 
        # DFun derivative array
        dfun.adf2array(Phi, out)
        # and return
        return out 

    
    