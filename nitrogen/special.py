"""
nitrogen.special
----------------

Special math and basis functions 
as differentiable DFun objects.

==============   ====================================================
Function         Description
--------------   ----------------------------------------------------
SinCosDFun       Real sine-cosine basis.
LegendreLMCos    Associated Legendre functions with cosine argument.
RealSphericalH   Real spherical harmonics.
LaguerreL        Generalized Laguerre polynomials, :math:`L^{(\\alpha)}_n(x)`.
RadialHO         Radial harmonic oscillator eigenfunctions in :math:`d` dimensions.
BesselJ          Bessel functions of the first kind, :math:`J_{\\nu}(x)`.
Real2DHO         Real 2-D isotropic harmonic oscillator wavefunctions.
==============   ====================================================

"""

import nitrogen.dfun as dfun
import nitrogen.autodiff.forward as adf
import numpy as np 
import scipy.special 
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
    These are normalized.
    
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
    
class Monomial(dfun.DFun):
    
    """
    The monominal in multiple variables
             
    """
    
    def __init__(self, pows):
        """
        
        Parameters
        ----------
        pows : array_like
            A list of non-negative integer exponents.
            
        """
        pows = np.array(pows)
        nx = len(pows)
        maxpow = np.max(pows) # The maximum
        if np.any(pows < 0):
            raise ValueError("All elements of pows must be non-negative integers.")
        
        super().__init__(self._monomial, nf = 1, nx = nx,
                         maxderiv = None, zlevel = maxpow)
        
        self.pows = pows 
        return 
    
    def _monomial(self, X, deriv = 0, out = None, var = None):
        """ evaluation function """
        raise NotImplementedError("NONO")
        nd,nvar = dfun.ndnvar(deriv, var, self.nx)
        if out is None:
            base_shape = X.shape[1:]
            out = np.ndarray( (nd, self.nf) + base_shape, dtype = X.dtype)
            
        x = dfun.X2adf(X, deriv, var)
        res = adf.powi(x[0], self.pows[0]) 
        for i in range(1, self.nx):
            res = res * adf.powi(x[i], self.pows[i])
            
        dfun.adf2array(res, out)
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

    
class BesselJ(dfun.DFun):
    """
    
    Bessel functions of the first kind.
    
    """
    
    def __init__(self, v):
        """
        Bessel function of the first kind,
        :math:`J_{\\nu}(x)`.

        Parameters
        ----------
        v : float
            The real order parameter.

        """
        
        super().__init__(self._jv, nf = 1, nx = 1, 
                         maxderiv = None, zlevel = None)
        
        self.v = v 
        
        return 
        
    def _jv(self, X, deriv = 0, out = None, var = None):
        """
        Bessel function of the first kind, Jv.
        Derivative array eval function.
        """
        # Setup
        nd,nvar = dfun.ndnvar(deriv, var, self.nx)
        if out is None:
            base_shape = X.shape[1:]
            out = np.ndarray( (nd, self.nf) + base_shape, dtype = X.dtype)
        
        # Make adf objects for theta and phi 
        x = X[0] 
        
        #
        # Calculate derivatives
        #
        # For now, we will just use the SciPy
        # Bessel function derivatives routine for 
        # each deriative order. This routine
        # makes use of the recursive definition of 
        # derivatives of Jv 
        #
        # (d/dx)^k J_v = (1/2)**k  * Sum_n=0^k  (-1)**n * (k choose n) * J_{v-k+2n}
        #
        # By calling it for each derivative order, 
        # we are calculating some Bessel functions
        # multiple times. But this is not that expensive anyway,
        # so don't worry about the efficiency issue.
        #
        for k in range(nd):
            dk = scipy.special.jvp(self.v, x, n = k)
            np.copyto(out[k:(k+1),0], dk)
        
        return out 
    
    
def _laguerre_gen(x, alpha, nmax):
    
    """ Calculate generalized Laguerre polynomials
    of order `alpha` up to degree `nmax` with generic algebra,
    
    .. math:
       L^{(\\alpha)}_n(x)
      
    
    
    Parameters
    ----------
    x : ndarray, adarray, or other algebraic object
        The argument of the Laguerre polynomials
    alpha : float
        The order parameter.
    nmax : int
        The maximum degree.
        
    Returns
    -------
    L : list
        The generalized Laguerre polynomials as the 
        same type of `x`.
    
    """
    L = []
    
    if nmax >= 0:
        L0 = 1.0 + 0.0 * x # L_0 = 1, always
        L.append(L0)
    
    if nmax >= 1:
        L1 = 1.0 + alpha - x # L_1 = 1 + alpha - x, always
        L.append(L1)
        
    for n in range(2,nmax+1):
        #
        # Use recurrence relation
        #
        # L^a_n = (2n-1+a-x)L^a_{n-1} - (n-1+a)L^a_{n-2}
        #         -------------------------------------
        #                            n
        #
        
        f1 = (2*n - 1 + alpha - x) * L[-1]
        f2 = (n - 1 + alpha) * L[-2] 
        L.append((f1-f2) / n)
        
    return L 

class LaguerreL(dfun.DFun):  
    """
    Generalized Laguerre polynomials of a 
    given order, :math:`\\alpha`.
    
    .. math::
       L_n^{(\\alpha)}(x)
    
    Attributes
    ----------
    
    nmax : int
        The maximum degree.
    alpha : float 
        The associated Legendre function order index.
    """
    
    def __init__(self, alpha, nmax):
        """
        Create associated Legendre basis DFuns

        Parameters
        ----------
        m : float
            The real order parameter.
        nmax : int
            The maximum degree.
        """
        
        if nmax < 0:
            return ValueError("nmax must be >= 0")
         
        super().__init__(self._Lx, nf = nmax + 1, nx = 1,
                         maxderiv = None, zlevel = None)
        #
        # The zlevel is actually finite, but 
        # I will ignore that here.
        #
        self.nmax = nmax 
        self.alpha = alpha 
        
        return 
        
    def _Lx(self, X, deriv = 0, out = None, var = None):
        """
        basis evaluation function 
        
        Use recursion relations for generalized 
        Laguerre polynomials
    
        """    
        nd,nvar = dfun.ndnvar(deriv, var, self.nx)
        if out is None:
            base_shape = X.shape[1:]
            out = np.ndarray( (nd, self.nf) + base_shape, dtype = X.dtype)
        
        # Create adf object for theta
        x = dfun.X2adf(X,deriv,var)[0]
        
        # Calculate Laguerre polynomials
        # with generic algebra
        L = _laguerre_gen(x, self.alpha, self.nmax)
            
        # Convert adf objects to a single
        # derivative array
        dfun.adf2array(L, out)
        
        return out 
    
    
def _radialHO_gen(r, expar2, nmax, ell, d, alpha):
    
    """ Calculate radial eigenfunctions of the d-dimensional
    isotropic harmonic oscillator for generalized angular 
    momentum quantum number :math:`\ell`.
    
    .. math:
       R^{(\ell)}_n(r) = (-1)^n \\alpha^{d/4} \left[ \\frac{2 \Gamma(n+1) }{Gamma(n+\ell + d/2)} \\right]^{1/2} e^{-\alpha r^2/2} (\alpha^{1/2} r)^\ell L_n^{(\ell + d/2 - 1)}(\alpha r^2)
       
    
    
    Parameters
    ----------
    r : ndarray, adarray, or other algebraic object
        The argument of the radial function
    expar2 : ndarray, adarray or other algebraic object
        The expression :math:`e^{-\alpha r^2 / 2}`.
    nmax: int
        The maximum degree
    ell : int
        The angular momentum quantum number: 0, 1, 2, ...
    d : int 
        The dimensionality, d >= 2.
    alpha : float
        The radial scaling parameter. This has units
        of inverse-length-squared.
        
    Returns
    -------
    R : list
        The (nmax+1) radial eigenfunctions.
        
    Notes
    -----
    The standard total vibrational quantum number of the d-dimensional
    HO is :math:`v = 2n + \ell`. 
    
    """
    
    rell = 1.0 + 0.0*r 
    for i in range(ell): 
        rell = rell * r  
    # rell = r^ell
    
    r2 = r * r # r^2
    x = alpha * r2 # The Laguerre argument
    
    R = []
    
    if d < 2:
        raise ValueError("d must be >= 2")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
        
    c0 =  (alpha)**(d/4 + ell/2) * np.sqrt(2 / scipy.special.gamma(  ell+d/2))
    c1 = -(alpha)**(d/4 + ell/2) * np.sqrt(2 / scipy.special.gamma(1+ell+d/2))
    
    nu = ell + d/2 - 1 # The Laguerre order 
    
    if nmax >= 0:
        R0 = c0 * expar2 * rell
        R.append(R0)
    
    if nmax >= 1:
        R1 = c1 * expar2 * rell * (1 + nu - x)  
        R.append(R1)
        
    for n in range(2,nmax+1):
        #
        # Use recurrence relation
        #
        # R_n =  -(2n-1+nu-x) * sqrt(n/(n+nu)) * R_{n-1} - (n-1+nu) * sqrt(n(n-1)/((n+nu)(n+nu-1))) * R_{n-2}
        #        -------------------------------------------------------------------------------------------
        #                                       n
        #
        f1 = -np.sqrt(n/(n+nu)) * (2*n - 1 + nu - x) * R[-1]
        f2 = -np.sqrt(n*(n-1)/((n+nu)*(n+nu-1))) * (n - 1 + nu) * R[-2]
        
        R.append((f1 + f2) / n)
        
    return R

class RadialHO(dfun.DFun):  
    """
    Radial eigenfunctions for a :math:`d`-dimensional isotropic
    harmonic oscillator.
    
    .. math::
       R_n^{(\ell)}(r) = (-1)^n \\alpha^{d/4} \\left[ \\frac{2 \\Gamma(n+1) }
           {\\Gamma(n+\ell + d/2)} \\right]
           ^{1/2} e^{-\\alpha r^2/2} (\\alpha^{1/2} r)^\ell L_n^{(\ell + d/2 - 1)}
           (\\alpha r^2)
    
    Attributes
    ----------
    nmax : int
        The maximum Laguerre index.
    ell : int
        The generalized angular momentum quantum number.
    d : int 
        The dimensionality
    alpha : float
        The radial scaling parameter :math:`\\alpha` with units
        inverse-length-squared.
    
    Notes
    -----
    These wavefunctions are orthonormal with respect to an integration
    volume element of :math:`r^{d-1} dr` over :math:`r = [0,\\infty)`. An
    isotropic harmonic oscillator of mass :math:`m` and frequency :math:`\\omega`
    has :math:`\\alpha = m \\omega / \\hbar`. The conventional vibrational
    quantum number :math:`v` is related to the Laguerre polynomial degree 
    parameter as :math:`v = 2n + \\ell`, and the energy eigenvalue is
    :math:`E /\\hbar \\omega = v + d/2 = 2n + \\ell + d/2`.
    
    For a given :math:`\ell` and :math:`d`, the matrix elements of 
    :math:`r^2` are tri-diagonal,
    
    .. math::
       \\langle n \\vert  r^2 \\vert n \\rangle &= \\alpha^{-1}(2n + \\ell + d/2) \\\\
       \\langle n+1 \\vert r^2 \\vert n\\rangle = 
           \\langle n \\vert r^2 \\vert n + 1 \\rangle &= \\alpha^{-1}\\sqrt{(n+1)(n+\\ell + d/2)}.
       
    The differential operator,
    
    .. math::
       \\hat{D}^2 \\equiv \\partial_r^2 + \\frac{d-1}{r} \\partial_r -
           \\frac{\\ell(\\ell + d - 2)}{r^2},
          
    is also tri-diagonal with matrix elements,
    
    .. math::
       \\langle n \\vert  \hat{D}^2 \\vert n \\rangle &= -\\alpha(2n + \\ell + d/2) \\\\
       \\langle n+1 \\vert \hat{D}^2\\vert n\\rangle = 
           \\langle n \\vert \hat{D}^2 \\vert n + 1 \\rangle &= \\alpha\\sqrt{(n+1)(n+\\ell + d/2)}.
       
    By inspection, we can now see that the Hamiltonian operator
    
    .. math::
       \\hat{H}/\\hbar\\omega = -\\frac{1}{2}\\alpha^{-1} \\hat{D}^2 + \\frac{1}{2}\\alpha r^2
       
    is diagonal, with eigenvalue :math:`2n + \\ell + d/2 = v + d/2`.


    """
    
    def __init__(self, nmax, ell, d = 2, alpha = 1.0):
        """
        Create radial harmonic oscillator wavefunctions.

        Parameters
        ----------
        nmax : int
            The maximum Laguerre index.
        ell : int
            The generalized angular momentum quantum number.
        d : int, optional
            The dimensionality. The default is 2.
        alpha : float, optional
            The radial scaling parameter, :math:`\\alpha`, with units
            inverse-length-squared. The default is 1.
        """
        
        if nmax < 0:
            return ValueError("nmax must be >= 0")
         
        super().__init__(self._Rr, nf = nmax + 1, nx = 1,
                         maxderiv = None, zlevel = None)
        #
        #
        self.nmax = nmax 
        self.ell = ell
        self.d = d
        self.alpha = alpha 
        
        return 
        
    def _Rr(self, X, deriv = 0, out = None, var = None):
        """
        basis evaluation function 
        
        Use generic algebra evaluation function
    
        """    
        nd,nvar = dfun.ndnvar(deriv, var, self.nx)
        if out is None:
            base_shape = X.shape[1:]
            out = np.ndarray( (nd, self.nf) + base_shape, dtype = X.dtype)
        
        # Create adf object for theta
        r = dfun.X2adf(X,deriv,var)[0]
        expar2 = adf.exp(-0.5 * self.alpha * (r*r))
        
        # Calculate radial wavefunctions
        R = _radialHO_gen(r, expar2, self.nmax, self.ell, self.d, self.alpha)
 
        # Convert adf objects to a single
        # derivative array
        dfun.adf2array(R, out)
        
        return out 
    
class Real2DHO(dfun.DFun):
    
    """
    Real-valued eigenfunctions of 2-D 
    isotropic harmonic oscillator in
    cylindrical coordinates.
    
    .. math::
       \chi_n^{\\ell}(r, \\phi) = R_n^{|\\ell|}(r) f_{\\ell} (\\phi)
    
    For definitions of the radial wavefunctions
    :math:`R_n^{\\ell}` and sine-cosine functions :math:`f_{\\ell}`, see
    :class:`~nitrogen.special.RadialHO` and 
    :class:`~nitrogen.special.SinCosDFun`.
    
    Attributes
    ----------
    v : ndarray
        The :math:`v` quantum numbers, where
        :math:`v = 2n + \\ell`.
    ell : ndarray
        The :math:`\\ell` quantum numbers.
    n : ndarray
        The :math:`n` Laguerre degree.
    vmax : int
        The initial `vmax` parameter.
    alpha : float
        The radial scaling parameter, :math:`\\alpha`.
    
    """
    
    def __init__(self, vmax, alpha, ell = None):
        """
        Create a 2D harmonic oscillator basis.
        
        Parameters
        ----------
        vmax : int 
            The maximum vibrational quantum number :math:`v`, 
            in the conventional sum-of-modes sense.
        alpha : float
            The radial scaling parameter, :math:`\\alpha`.
        ell : scalar or 1-D array_like, optional
            The angular momentum quantum number.
            If scalar, then all :math:`\\ell` with 
            :math:`|\\ell| \leq` abs(`ell`) will be included. If array_like,
            then `ell` lists all (signed) :math:`\\ell` values to be included.
            A value of None is equivalent to `ell` = `vmax`. The default is 
            None.

        """
        
        if vmax < 0 :
            return ValueError("vmax must be >= 0")
        
        # Parse ell list
        if ell is None:
            ell = np.arange(-vmax, vmax+1)
        elif np.isscalar(ell):
            ell = np.arange(-abs(ell), abs(ell) + 1)
        else:
            ell = np.array(ell)
            
        if vmax < np.max(abs(ell)):
            print(f"Warning: values of ell above vmax = {vmax:d} will have no basis functions.")
            
        ell_list = []
        v_list = []
        for ELL in ell:
            for V in range(abs(ELL), vmax+1, 2):
                ell_list.append(ELL)
                v_list.append(V) 
        
        nf = len(ell_list) # = len(v_list)
        if nf == 0:
            raise ValueError("Zero basis functions!")
        
        
        super().__init__(self._chinell, nf = nf, nx = 2,
                         maxderiv = None, zlevel = None)
        
        # Save the quantum number lists 
        # as ndarray's
        self.ell = np.array(ell_list) 
        self.v = np.array(v_list) 
        self.n = (self.v - abs(self.ell))//2
        self.vmax = vmax 
        
        # Save the alpha scaling parameter
        self.alpha = alpha 
        
        # # Set up DFun's for the radial basis and the 
        # # polar angular basis as separable factors 
        # self.phi_basis = SinCosDFun(ell) # only supplies functions for index in `ell`
        # radial_bases = []
        # for ELL in ell: # Radial HO basis for each ELL in `ell`
        #     nmax = round(np.floor(vmax-abs(ELL))/2)
        #     radial_bases.append(RadialHO(nmax, ELL, d = 2, alpha = self.alpha))
        # self.radial_bases = radial_bases
        
        
        return 
    
    def _chinell(self, X, deriv = 0, out = None, var = None):
        
        # Setup
        nd,nvar = dfun.ndnvar(deriv, var, self.nx)
        if out is None:
            base_shape = X.shape[1:]
            out = np.ndarray( (nd, self.nf) + base_shape, dtype = X.dtype)
        
        # Make adf objects for theta and phi 
        x = dfun.X2adf(X, deriv, var)
        r = x[0]        # Cylindrical radius
        phi = x[1]      # Angular coordinate
        
        #########################################
        #
        # Calculate R and f factors first
        #        
        # Calculate the radial wavefunctions
        Rnell = [] 
        abs_ell_uni = np.unique(abs(self.ell)) # list of unique |ell|
        expar2 = adf.exp(-0.5 * self.alpha * (r*r))  # The exponential radial factor
        for aELL in abs_ell_uni:
            nmax = round(np.floor(self.vmax - aELL) / 2)
            Rnell.append(_radialHO_gen(r, expar2, nmax, aELL, 2, self.alpha))
        
        
        # Calculate the phi factors, f_ell(phi)
        sig_ell_uni = np.unique(self.ell) # list of unique signed ell
        fell = []
        for sELL in sig_ell_uni:
            if sELL == 0:
                fell.append(adf.const_like(1/np.sqrt(2*np.pi), phi))
            elif sELL < 0:
                fell.append(1/np.sqrt(np.pi) * adf.sin(abs(sELL) * phi))
            else: #sELL > 0
                fell.append(1/np.sqrt(np.pi) * adf.cos(sELL * phi))
        #
        ##############################################
        
        ###########################################
        # Now calculate the list of real 2-D HO wavefunctions
        chi = []
        for i in range(self.nf):
            
            ell = self.ell[i] 
            n = self.n[i] 
            
            # Gather the radial factor
            abs_ELL_idx = np.where(abs_ell_uni == abs(ell))[0][0]
            R_i = Rnell[abs_ELL_idx][n] 
            
            # Gather the angular factor
            sig_ELL_idx = np.where(sig_ell_uni == ell)[0][0]
            f_i = fell[sig_ELL_idx]
            
            # Add their product to the list of functions
            chi.append(R_i * f_i)
        #
        ###########################################
            
        # Convert the list to a single 
        # DFun derivative array
        dfun.adf2array(chi, out)
        # and return
        return out 
