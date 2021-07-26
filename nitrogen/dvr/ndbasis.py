"""
ndbasis.py

Implements the NDBasis base class and common
sub-classes.

=============================================  =================================
:class:`~nitrogen.dvr.NDBasis` sub-class       Description
---------------------------------------------  --------------------------------- 
:class:`~nitrogen.dvr.SinCosBasis`             A sine-cosine (real Fourier) basis
:class:`~nitrogen.dvr.LegendreLMCosBasis`      Associated Legendre polynomials.
:class:`~nitrogen.dvr.RealSphericalHBasis`     Real spherical harmonics.
:class:`~nitrogen.dvr.Real2DHOBasis`           Two-dimensional harmonic osc. basis.
:class:`~nitrogen.dvr.RadialHOBasis`           Radial HO basis in d dimensions.
=============================================  =================================

"""

import nitrogen.special as special
import scipy.special 
import numpy as np

class NDBasis:
    """
    
    A generic multi-dimensional finite basis representation
    supporting quadrature integration/transformation.
    :class:`NDBasis` objects define a set of :math:`N_b` basis functions,
    :math:`\\phi_i(\\vec{x})`, :math:`i = 0,\ldots,N_b - 1`, where 
    :math:`\\vec{x}` is an :math:`n_d`-dimensional coordinate vector. Matrix elements
    with these functions are defined with respect to a weighted integral,
    
    .. math::
       
      \langle \phi_i \\vert \phi_j \\rangle = \int d\\vec{x}\,\Omega(\\vec{x}) \\phi_i(\\vec{x}) \\phi_j(\\vec{x}),
     
    where :math:`\\Omega(\\vec{x})` is the weight function (:attr:`NDBasis.wgtfun`).
    
    These integrals can be approximated with a quadrature over :math:`N_q`
    (possibly scattered) grid points :math:`\\vec{x}_k`,
    
    .. math::
        
       \int d\\vec{x}\, \\Omega(\\vec{x}) f(\\vec{x}) \\approx \sum_{k=0}^{N_q-1} w_k f(\\vec{x}_k),
    
    where :math:`w_k` are the quadrature weights (:attr:`NDBasis.wgt`). 
    
    Attributes
    ----------
    nd : int
        The number of dimensions (i.e. coordinates)
    Nb : int 
        The number of basis functions
    Nq : int
        The number of quadrature points
    qgrid : (`nd`,`Nq`) ndarray
        The quadrature grid.
    wgt : (`Nq`,) ndarray
        The quadrature weights.
    bas : (`Nb`,`Nq`) ndarray
        The basis functions evaluated on the quadrature grid.
    basisfun : DFun
        An `Nb`-output-valued DFun of the basis functions 
        with `nd` input variables.
    wgtfun : DFun
        The weight function associated with matrix element
        integrals of the basis.
        
    """    
    
    def __init__(self, basisfun, wgtfun, qgrid, wgt):
        """
        Initialize a generic NDBasis.

        Parameters
        ----------
        basisfun : DFun
            The basis function DFun.
        wgtfun : DFun or None
            The integration weight function. 
            If None, this is assumed to be unity.
        qgrid : (`nd`, `Nq`) ndarray
            The quadrature points.
        wgt : (`Nq`,) ndarray
            The quadrature weights.
            
        Returns
        -------
        None.

        """
        self.basisfun = basisfun 
        self.wgtfun = wgtfun 
        self.qgrid = qgrid 
        self.wgt = wgt
        
        self.nd = basisfun.nx  # The number of dimensions
        self.Nb = basisfun.nf  # The number of basis functions
        self.Nq = qgrid.shape[1] # The number of quadrature points
        
        
        if qgrid.shape != (self.nd, self.Nq):
            raise ValueError("qgrid is the wrong size!")
        if wgtfun is not None:
            if wgtfun.nx != basisfun.nx or wgtfun.nf != 1 :
                raise ValueError("invalid wgtfun")
        if wgt.shape != (self.Nq,):
            raise ValueError("wgt is the wrong size!")
        
        # Calculate the `bas` grid
        self.bas = basisfun.val(qgrid)
        
        return 
        
    
    def fbrToQuad(self, v, axis = 0):
        """
        Transform an axis from the FBR
        to the quadrature grid representation.

        Parameters
        ----------
        v : (...,`Nb`,...) ndarray
            An array with the `axis` index spanned
            by this basis.

        Returns
        -------
        w : (..., `Nq`, ...) ndarray
            The transformed array.

        """
        
        return self._fbrToQuad(v, axis)
    
    def quadToFbr(self, w, axis = 0):
        """
        Transform an axis from the quadrature representation
        to the FBR.

        Parameters
        ----------
        w : (...,`Nq`,...) ndarray
            An array with the `axis` index spanned
            by this quadrature.

        Returns
        -------
        v : (..., `Nb`, ...) ndarray
            The transformed array.

        """
        
        return self._quadToFbr(w, axis)
    
    def _fbrToQuad(self, v, axis = 0):
        
        """ The default implemention of
        the FBR to quadrature transformation"""
        
        UT = self.bas.T # An (Nq,Nb) array 
        
        # Apply the FBR-to-grid transformation
        # to the axis
        w = np.tensordot(UT,v, axes = (1,axis) )
        w = np.moveaxis(w, 0, axis)
        
        # Broadcast the wgt's
        shape = [1] * v.ndim 
        shape[axis] = self.Nq 
        w *= np.sqrt(self.wgt).reshape(tuple(shape))
        
        return w
    
    def _quadToFbr(self, w, axis = 0):
        
        """ The default implemention of
        the quadrature to FBR transformation"""
        
        U = self.bas.conj() # An (Nb, Nq) array 
        
        # Broadcast the wgt's
        shape = [1] * w.ndim 
        shape[axis] = self.Nq
        w = w * np.sqrt(self.wgt).reshape(tuple(shape))
        
        # Apply the grid-to-FBR transformation
        # to the axis
        v = np.tensordot(U,w, axes = (1,axis) )
        v = np.moveaxis(v, 0, axis)
        
        return v

class SinCosBasis(NDBasis):
    
    """
    
    A 1-D, real sine/cosine basis set,
    
    .. math::

       f_m(\\phi) &= 1/\sqrt{\pi} \sin(|m|\phi) &\ldots m < 0\\\\
       &= 1/\sqrt{2\pi}              &\ldots m = 0\\\\
       &= 1/\sqrt{\pi} \cos(m \phi)  &\ldots m > 0 

    Attributes
    ----------
    
    m : ndarray
        The m quantum number of each basis function.
    angle : {'rad', 'deg'}
        The angular unit. 
        
    See Also
    --------
    nitrogen.special.SinCosDFun : DFun sub-class for sine-cosine basis set.
    
    """
    
    def __init__(self, m = 10, Nq = None, angle = 'rad'):
        """
        A real sine-cosine Fourier basis.

        Parameters
        ----------
        
        m : int or 1-D array_like of int, optional
            If scalar, the :math:`2|m|+1` basis functions with
            index :math:`\leq |m|` will be included. If array_like,
            then `m` lists all m-indices to be included.
            The default is `m` = 10.
        Nq : int, optional
            The number of quadrature points. The default
            is 2*(max(abs(`m`)) + 1)
        angle : {'rad', 'deg'}
            The angular unit. The default is 'rad'. 

        """
        
        # Create a DFun for the basis functions
        basisfun = special.SinCosDFun(m, angle = angle)
        
        if angle != 'rad' and angle != 'deg':
            raise ValueError('unexpected angular unit') 
        
        #
        # Construct the quadrature grid
        # and weights.
        # We use Nq  points uniformly
        # distributed over phi = [0,2*pi)
        # The weights are also uniform 
        # and equal 2*pi / Nq 
        #
        if Nq is None: 
            # Nq = 2*(mmax + 1)
            Nq = 2 * (max(abs(basisfun.m)) + 1) 
        
        if angle == 'rad':
            qgrid = np.linspace(0,2*np.pi,Nq+1)[:-1]
            wgt = np.full((Nq,), 2*np.pi / Nq)
        else: # angle == 'deg'
            qgrid = np.linspace(0, 360.0,Nq+1)[:-1]
            wgt = np.full((Nq,), 360.0 / Nq) 
        
        qgrid = qgrid.reshape((1,Nq))
        super().__init__(basisfun, None, qgrid, wgt) 
        
        self.m = basisfun.m  # The m quantum number list
        self.angle = angle
        
        # For now, we will use the default
        # FBR/grid transformation routine for
        # NDBasis objects. As this is a Fourier
        # basis, an FFT could probably be used with
        # better scaling (n*log(n) vs n**2).
        #
        return

class LegendreLMCosBasis(NDBasis):
    
    """
    
    Associated Legendre polynomials with cosine
    argument, :math:`F_\ell^m(\\theta) \propto P_\ell^m(\cos\\theta)`.
    See :class:`~nitrogen.special.LegendreLMCos`.
    
    Quadrature is performed with a Gauss-Legendre grid.
    
    These functions are eigenfunctions of the differential
    operator
    
    .. math::
       -\\frac{\partial^2}{\partial \\theta^2} - \cot \\theta \\frac{\partial}{\partial \\theta} + \\frac{m^2}{\sin^2\\theta}
       
    with eigenvalue :math:`\ell(\ell+1)`.
             
    Attributes
    ----------
    m : int
        The associated Legendre order.
    l : ndarray
        A list of l-indices of the basis functions.
        
    See Also
    --------
    nitrogen.special.LegendreLMCos : DFun sub-class for associated Legendre functions.

    
    """
    
    def __init__(self, m, lmax, Nq = None):
        """
        Create a LegendreLMCosBasis.

        Parameters
        ----------
        
        m : int 
            The associated Legendre order.
        lmax : int
            The maximum `l` index.
        Nq : int, optional
            The number of quadrature points. The default
            is 2*lmax + 1.

        """
        
        # Create a DFun for the basis functions
        basisfun = special.LegendreLMCos(m, lmax) 
        
        #
        # Construct the quadrature grid
        # and weights.
        if Nq is None: 
            Nq = 2*lmax + 1 # default quadrature grid size
        
        x,w = scipy.special.roots_legendre(Nq)
        qgrid = np.flip( np.arccos(x) ).reshape((1,Nq))
        wgt = np.flip(w)
        
        super().__init__(basisfun, special.Sin(), qgrid, wgt) 
        
        self.m = basisfun.m  # The associated Legendre order
        self.l = basisfun.l  # The l-index list

        return
    
class RealSphericalHBasis(NDBasis):
    
    """
    
    Real spherical harmonics,
    :math:`\\Phi_\ell^m(\\theta,\\phi) = F_\ell^m(\\theta)f_m(\\phi)`.
    See :class:`~nitrogen.special.RealSphericalH`.
    
    The integration weight function is :math:`\\Omega(\\theta,\\phi) = \\sin(\\theta)`.
    Quadrature is performed with a direct product of 
    a Gauss-Legendre grid over :math:`\\theta` and a uniform Fourier
    grid over :math:`\\phi`.
    
    These functions are eigenfunctions of the differential
    operator
    
    .. math::
       -\\frac{\partial^2}{\partial \\theta^2} - \cot \\theta \\frac{\partial}{\partial \\theta} - \\frac{\partial_\\phi^2}{\sin^2\\theta}
       
    with eigenvalue :math:`\ell(\ell+1)`.
             
    Attributes
    ----------
    m : ndarray
        The project quantum number :math:`m`.
    l : ndarray
        The azimuthal quantum number :math:`\ell`.
        
    See Also
    --------
    nitrogen.special.RealSphericalH : DFun sub-class real spherical harmonics.
    nitrogen.special.LegendreCosLM : DFun sub-class for associated Legendre polynomials.
    nitrogen.special.SinCosDFun : DFun sub-class for sine-cosine basis.

    
    """
    
    def __init__(self, m, lmax, Ntheta = None, Nphi = None):
        """
        Create a RealSphericalH basis.

        Parameters
        ----------
        
        m : int or array_like
            The associated Legendre order(s).
        lmax : int
            The maximum :math:`\ell` quantum number.
        Ntheta : int, optional
            The number of quadrature points over :math:`\\theta`.
            The default is 2*`lmax` + 1.
        Nphi : int, optional
            The number of quadrature points over :math:`\\phi`.
            The default is 2(mmax+1).

        """
        
        # Create a DFun for the basis functions
        basisfun = special.RealSphericalH(m, lmax)
        
        #
        # Construct the quadrature grid
        # and weights.
        if Ntheta is None: 
            Ntheta = 2*max(basisfun.l) + 1 # default quadrature grid size
        if Nphi is None:
            Nphi = 2*(max(abs(basisfun.m)) + 1) 
        
        # theta grid
        x,w = scipy.special.roots_legendre(Ntheta)
        theta_grid = np.flip(np.arccos(x)).reshape((1,Ntheta))
        theta_wgt = np.flip(w)
        # phi grid 
        phi_grid = np.linspace(0,2*np.pi,Nphi+1)[:-1].reshape((1,Nphi))
        phi_wgt = np.full((Nphi,), 2*np.pi / Nphi)
        
        qgrid = np.stack(np.meshgrid(theta_grid, phi_grid, indexing = 'ij'))
        qgrid = qgrid.reshape((2,Ntheta*Nphi)) # (nd, Nq)
        
        wgt = theta_wgt.reshape((Ntheta,1)) * phi_wgt 
        wgt = wgt.reshape((Ntheta*Nphi,)) 
        
        super().__init__(basisfun, special.Sin(nx=2), qgrid, wgt) 
        
        self.m = basisfun.m  # The associated Legendre order
        self.l = basisfun.l  # The l-index list

        return
    #
    # TO-DO:
    # For now, the default NDBasis FBR-to-grid routines
    # will be used. This should be replaced with a new
    # routine that takes advantage of the separability
    # of the spherical harmonic basis functions.
    #
    
class Real2DHOBasis(NDBasis):
    
    """
    
    Real 2-D isotropic harmonic oscillator basis functions
    in cylindrical coordinates.
    :math:`\\chi_n^{\\ell}(r,\\phi) = R_n^{\\ell}(r)f_{\\ell}(\\phi)`.
    See :class:`~nitrogen.special.Real2DHO`.
    
    Quadrature is performed with a direct product of 
    a Gauss-Laguerre-type grid over :math:`r` and a uniform Fourier
    grid over :math:`\\phi`.
    
    The integration weight function is :math:`\\Omega(r,\\phi) = r`.
             
    Attributes
    ----------
    v : ndarray
        The :math:`v` quantum numbers, where
        :math:`v = 2n + \\vert \\ell \\vert`.
    ell : ndarray
        The :math:`\\ell` quantum numbers.
    n : ndarray
        The :math:`n` Laguerre degree.
    rmax : float
        The radial extent of the basis.
    alpha : float
        The radial scaling parameter, :math:`\\alpha`, 
        corresponding to radial extent `R`.
    angle : {'rad', 'deg'}
        The angular units. 
        
    See Also
    --------
    nitrogen.special.Real2DHO : DFun sub-class real 2-D harmonic oscillator wavefunctions.
    nitrogen.special.RadialHO : DFun sub-class for d-dimensional radial harmonic oscillator wavefunctions.
    nitrogen.special.SinCosDFun : DFun sub-class for sine-cosine basis.

    
    """
    
    def __init__(self, vmax, rmax, ell = None, Nr = None, Nphi = None, angle = 'rad'):
        """
        Create a Real2DHO basis.

        Parameters
        ----------
        vmax : int 
            The maximum vibrational quantum number :math:`v`, 
            in the conventional sum-of-modes sense.
        rmax : float
            The radial extent of the basis.
        ell : scalar or 1-D array_like, optional
            The angular momentum quantum number.
            If scalar, then all :math:`\\ell` with 
            :math:`|\\ell| \leq` abs(`ell`) will be included. If array_like,
            then `ell` lists all (signed) :math:`\\ell` values to be included.
            A value of None is equivalent to `ell` = `vmax`. The default is 
            None.
        Nr : int, optional
            The number of quadrature points over :math:`r`.
            The default is `vmax` + 1.
        Nphi : int, optional
            The number of quadrature points over :math:`\\phi`.
            The default is :math:`2(\\ell_{max} + 1)`.
        angle : {'rad', 'deg'} 
            The angular unit. The default is 'rad'. 

        """

        #
        # Construct the quadrature grid
        # and weights.
        if Nr is None: 
            Nr = vmax + 1 # default quadrature grid size
        # r grid 
        d = 2 # The dimensionality 
        #
        # For dimensionality d, we should use a radial
        # quadrature built on Laguerre polynomials 
        # of order nu = d/2 - 1, which is the order 
        # corresponding to the lambda = 0 angular momentum
        # series. (nu = lambda + d/2 - 1)
        #
        x,wx = scipy.special.roots_genlaguerre(Nr, d/2 - 1) 
        alpha = max(x) / rmax**2  # Determine the alpha scaling parameter
                                  # needed to place the last radial grid point
                                  # at rmax 
        r_grid = np.sqrt(x/alpha)
        r_wgt =  np.exp(x) * wx/2.0 * (alpha)**(-d/2)
        
        #
        # Now that we know the value of alpha, we can
        # construct the basis DFun
        basisfun = special.Real2DHO(vmax, alpha, ell, angle = angle)
        
        # Now that we have the basis set set up, 
        # we know what the maximum ell value is.
        # 
        if Nphi is None:
            Nphi = 2*(max(abs(basisfun.ell)) + 1) 
        # Calculate the phi quadrature grid
        if angle == 'rad':
            phi_grid = np.linspace(0,2*np.pi,Nphi+1)[:-1].reshape((1,Nphi))
            phi_wgt = np.full((Nphi,), 2*np.pi / Nphi)
        else: # angle == 'deg'
            phi_grid = np.linspace(0, 360.0, Nphi+1)[:-1].reshape((1,Nphi))
            phi_wgt = np.full((Nphi,), 360.0 / Nphi)
        # Combine the two grids
        qgrid = np.stack(np.meshgrid(r_grid, phi_grid, indexing = 'ij'))
        qgrid = qgrid.reshape((2,Nr*Nphi)) # (nd, Nq)
        # and the weights...
        wgt = r_wgt.reshape((Nr,1)) * phi_wgt 
        wgt = wgt.reshape((Nr*Nphi,)) 
        
        # Volume element = r * dr dphi
        super().__init__(basisfun, special.Monomial([1,0]), qgrid, wgt) 
        
        self.v = basisfun.v         # The convention vibrational quantum number
        self.ell = basisfun.ell     # The angular momentum quantum number
        self.n = basisfun.n         # The Laguerre index: v = 2*n + ell
        self.rmax = rmax            # The radial extent
        self.alpha = basisfun.alpha # The corresponding alpha scaling parameter
        self.angle = angle          # The angular units
        # (private attributes)
        self._nr = Nr               # The number of radial quadrature points
        self._nphi = Nphi           # The number of angular quadrature points

        return
    #
    # TO-DO:
    # For now, the default NDBasis FBR-to-grid routines
    # will be used. This should be replaced with a new
    # routine that takes advantage of the separability
    # of the basis functions.
    #
    
class RadialHOBasis(NDBasis):
    
    """
    A radial basis for a harmonic oscillator in :math:`d` dimensions.
    """
    
    def __init__(self, vmax, rmax, ell, d = 2, Nr = None):
        """
        Create a RadialHO basis.

        Parameters
        ----------
        vmax : int 
            The number of basis functions is vmax - ell + 1. This is *not*
            the conventional vibrational quantum number.
        rmax : float
            The radial extent of the basis.
        ell : scalar
            The generalized angular momentum quantum number.
        d : int
            The dimension, :math:`d`.
        Nr : int, optional
            The number of quadrature points over :math:`r`.
            The default is `vmax` + 3.

        """

        #
        # Construct the quadrature grid
        # and weights.
        nmax = vmax - abs(ell) 
        if Nr is None: 
            Nr = vmax + 3 # default quadrature grid size
        # r grid 
        
        #
        # For dimensionality d, we should use a radial
        # quadrature built on Laguerre polynomials 
        # of order nu = d/2 - 1, which is the order 
        # corresponding to the lambda = 0 angular momentum
        # series. (nu = lambda + d/2 - 1)
        #
        x,wx = scipy.special.roots_genlaguerre(Nr, d/2 - 1) 
        alpha = max(x) / rmax**2  # Determine the alpha scaling parameter
                                  # needed to place the last radial grid point
                                  # at rmax 
        r_grid = np.sqrt(x/alpha)
        r_wgt =  np.exp(x) * wx/2.0 * (alpha)**(-d/2)
        
        qgrid = r_grid.reshape((1,Nr))
        wgt = r_wgt.reshape((Nr,))
        
        #
        # Now that we know the value of alpha, we can
        # construct the basis DFun
        basisfun = special.RadialHO( nmax, abs(ell), d, alpha)
        
        
        # Volume element = r**(d-1)
        super().__init__(basisfun, special.Monomial([d-1]), qgrid, wgt) 
        
        self.nmax = nmax            # The convention vibrational quantum number
        self.ell = abs(ell)         # The angular momentum quantum number
        self.rmax = rmax            # The radial extent
        self.alpha = basisfun.alpha # The corresponding alpha scaling parameter
        self.d = d                  # The dimension 
        # (private attributes)
        self._nr = Nr               # The number of radial quadrature points

        return