"""
ndbasis.py

Implements the NDBasis base class and common
sub-classes.

=============================================  =================================
:class:`~nitrogen.ndbasis.NDBasis` sub-class   Description
---------------------------------------------  --------------------------------- 
:class:`~nitrogen.dvr.SinCosBasis`             A sine-cosine (real Fourier) basis
:class:`~nitrogen.dvr.LegendreLMCosBasis`      Associated Legendre polynomials.
:class:`~nitrogen.dvr.RealSphericalHBasis`     Real spherical harmonics.
=============================================  =================================

"""

import nitrogen.special as special
import scipy.special 
import numpy as np

class NDBasis:
    """
    
    A generic multi-dimensional finite basis representation
    supporting quadrature integration/transformation.
    
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
            return ValueError("qgrid is the wrong size!")
        if wgtfun is not None:
            if wgtfun.nx != basisfun.nx or wgtfun.nf != 1 :
                return ValueError("invalid wgtfun")
        if wgt.shape != (self.Nq,):
            return ValueError("wgt is the wrong size!")
        
        # Calculate the `bas` grid
        self.bas = basisfun.val(qgrid)
        
        
    
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
        
        U = self.bas.T # An (Nq,Nb) array 
        
        # Apply the FBR-to-grid transformation
        # to the axis
        w = np.tensordot(U,v, axes = (1,axis) )
        w = np.moveaxis(w, 0, axis)
        
        # Broadcast the wgt's
        shape = [1] * v.ndim 
        shape[axis] = self.Nq 
        w *= np.sqrt(self.wgt).reshape(tuple(shape))
        
        return w
    
    def _quadToFbr(self, w, axis = 0):
        
        """ The default implemention of
        the quadrature to FBR transformation"""
        
        U = self.bas # An (Nb, Nq) array 
        
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
        
    See Also
    --------
    nitrogen.special.SinCosDFun : DFun sub-class for sine-cosine basis set.
    
    """
    
    def __init__(self, m = 10, Nq = None):
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

        """
        
        # Create a DFun for the basis functions
        basisfun = special.SinCosDFun(m)
        
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
            
        qgrid = np.linspace(0,2*np.pi,Nq+1)[:-1]
        qgrid = qgrid.reshape((1,Nq))
        wgt = np.full((Nq,), 2*np.pi / Nq)
        
        super().__init__(basisfun, None, qgrid, wgt) 
        
        self.m = basisfun.m  # The m quantum number list
        
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
            is lmax + 1.

        """
        
        # Create a DFun for the basis functions
        basisfun = special.LegendreLMCos(m, lmax) 
        
        #
        # Construct the quadrature grid
        # and weights.
        if Nq is None: 
            Nq = lmax + 1 # default quadrature grid size
        
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
            The default is `lmax` + 1.
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
            Ntheta = max(basisfun.l) + 1 # default quadrature grid size
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