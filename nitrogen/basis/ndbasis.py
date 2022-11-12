"""
ndbasis.py

Implements the NDBasis base class and common
sub-classes.

=============================================  =================================
:class:`~nitrogen.basis.NDBasis` sub-class       Description
---------------------------------------------  --------------------------------- 
:class:`~nitrogen.basis.SinCosBasis`           A sine-cosine (real Fourier) basis
:class:`~nitrogen.basis.LegendreLMCosBasis`    Associated Legendre polynomials.
:class:`~nitrogen.basis.RealSphericalHBasis`   Real spherical harmonics.
:class:`~nitrogen.basis.Real2DHOBasis`         Two-dimensional harmonic osc. basis.
:class:`~nitrogen.basis.RadialHOBasis`         Radial HO basis in d dimensions.
=============================================  =================================

"""

import nitrogen.special as special
import nitrogen.dfun as dfun 
import scipy.special 
import numpy as np

from .ndbasis_c import _structured_op_double

__all__ = ['NDBasis','StructuredBasis',
           'SinCosBasis', 'LegendreLMCosBasis',
           'RealSphericalHBasis','Real2DHOBasis','RadialHOBasis']

from .genericbasis import GriddedBasis 

class NDBasis(GriddedBasis):
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

        """
        
        super().__init__(qgrid, basisfun.nf, wgtfun = wgtfun) 
        
        self.basisfun = basisfun 
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
        self.bas = basisfun.val(qgrid) # (Nb, Nq) 
        
        
        self._UH = (self.bas * np.sqrt(self.wgt)).T    # UH is the generic W
        self._U  = self.bas.conj() * np.sqrt(self.wgt) # U is the generic W^dagger
        
        dbas = basisfun.f(qgrid, deriv = 1) 
        self._Zi = [(dbas[i+1] * np.sqrt(self.wgt)).T for i in range(self.nd) ]
        # The generic Z matrix
        # for each coordinate variable
        
        # The D_i derivative quadrature rep.
        Di = [self._Zi[i] @ self._U for i in range(self.nd)] 
        self._Di = Di 
        
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
    
    def quadD(self, x, var, axis = 0):
        """
        Apply the quadrature representation derivative operator.

        Parameters
        ----------
        x : (...,`Nq`,...) ndarray
            The quadrature representation array.
        var : int
            The coordinate of the derivative.
        axis : int, optional
            The quadrature axis. The default is 0. 

        Returns
        -------
        y : (...,`Nq`,...) ndarray
            The result.

        """
        return self._quadD(x,var,axis)
        
    def quadDH(self, x, var,axis = 0):
        """
        Apply the quadrature representation D^dagger operator.

        Parameters
        ----------
        x : (...,`Nq`,...) ndarray
            The quadrature representation array.
        var : int
            The coordinate of the derivative.
        axis : int, optional
            The quadrature axis. The default is 0. 

        Returns
        -------
        y : (...,`Nq`,...) ndarray
            The result.

        """
        return self._quadDH(x,var,axis)
    
    def _fbrToQuad(self, v, axis = 0):
        
        """ The default implemention of
        the FBR to quadrature transformation"""
        
        # Apply the FBR-to-grid transformation
        # to the axis
        w = np.tensordot(self._UH, v, axes = (1,axis) )
        w = np.moveaxis(w, 0, axis)
        
        return w
    
    def _quadToFbr(self, w, axis = 0):
        
        """ The default implemention of
        the quadrature to FBR transformation"""
        
        # Apply the grid-to-FBR transformation
        # to the axis
        v = np.tensordot(self._U, w, axes = (1,axis) )
        v = np.moveaxis(v, 0, axis)
        
        return v
    
    def _quadD(self, x, var, axis = 0):
        """ The default implementation of quadD"""
        y = np.tensordot(self._Di[var], x, axes = (1,axis))
        y = np.moveaxis(y, 0, axis) 
        return y
    def _quadDH(self, x, var, axis = 0):
        """ The default implementation of quadDH"""
        y = np.tensordot(self._Di[var].conj().T, x, axes = (1,axis))
        y = np.moveaxis(y, 0, axis) 
        return y 
    
    #
    # GriddedBasis methods
    #
    def _basis2grid(self, x, axis = 0):
        return self.fbrToQuad(x, axis = axis)
    def _grid2basis(self,x, axis = 0):
        return self.quadToFbr(x, axis = axis) 
    def _basis2grid_d(self,x,var, axis = 0):
        y = np.tensordot(self._Zi[var], x, axes = (1,axis))
        y = np.moveaxis(y, 0, axis) 
        return y
    def _grid2basis_d(self, x,var, axis = 0):
        y = np.tensordot(self._Zi[var].conj().T, x, axes = (1,axis))
        y = np.moveaxis(y, 0, axis) 
        return y 
    def _d_grid(self,x,var, axis = 0):
        return self.quadD(x, var, axis = axis)
    def _dH_grid(self,x,var, axis = 0):
        return self.quadDH(x, var, axis = axis) 
    
    
class StructuredBasis(NDBasis):
    """
    Multi-dimensional bases that have a structured product
    form, such as spherical harmonics, D-matrices, etc,
    and direct product quadrature grids. 
    """
    
    def __init__(self, bases, structure):
        """
        Initialize a generic StructuredBasis.

        Parameters
        ----------
        bases : list of NDBasis objects
            The basis function factors
        structure : (Nb, nfs) array_like
            The product structure of each basis function in
            terms of the indices of each factor basis in `bases`.
            
        Notes
        -----
        Currently only 1-dimensional factor bases are supported. This may
        be expanded in the future.
        
        """
        
        structure = np.array(structure)
        if structure.ndim != 2:
            raise ValueError("structure must be 2-d")
        
        Nb,nfs = structure.shape
        nd = nfs # Assume the number of coordinates is the number of factors
                 # (i.e. all factors are 1-d. this may change in future)
        if len(bases) != nfs:
            raise ValueError("len(bases) must equal structure.shape[1]")
        
        # Check that each basis is 1-d
        # This may be extended in the future.
        for i in range(nfs):
            if bases[i].nd != 1:
                raise ValueError("All basis factors must be 1-dimensional. "
                                 "This may change in the future.")
        basisfuns = [b.basisfun for b in bases]
        basisfun = dfun.SelectedProduct(basisfuns, structure) 
        
        #
        # wgtfun is a simple product of 
        # each element in wgtfuns
        wgtfun = dfun.SimpleProduct([b.wgtfun for b in bases])
        
        # Construct the direct product 
        # quadrature grids and shape into 
        # a (nd, Nq) array 
        qgrids = [b.qgrid for b in bases] 
        qgrid = np.stack(np.meshgrid(*qgrids, indexing = 'ij')).reshape((nd,-1))
        
        # Construct the total quadrature weights
        # by taking the direct product of all
        # individual weights 
        wgts = [b.wgt for b in bases]
        wgt = np.array(wgts[0]).copy()
        for i in range(1, nfs):
            wgt = np.outer(wgt, wgts[i]).reshape((-1,))
        
        super().__init__(basisfun, wgtfun, qgrid, wgt)
        
        # 
        # Analyze the structure 
        # ---------------------
        #
        # Determine the size of the recursively smaller
        # sub-structures, working from left-to-right
        #
        # The first layer is just the structured basis
        # itself, which we do not include
        Nsub = []
        #
        # At the same time, record which index of the immediate
        # sub-structure is paired with each member of
        # the structure before it. These inverse indices are
        # provided by np.unique(..., return_inverse = True).
        # There are only nd - 1 of these to keep track of.
        #
        sub_idx = [] 
        #
        # We also need to keep track of the first index of any
        # given sub-structure
        #
        lead_idx = [structure[:,0].copy().astype(np.uint)] 
        
        sub_structure = structure # Initialize
        final_nb = Nb
        for i in range(1,nfs):
            #
            # The sub-structure is the list of unique 
            # rows for the remaining columns of the given structure
            #
            sub_structure, inverse_idx = np.unique(sub_structure[:,1:], 
                                                   return_inverse = True, 
                                                   axis = 0)
            final_nb = sub_structure.shape[0]
            Nsub.append(sub_structure.shape[0]) 
            sub_idx.append(inverse_idx.astype(np.uint))
            lead_idx.append(sub_structure[:,0].copy().astype(np.uint))
            
        #
        # The final sub-structure is just a dummy singleton
        Nsub.append(1)
        sub_idx.append(np.array([0]*final_nb))
        
        Nsub = tuple(Nsub)
        
        # For each factor, we need the W and Z 
        # quadrature transformation matrices from their
        # respective bases to grids 
        Ws = [] 
        Zs = [] # a list of lists 
                
        for i in range(nfs):
            b = bases[i] 
            dbas = b.basisfun.f(b.qgrid, deriv = 1)
            
            Wi =  (dbas[0] * np.sqrt(b.wgt)).T 
            Zi = []
            for j in range(b.nd):
                Zij =  (dbas[j+1] * np.sqrt(b.wgt)).T
                Zi.append(Zij)
            
            Ws.append(Wi)
            Zs.append(Zi)
        
        self.nfs = nfs # The number of factors
        self.Nsub = Nsub 
        self.sub_idx = sub_idx 
        self.lead_idx = lead_idx 
        self.Ws = Ws 
        self.Zs = Zs 
        
        return 
    #@profile
    def _fbrToQuad(self, v, axis = 0):
        """ structured basis fbr-to-quadrature transformation
        """
        
        #
        # Prepare data in proper shape
        #
        x = _reshape_axis_to_center(v, axis)
        
        for i in range(self.nfs): 
            
            W = self.Ws[i] # The quadrature transformation for this factor
            nq = W.shape[0]
            nsub = self.Nsub[i] 
            
            y = np.empty_like(x, shape = (x.shape[0],nq,nsub,x.shape[2]))
            
            # Now perform the quadrature transformation for this 
            # factor 
            _structured_op_double(x, y, W, 
                                  self.lead_idx[i].astype(np.intc), 
                                  self.sub_idx[i].astype(np.intc))
            #
            # y contains the intermediate result.
            # as (..., nq, nsub, ...)
            # reshape it so that the remaining sub-structure index (axis = 2)
            # is in the middle for the next iteration
            x = _reshape_axis_to_center(y, 2)
            
        #
        # y (and x) should now be the completed quadrature representation
        # of this basis. 
        out_shape = list(v.shape) 
        out_shape[axis] = self.Nq 
        y = y.reshape(tuple(out_shape))
        
        return y 
    
    def _quadToFbr(self, w, axis = 0):
        """ structured basis quad-to-fbr transformation
        """
        #
        # Reshape data
        #
        x = _reshape_axis_to_center(w, axis)
        # x now has shape (..., Nq, ...) 
        #
        for i in range(self.nfs-1, -1, -1): 
            # Working backwards
            
            Wh = self.Ws[i].conj().T # W^dagger 
            nq = Wh.shape[1] # the number of quadrature points for this factor
            nsub = self.Nsub[i] # the sub-structure after this factor 
            
            #
            # reshape x to a 4-d, push everything to the left
            x = x.reshape( (-1, nq, nsub, x.shape[-1])) 
            
            # reverse structured op
            
            # y is now (..., nb, ...)
            # where nb is the nsub of the left-ward factor.
            #
            # Let x reference y for the next iteration
            x = y 
            
            
            
def _reshape_axis_to_center(x, axis):
    """
    Reshape x to a 3D array with 
    `axis` at the center. The first and/or
    last axes will be singleton if necessary
    
    """
    
    n_pre = 1 
    for i in range(axis):
        n_pre *= x.shape[i]
    
    n_post = 1
    for i in range(axis+1,x.ndim):
        n_post *= x.shape[i]
        
    new_shape = (n_pre, x.shape[axis], n_post) 
    
    return np.reshape(x, new_shape) 
    
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
            is 2*max(abs(`m`)) + 3 
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
            # Nq = 2*mmax + 3
            Nq = 2 * max(abs(basisfun.m)) + 3 
        
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
        The projection quantum number :math:`m`.
    l : ndarray
        The azimuthal quantum number :math:`\ell`.
        
    See Also
    --------
    nitrogen.special.RealSphericalH : DFun sub-class real spherical harmonics.
    nitrogen.special.LegendreLMCos : DFun sub-class for associated Legendre polynomials.
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