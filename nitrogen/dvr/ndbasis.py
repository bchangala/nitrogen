"""
ndbasis.py

Implements the NDBasis base class and common
sub-classes: 

    1) SinCosBasis: a sine-cosine basis function
       (i.e. a real Fourier/exponential basis)
    2) LegendreLMCosBasis: Associated Legendre 
       polynomials with cosine arguments.

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

       f_m(\\phi) = 1/\sqrt{\pi} \sin(|m|\phi) \ldots m < 0
       
       = 1/\sqrt{2\pi}              \ldots m = 0
        
       = 1/\sqrt{\pi} \cos(m \phi)  \ldots m > 0 

    Attributes
    ----------
    
    m : ndarray
        The m quantum number of each basis function.
    
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