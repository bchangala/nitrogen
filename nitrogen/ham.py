"""
nitrogen.ham
------------

Hamiltonian construction routines.

"""

import numpy as np
from nitrogen.dvr import DVR
from nitrogen.dvr import NDBasis
import nitrogen.dvr.ops as dvrops
from nitrogen.dfun import sym2invdet
import nitrogen.angmom as angmom
import nitrogen.constants
import nitrogen.dvr 

from scipy.sparse.linalg import LinearOperator

def hdpdvr_bfJ(dvrs, cs, pes, masses, Jlist = 0, Vmax = None, Vmin = None):
    """
    Direct-product DVR grid body-frame Hamiltonian for 
    angular momentum J.

    Parameters
    ----------
    dvrs : list of DVR objects and scalars
        A list of :class:`nitrogen.dvr.DVR` basis set objects or
        scalar numbers. The length of the list must be equal to
        the number of coordinates in `cs`. Scalar elements indicate
        a fixed value for that coordinate.
    cs : CoordSys
        An *atomic* coordinate system.
    pes : DFun or function
        A potential energy function f(Q) with respect
        to `cs` coordinates.
    masses : array_like
        Masses.
    Jlist : int or array_like
        Total angular momentum value(s).
    Vmax : float, optional
        Maximum potential energy allowed. Higher values will be 
        replaced with `Vmax`. If None, this is ignored.
    Vmin : float, optional
        Minimum potential energy allowed. Lower values will be 
        replaced with `Vmin`. If None, this is ignored.
    
    Returns
    -------
    H : LinearOperator or list of LinearOperator
        The rovibrational Hamiltonian operator(s). If `Jlist` is a 
        scalar, then a single LinearOperator is returned. If `Jlist`
        is an array, then a list of LinearOperators is returned,
        whose elements are the corresponding Hamiltonians for each
        value of `Jlist`.

    """
    
    if len(dvrs) != cs.nQ:
        raise ValueError("The length of dvrs does not equal cs.nQ")
    
    # Determine the active and fixed coordinates
    vvar = []
    grids = []
    vshape = []
    NV = 1
    Dlist = []
    
    for i in range(len(dvrs)):
        
        if isinstance(dvrs[i], DVR): 
            # Active coordinate
            vvar.append(i)
            grids.append(dvrs[i].grid)
            ni = dvrs[i].num
            vshape.append(ni)
            NV *= ni
            Dlist.append(dvrs[i].D)
        else:
            # Inactive coordinate
            grids.append(dvrs[i]) # scalar value
            vshape.append(1)
            Dlist.append(None)
            
    vshape = tuple(vshape)
    # To summarize:
    # -------------
    # vvar is a list of active coordinates
    # grids contains 1D active grids and fixed scalar values
    # vshape is the grid shape *including* singleton fixed coordinates
    # NV is the total vibrational grid size
    # Dlist is the derivative operator list, including None's for fixed coord.
        
    if len(vvar) < 1:
        raise ValueError("There must be at least one active coordinate")
        
    # Calculate the coordinate grids
    Q = np.stack(np.meshgrid(*grids, indexing = 'ij'))
    
    # Calculate the metric tensor
    g = cs.Q2g(Q, masses = masses, deriv = 0, vvar = vvar, rvar = 'xyz',
               mode = 'bodyframe')
    
    # Calculate the inverse metric and metric determinant
    G, detg = sym2invdet(g, 0, len(vvar))
    
    # Calculate the final KEO grids
    gi2 = np.sqrt(detg[0])      # |g|^1/2
    gim4 = 1.0/(np.sqrt(gi2))   # |g|^-1/4
    Gkl = G[0]                  # G inverse metric
    hbar = nitrogen.constants.hbar    # hbar in [A, u, cm^-1] units
    
    # Calculate the PES grid
    try: # Attempt DFun interface
        V = pes.f(Q, deriv = 0)[0,0]
    except:
        V = pes(Q) # Attempt raw function
    
    
    if Vmax is not None:
        V[V > Vmax] = Vmax 
    if Vmin is not None: 
        V[V < Vmin] = Vmin 
    

    ######################
    # Create a `maker` function to construct the
    # mv routine. This needs a maker because I need certain
    # J-dependent references to have local scope here!
    def make_mvJ(J):
        # Construct the matrix-vector routine for
        # Hamiltonian with angular momentum J
        #
        NJ = (2*J+1)
        # Calculate the rotational operators
        iJ = angmom.iJbf_wr(J)      # Body-fixed angular momentum operators (times i/hbar)
        iJiJ = angmom.iJiJbf_wr(J)  # Anti-commutators of iJ/hbar
        NH = NJ * NV
        rvshape = (NJ,) + vshape
        ################################################
        # Matrix-vector routine for H(J)
        def mv(x):
            
            xgrid = x.reshape(rvshape) # Reshape vector to separate rot-vib indices
            y = np.zeros_like(xgrid)   # Result grids
            
            ############################################
            # Vibrational-only terms
            for k in range(NJ):
                xk = xgrid[k]   # The vibrational grid for the k^th rotation block
                
                yk = V * xk 
                yk += (hbar**2/2.0) * dvrops.opDD_grid(xk, gim4, gi2, Gkl, Dlist)
                
                y[k] += yk
            #
            ############################################
            
            ############################################
            # Rotation and rotation-vibration terms
            #
            if J > 0:
                for kI in range(NJ):
                    for kJ in range(NJ):
                        xk = xgrid[kJ] # The vibrational grid for the kJ rotational index
                        #############################################
                        # Compute mat-vec for y[kI] <--- x[kJ] block
                        #
                        # Rotation terms:
                        for a in range(3):
                            for b in range(a+1): # Loop only over half of Gab (it is symmetric)
                                #
                                # y[kI] <-- -hbar**2/4 * G_ab * [iJa,iJb]+
                                Gab = Gkl[dvrops.IJ2k(len(vvar)+a,len(vvar)+b)] # G_a,b
                                
                                if a == b:
                                    symfactor = 1.0 # On G_a,b diagonal
                                else:
                                    symfactor = 2.0 # On the off-diagonal -- two equal terms
                                
                                if iJiJ[a][b][kI,kJ] == 0.0:
                                    continue # Zero rotational matrix element
                                else:
                                    y[kI] += (symfactor * -hbar**2 / 4.0) \
                                        * iJiJ[a][b][kI,kJ] * (Gab * xk)
                        #
                        #
                        # Vibration-rotation terms:
                        for a in range(3):
                            # Extract the rot-vib row for all active vibs with axis `a`
                            Gka = Gkl[dvrops.IJ2k(len(vvar)+a,0) : dvrops.IJ2k(len(vvar)+a,len(vvar))]
                            
                            if iJ[a][kI,kJ] == 0.0:
                                continue # Zero rotational matrix element
                            else:
                                lambdax = dvrops.opD_grid(xk,Gka,Dlist) # (lambda/hbar) * x
                                y[kI] += -hbar**2/2.0 * iJ[a][kI,kJ] * lambdax
                        #
                        #
                        ###############################################
            #
            #######################################################
                        
            # Reshape rot-vib grid to a 1D vector
            return y.reshape((NH,))
        #
        # end matrix-vector routine
        ############################################
        return NH, mv # return the rank and mv function
    # end maker function
    #########################################
    
    # Finally, construct the LinearOperators
    #
    Hlist = []
    dtype = np.result_type(Q,Gkl,V)
    for J in np.array(Jlist).ravel(): # convert Jlist to iterable
        NHJ, mvJ = make_mvJ(J)
        HJ = LinearOperator((NHJ,NHJ), matvec = mvJ, dtype = dtype)
        Hlist.append(HJ)
    
    if np.isscalar(Jlist):
        return Hlist[0]
    else:
        return Hlist 


class DirProdDvrCartN(LinearOperator):
    """
    A LinearOperator subclass for direct-product DVR
    Hamiltonians for N Cartesian coordinates.
    
    
    Attributes
    ----------
    V : ndarray
        The PES grid with shape `vshape`.
    NH : int
        The size of the Hamiltonian (the number of direct-product grid points).
    vvar : list 
        The active coordinates.
    grids : list
        Grids (for active) and fixed scalar values (for inactive) 
        of each coordinate.
    vshape : tuple
        The shape of the direct-product grid.
    masses : ndarray
        The mass of each coordinate.
    hbar : float
        The value of :math:`\\hbar`. 
    """
    
    def __init__(self, dvrs, pes, masses = None, hbar = None, Vmax = None, Vmin = None):
        """
        

        Parameters
        ----------
        dvrs : list of DVR objects and/or scalars
            A list of :class:`nitrogen.dvr.DVR` objects and/or
            scalar numbers. The length of the list must be equal to
            the number of coordinates, `nx`. Scalar elements indicate
            a fixed value for that coordinate.
        pes : DFun or function
            A potential energy function f(X) with respect
            to the `nx` Cartesian coordinates.
        masses : array_like, optional
            A list of `nx` masses. If None, these will be assumed to be unity.
        hbar : float, optional
            The value of :math:`\\hbar`. If None, the default NITROGEN units are used.
        Vmax : float, optional
            Maximum potential energy allowed. Higher values will be 
            replaced with `Vmax`. If None, this is ignored.
        Vmin : float, optional
            Minimum potential energy allowed. Lower values will be 
            replaced with `Vmin`. If None, this is ignored.
            
        """
        
        nx = len(dvrs)
        
        if masses is None:
            masses = [1.0 for i in range(nx)]
        
        if len(masses) != nx:
            raise ValueError("The number of masses must equal the length of dvrs")

        if hbar is None:
            hbar = nitrogen.constants.hbar 
            
        # Determine the active and fixed coordinates
        vvar = []
        grids = []
        vshape = []
        NH = 1
        D2list = []
        
        for i in range(len(dvrs)):
            
            if np.isscalar(dvrs[i]): # inactive
                grids.append(dvrs[i])
                vshape.append(1)
                D2list.append(None) 
            else: # active, assume DVR object
                vvar.append(i)
                grids.append(dvrs[i].grid)
                ni = dvrs[i].num # length of this dimension 
                vshape.append(ni) 
                NH *= ni 
                D2list.append(dvrs[i].D2)

        vshape = tuple(vshape)
        # To summarize:
        # -------------
        # vvar is a list of active coordinates
        # grids contains 1D active grids and fixed scalar values
        # vshape is the grid shape *including* singleton fixed coordinates
        # NH is the total grid size
        # D2list is the second-derivative operator list, including None's for fixed coord.
            
        if len(vvar) < 1:
            raise ValueError("There must be at least one active coordinate")
            
        # Calculate the coordinate grids
        Q = np.stack(np.meshgrid(*grids, indexing = 'ij'))
        
        # Calculate the PES grid
        try: # Attempt DFun interface
            V = pes.f(Q, deriv = 0)[0,0]
        except:
            V = pes(Q) # Attempt raw function
        # Check max and min limits
        if Vmax is not None:
            V[V > Vmax] = Vmax 
        if Vmin is not None: 
            V[V < Vmin] = Vmin 
        
        # Determine the operator data-type
        dtype = np.result_type(Q,V)
         
        # Initialize the LinearOperator
        #super().__init__((NH,NH), matvec = self._cartn_mv, dtype = dtype)
        # Define the required LinearOperator attributes
        self.shape = (NH,NH)
        self.dtype = dtype
        
        # Define new attributes
        self.NH = NH
        self.vvar = vvar 
        self.grids = grids 
        self.vshape = vshape 
        self._D2list = D2list 
        self.masses = np.array(masses)
        self.hbar = hbar 
        self.V = V
    
    def _matvec(self, x):
        
        xgrid = x.reshape(self.vshape) # Reshape vector to direct-product grid
        y = np.zeros_like(xgrid)        # Result grid
        
        hbar = self.hbar 
        ############################################
        # PES term
        y += self.V * xgrid 
        # KEO terms
        y += (-hbar**2 /2.0) * dvrops.opO_coeff(xgrid, self._D2list, 1.0/self.masses)
        #
        ############################################
                    
        # Reshape result grid to a 1D vector
        return y.reshape((self.NH,))        
    
class DirProdDvrCartNQD(LinearOperator):
    """
    A LinearOperator subclass for direct-product DVR
    Hamiltonians of N Cartesian coordinates using
    a multi-state quasi-diabatic (QD) Hamiltonian.
    
    
    Attributes
    ----------
    Vij : list
        Vij[i][j] refers to the diabat/coupling grid between states i and j.
    NH : int
        The size of the Hamiltonian.
    NV : int
        The size of the coordinate grid.
    NS : int
        The number of states.
    vvar : list 
        The active coordinates.
    grids : list
        Grids (for active) and fixed scalar values (for inactive) 
        of each coordinate.
    vshape : tuple
        The shape of the direct-product coordinate grid.
    masses : ndarray
        The mass of each coordinate.
    hbar : float
        The value of :math:`\\hbar`. 
    """
    
    def __init__(self, dvrs, pes, masses = None, hbar = None, pesorder = 'LR'):
        """
        
        Parameters
        ----------
        dvrs : list of DVR objects and/or scalars
            A list of :class:`nitrogen.dvr.DVR` objects and/or
            scalar numbers. The length of the list must be equal to
            the number of coordinates, `nx`. Scalar elements indicate
            a fixed value for that coordinate.
        pes : DFun or function
            A potential energy/coupling surface function f(X) with respect
            to the `nx` Cartesian coordinates. This must have `NS`*(`NS`+1)/2
            output values for `NS` states.
        masses : array_like, optional
            A list of `nx` masses. If None, these will be assumed to be unity.
        hbar : float
            The value of :math:`\\hbar`. If None, the default value is that in 
            NITROGEN units. 
        pesorder : {'LR', 'LC'}, optional
            V matrix electronic state ordering. The `pes` function returns
            the *lower* triangle of the diabatic potential matrix. If 'LR'
            (default), the values are returned in row major order. If 'LC'
            the values are returned in column major order.
        """
        
        nx = len(dvrs)
        
        if masses is None:
            masses = [1.0 for i in range(nx)]
        
        if len(masses) != nx:
            raise ValueError("The number of masses must equal the length of dvrs")

        if hbar is None:
            hbar = nitrogen.constants.hbar 
    
        # Determine the active and fixed coordinates
        vvar = []
        grids = []
        vshape = []
        NV = 1
        D2list = []
        
        for i in range(len(dvrs)):
            
            if np.isscalar(dvrs[i]): # inactive
                grids.append(dvrs[i])
                vshape.append(1)
                D2list.append(None) 
            else: # active, assume DVR object
                vvar.append(i)
                grids.append(dvrs[i].grid)
                ni = dvrs[i].num # length of this dimension 
                vshape.append(ni) 
                NV *= ni 
                D2list.append(dvrs[i].D2)

        vshape = tuple(vshape)
        # To summarize:
        # -------------
        # vvar is a list of active coordinates
        # grids contains 1D active grids and fixed scalar values
        # vshape is the grid shape *including* singleton fixed coordinates
        # NV is the total grid size
        # D2list is the second-derivative operator list, including None's for fixed coord.
            
        if len(vvar) < 1:
            raise ValueError("There must be at least one active coordinate")
            
        # Calculate the coordinate grids
        Q = np.stack(np.meshgrid(*grids, indexing = 'ij'))
        
        # Calculate the PES/coupling grids
        try: # Attempt DFun interface
            V = pes.f(Q, deriv = 0)[0]  # shape is (NS*(NS+1)/2,) + vshape
        except:
            V = pes(Q) # Attempt raw function
            
        NS = round(np.floor(np.sqrt(2*V.shape[0])))
        
        # Make of list of individual grids
        Vij = [ [0 for j in range(NS)] for i in range(NS)]
        # Populate Vij list with references to individual grids
        if pesorder == 'LR':
            k = 0
            for i in range(NS):
                for j in range(i+1):
                    Vij[i][j] = V[k] # view of sub-array
                    Vij[j][i] = Vij[i][j] # Diabatic symmetry
                    k += 1
                    
        else: # pesorder = 'LC'
            k = 0
            for j in range(NS):
                for i in range(j+1):
                    Vij[i][j] = V[k] # view of sub-array
                    Vij[j][i] = Vij[i][j] # Diabatic symmetry
                    k += 1
        # Vij[i][j] now refers to the correct diabat/coupling surface
        
        NH = NS * NV  # The total size of the Hamiltonian
        
        # Determine the operator data-type
        dtype = np.result_type(Q,V)
         
        # Initialize the LinearOperator
        #super().__init__((NH,NH), matvec = self._cartn_mv, dtype = dtype)
        # Define the required LinearOperator attributes
        self.shape = (NH,NH)
        self.dtype = dtype
        
        # Define new attributes
        self.NH = NH
        self.NV = NV 
        self.NS = NS
        self.vvar = vvar 
        self.grids = grids 
        self.vshape = vshape 
        self._D2list = D2list 
        self.masses = np.array(masses)
        self.hbar = hbar 
        self.Vij = Vij
        
    
    def _matvec(self, x):
        
        evshape = (self.NS,) + self.vshape # Elec.-vib. tensor shape
        
        xgrid = x.reshape(evshape)  # Reshape vector to elec-vib grid
        y = np.zeros_like(xgrid)    # Result grid
        
        hbar = self.hbar 
        
        ############################################
        #
        for j in range(self.NS): # input block j
            xj = xgrid[j] # The coordinate grid of the j^th electronic block
            
            for i in range(self.NS): # output block i
                #######################################
                #
                # PES/coupling term (i <--> j)
                y[i] += self.Vij[i][j] * xj 
                #
                #
                # KEO (diagonal blocks only)
                if i == j:
                    y[i] += (-hbar**2 / 2.0) * dvrops.opO_coeff(xj, self._D2list, 1.0/self.masses)
                #
                #
                #######################################
        #        
        #
        ############################################
                    
        # Reshape result grid to a 1D vector
        return y.reshape((self.NH,)) 
    
    
class Polar2D(LinearOperator):
    """
    
    A Hamiltonian for a particle in two dimensions
    with polar coordinates :math:`(r,\\phi)` represented by the
    :class:`~nitrogen.dvr.Real2DHOBasis`
    two-dimensional harmonic oscillator basis set. 
    The differential operator is
    
    .. math::
       \\hat{H} = -\\frac{\\hbar^2}{2m} \\left[\\partial_r^2 + 
           \\frac{1}{r}\\partial_r + \\frac{1}{r^2}\\partial_\\phi^2\\right]
           + V(r,\\phi)
           
    with respect to integration as :math:`\int_0^\infty\,r\,dr\, \int_0^{2\pi}\,d\\phi`.
           
    
    """
    
    def __init__(self, Vfun, mass, vmax, R, ell = None, Nr = None, Nphi = None, hbar = None):
        """
        Class initializer.

        Parameters
        ----------
        Vfun : function or DFun
            A function evaluating the potential energy for a (2,...)-shaped
            input array containing :math:`(r,\\phi)` values and returning
            a (...)-shaped output array or a DFun with `nx` = 2.
        mass : float
            The mass.
        vmax : int
            Basis set paramter, see :class:`~nitrogen.dvr.Real2DHOBasis`.
        R : float
            Basis set parameter, see :class:`~nitrogen.dvr.Real2DHOBasis`.
        ell : int, optional
            Basis set parameter, see :class:`~nitrogen.dvr.Real2DHOBasis`.
            The default is None.
        Nr,Nphi : int, optional
            Quadrature parameter, see :class:`~nitrogen.dvr.Real2DHOBasis`.
            The default is None.
        hbar : float, optional
            The value of :math:`\\hbar`. If None, the default value in 
            standard NITROGEN units is used (``n2.constants.hbar``).

        """
        
        # Create the 2D HO basis set
        basis = nitrogen.dvr.Real2DHOBasis(vmax, R, ell, Nr, Nphi)
        NH = basis.Nb # The number of basis functions 
        
        # Calculate the potential energy surface on the 
        # quadrature grid 
        try: # Attempt DFun interface
            V = Vfun.f(basis.qgrid, deriv = 0)[0,0]
        except:
            V = Vfun(basis.qgrid) # Attempt raw function. Expecting (Nq,) output
            
        # Parse hbar 
        if hbar is None:
            hbar = nitrogen.constants.hbar 
        
        # Pre-construct the KEO operator matrix 
        T = np.zeros((NH,NH))
        for i in range(NH):
            elli = basis.ell[i]
            ni = basis.n[i]
            
            for j in range(NH):
                ellj = basis.ell[j]
                nj = basis.n[j]
                
                # Check \Delta \ell = 0
                if elli != ellj:
                    continue
                
                # Check for diagonal 
                if ni == nj:
                    T[i,j] = -basis.alpha*(2*ni + abs(elli) + 1)
                elif abs(ni-nj) == 1: # Check for off-diagonal
                    n = min(ni,nj)
                    T[i,j] = basis.alpha * np.sqrt((n+1)*(n+abs(elli)+1))
        #
        # Now finish with prefactor        
        T *= -(hbar**2 / (2*mass))
        
        # Define the required LinearOperator attributes
        self.shape = (NH,NH)
        self.dtype = V.dtype 
        
        # Additional attributes
        self.NH = NH        # The size of the Hamiltonian matrix
        self.basis = basis  # The NDBasis object
        self.V = V          # The PES quadrature grid
        self.KEO = T          # The KEO matrix 
        self.mass = mass    # Mass
        self.hbar = hbar    # The value of hbar.
    
    def _matvec(self, x):
        """
        Hamiltonian matrix-vector product routine
        """
        
        x = x.reshape((self.NH,))
        #
        # Compute kinetic energy operator
        #
        tx = self.KEO @ x 
        
        # 
        # Compute potential energy operator
        #
        xquad = self.basis.fbrToQuad(x,axis = 0) # xquad has shape (Nq,)
        vx = self.basis.quadToFbr(self.V * xquad) # vx has shape (NH,)
        
        return tx + vx 

class GeneralSpaceFixed(LinearOperator):
    """
    A general space-fixed frame Hamiltonian using
    mixed DVR-FBR basis sets.
    
    The kinetic energy operator (KEO) is constructed via
    the general curvilinear Laplacian
    for the coordinate system (`cs`) and the
    integration volume element :math:`\\rho` defined by the 
    basis set functions (`bases`). KEO matrix elements equal
    
    ..  math::
        
        \\int dq\\, \\rho \\Psi' (\\hat{T} \\Psi) = \\frac{\\hbar^2}{2} \\int dq\\, \\rho  (\\tilde{\\partial}_k \\Psi') G^{kl} (\\tilde{\\partial}_l \\Psi)
    
    where :math:`\\tilde{\\partial}_k = \\partial_k + \\frac{1}{2} \\tilde{\\Gamma}_k`,
    :math:`\\tilde{\\Gamma}_k = (\\partial_k \\Gamma) / \\Gamma`, 
    :math:`\\Gamma = \\rho / g^{1/2}`, and :math:`g` is the determinant
    of the coordinate system metric tensor. This form of the KEO assumes
    certain surface terms arising from integration-by-parts
    are zero. The necessary boundary conditions on the basis
    set functions to ensure this are not checked explicitly. *The user
    must use appropriate basis sets.*
    
    """
    
    def __init__(self, bases, cs, pes = None, masses = None, hbar = None):
        """

        Parameters
        ----------
        bases : list
            A list of :class:`~nitrogen.dvr.DVR` or
            :class:`~nitrogen.dvr.NDBasis` basis sets for active
            coordinates. Scalar elements will constrain the 
            corresponding coordinate to that fixed value.
        cs : CoordSys
            The coordinate system.
        pes : DFun or function, optional
            The potential energy surface, V(q). This accepts the 
            coordinates defined by `cs` as input. If None (default),
            no PES is used.
        masses : array_like, optional
            The coordinate masses. If None (default), unit masses
            are used. If `cs` is atomic, an array of length
            `cs.natoms` may be used to specify atomic masses.
        hbar : scalar, optional
            The value of :math:`\\hbar`. If None, the default value in 
            standard NITROGEN units is used (``n2.constants.hbar``).


        """
        
        
        ###################################
        # Construct a generic space-fixed 
        # Hamiltonian
        #
        
        ##########################################
        # First, construct the total direct product
        # grid
        Q = nitrogen.dvr.bases2grid(bases)
        # Q has as many axes as elements in bases, even
        # for multi-dimensional basis sets (whose quadrature grid
        # may not be a direct product structure). Fixed coordinates
        # (with bases[i] equal to a scalar) have a singleton axis
        # in Q.
        
        ##########################################
        # Parse hbar and masses
        #
        if hbar is None:
            hbar = nitrogen.constants.hbar 
        
        if masses is None:
            masses = [1.0] * cs.nX 
        elif cs.isatomic and len(masses) == cs.natoms:
            masses = np.repeat(masses, 3) 
        if len(masses) != cs.nX:
            raise ValueError("unexpected length of masses")
        
        ########################################
        # Evaluate potential energy function on 
        # quadrature grid 
        if pes is None:
            Vq = None # no PES to be used
        else:
            try: # Attempt DFun interface
                Vq = pes.f(Q, deriv = 0)[0,0]
            except:
                Vq = pes(Q) # Attempt raw function
        #
        #
        
        ########################################
        # Figure out which coordinates are active,
        # the shape of the FBR direct-product basis,
        # and which axis in the grid corresponds to 
        # each coordinate        
        axis_of_coord = [] # length = # of coord (cs.NQ)
        fbr_shape = []     # length = # of bases 
        NH = 1 
        vvar = []  # The ordered list of active coordinates 
        k = 0
        for ax,b in enumerate(bases):
            if isinstance(b, DVR): # A DVR basis is one-dimensional
                vvar.append(k)
                fbr_shape.append(b.num)
                NH *= b.num 
                axis_of_coord.append(ax) 
                k +=1 
            elif isinstance(b, NDBasis): # NDBasis is `nd` dimensional
                for i in range(b.nd):
                    vvar.append(k)
                    axis_of_coord.append(ax)
                    k += 1 
                fbr_shape.append(b.Nb)
                NH *= b.Nb
            else: # a scalar, not active 
                fbr_shape.append(1)
                axis_of_coord.append(ax)
                k += 1
        
        # FBR shape, including singleton non-active coordinates
        fbr_shape = tuple(fbr_shape)  
        
        # Check that there is at least one active coordinate        
        if len(vvar) == 0:
            raise ValueError("there must be an active coordinate!") 
        
        ########################################
        #
        # Calculate coordinate system metric functions
        #
        g = cs.Q2g(Q, deriv = 1, mode = 'simple', vvar = vvar, masses = masses)
        G,detg = nitrogen.dfun.sym2invdet(g, 1, len(vvar))
        # G and detg contain the derivative arrays for G = inv(g) and det(g)
        G = G[0] # need only value; lower triangle row-order
        #
        # Calculate gtilde_k = (d detg)/dQ_k / detg
        # Only do this for active coordinates (i.e. those in vvar) 
        gtilde = detg[1:] / detg[0]
        #
        
        ########################################
        #
        # Calculate the logarithmic derivatives of
        # the integration volume weight function 
        # defined by the basis sets. Evaluate over the 
        # quadrature grid Q.
        # (only necessary for active coordinates)
        rhotilde = [] 
        k = 0
        
        for b in bases: #
            if isinstance(b, DVR):
                # All DVR objects have unit weight function,
                # rho = 1.
                # So rhotilde_k = 0
                #
                rhotilde.append(np.zeros(Q.shape[1:])) 
                k += 1 
            
            elif isinstance(b, NDBasis):
                #
                # NDBasis objects provide their weight
                # function with the DFun wgtfun()
                # Evaluate wgtfun and its first derivatives
                # over the quadrature grid. It only takes
                # the coordinates belonging to this basis 
                # set as arguments
                #
                rho = b.wgtfun.f(Q[k:(k+b.nd)], deriv = 1) 
                # Note: using the *entire* quadrature grid is a big 
                # waste of effort because most of the arrays are the same value
                # One could slice-out the necessary coordinates and then
                # broadcast them back out to the entire quadrature grid,
                # but wgtfun is usually a simple, inexpensive function
                # so this is not a bottle-neck.
                #
                for i in range(b.nd): # for each coordinate in the basis
                    rhoi = rho[i+1][0] / rho[0][0] # calculate log. deriv.
                    rhotilde.append(rhoi) 
                    k += 1            
            # else, inactive coordinate; no entry in rhotilde.
            
        rhotilde = np.stack(rhotilde, axis = 0) # active only 
        
        # Calculate Gammatilde, the log deriv of the ratio
        # of the basis weight function and the Euclidean metric
        # of the coordinate system (g^1/2)
        #
        Gammatilde = rhotilde - 0.5 * gtilde  # active only
        
        ###################################
        #
        # Collect or construct the single
        # derivative operators for every
        # active coordinate.
        # 
        # Inactive coordinates will be included
        # in the list via None.
        #
        # For DVR objects, the derivative operator
        # is provided in the DVR representation
        # via DVR.D
        #
        # For NDBasis objects, we will construct
        # an effective operator that acts on the
        # the quadrature representation:
        #    1) it first transforms *back* to the 
        #       FBR representation, and then
        #    2) transform back to a quadrature
        #       evaluated using the derivative
        #       of the basis functions directly.
        #
        D = [] 
        for b in bases:
            if isinstance(b, DVR):
                D.append(b.D) 
            elif isinstance(b, NDBasis):
                # Evaluate the derivative of the basis functions
                # with respect to its coordinates on the basis set's
                # quadrature grid
                #
                dbas = b.basisfun.jac(b.qgrid)
                for i in range(b.nd):
                    # For each coordinate in the basis set
                    # 
                    quad2fbr = b.bas * np.sqrt(b.wgt)
                    dquad2fbr = dbas[:,i] * np.sqrt(b.wgt)
                    D.append(  dquad2fbr.T @ quad2fbr )       
            else:
                # an inactive coordinate, include None
                D.append(None)
         

        # Define the required LinearOperator attributes
        self.shape = (NH,NH)
        self.dtype = np.result_type(G, Gammatilde, Vq)
        
        # Additional attributes
        self.NH = NH        # The size of the Hamiltonian matrix
        self.bases = bases  # The basis sets
        self.Vq = Vq        # The PES quadrature grid
        self.G = G          # The inverse metric
        self.Gammatilde = Gammatilde # The pseudo-potential terms
        self.D = D          # The derivative operators 
        self.hbar = hbar    # The value of hbar.    
        self.fbr_shape = fbr_shape # The shape of the mixed DVR-FBR product basis 
        self.axis_of_coord = axis_of_coord # Grid axis of each coordinate
        
        return 
    
    def _matvec(self, x):
        """ The matrix-vector product function
        
        Input: x, 1D array
        
        """
        
        # Reshape x to a direct-product array
        # in the mixed DVR-FBR representation
        #
        x = np.reshape(x, self.fbr_shape) 
        
        # Convert x to from the mixed DVR-FBR
        # representation to the quadrature
        # representation
        xq = self._to_quad(x) 
        # Make a vector in the quadrature
        # representation to accumulate results
        yq = np.zeros_like(xq) 
        
        ##################################
        # Potential energy operator
        #
        if self.Vq is not None:
            yq += self.Vq * xq # diagonal product in quad. representation
        #
        #
    
        ##################################
        # Kinetic energy operator
        # 
        lactive = 0
        nd = len(self.D) # The number of coordinates (including inactive)
        for l in range(nd):
            # calculate dtilde_l acting on wavefunction,
            # result in the quadrature representation 
            
            if self.D[l] is None:
                continue # an in-active coordinate, no derivative to compute
            
            # Apply the derivative matrix to the appropriate index
            dl_x = dvrops.opO(xq, self.D[l], self.axis_of_coord[l]) 
            #
            # dtilde_l is the sum of the derivative 
            # and one-half Gammatilde_l
            #
            dtilde_l = dl_x + 0.5 * self.Gammatilde[lactive] * xq
            
            
            kactive = 0 
            for k in range(nd):
                
                if self.D[k] is None:
                    continue # inactive
                
                # Get the packed-storage index for 
                # G^{kactive, lactive}
                idx = nitrogen.linalg.packed.IJ2k(kactive,lactive)
                
                # Apply G^kl 
                Gkl_dl = self.G[idx] * dtilde_l
                
                # Now finish with (dtilde_k)
                # and put results in quadrature representation
                #
                # include the final factor of -hbar**2 / 2
                yq += self.hbar**2 * 0.25 * self.Gammatilde[kactive] * Gkl_dl 
                yq += self.hbar**2 * 0.50 * dvrops.opO(Gkl_dl, self.D[k].T, self.axis_of_coord[k]) 
                
                kactive += 1 
                
            lactive += 1
        
        # yq contains the complete 
        # matrix-vector result in the quadrature representation
        #
        # Convert this back to the mixed FBR-DVR representation
        # and reshape back to a 1-D vector
        #
        y = self._to_fbr(yq)
        return np.reshape(y, (-1,))
        
    def _to_quad(self, x, force_copy = False):
        
        """ convert the mixed FBR representation
        array to the quadrature array
        """
        for i,b in enumerate(self.bases):
            # i**th axis
            if isinstance(b, NDBasis):
                x = b.fbrToQuad(x, i)
        
        if force_copy:
            x = x.copy() 

        return x 

    def _to_fbr(self, x, force_copy = False):
        
        """ convert the quadrature array to 
        the mixed FBR representation array 
        """
        for i,b in enumerate(self.bases):
            # i**th axis
            if isinstance(b, NDBasis):
                x = b.quadToFbr(x, i)
        
        if force_copy:
            x = x.copy() 

        return x 
    
class Collinear(LinearOperator):
    """
    A generalized radial Hamiltonian for collinear 
    configurations for total angular momentum :math:`J`.
    
    The vibrational kinetic energy operator (KEO) is constructed via
    the general curvilinear Laplacian
    for the coordinate system (`cs`) and the
    vibrational integration volume element :math:`\\rho` defined by the 
    basis set functions (`bases`). KEO matrix elements equal
    
    ..  math::
        
        \\int dq\\, \\rho \\Psi' (\\hat{T} \\Psi) = \\frac{\\hbar^2}{2} \\int dq\\, \\rho  (\\tilde{\\partial}_k \\Psi') G^{kl} (\\tilde{\\partial}_l \\Psi)
    
    where :math:`\\tilde{\\partial}_k = \\partial_k + \\frac{1}{2} \\tilde{\\Gamma}_k`,
    :math:`\\tilde{\\Gamma}_k = (\\partial_k \\Gamma) / \\Gamma`, 
    :math:`\\Gamma = \\rho / (g_\\text{vib}^{1/2} I)`, 
    :math:`g_\\text{vib}` is the determinant
    of the vibrational block of the coordinate system metric tensor, and
    :math:`I` is the moment of inertia. This form of the KEO assumes
    certain surface terms arising from integration-by-parts
    are zero. The necessary boundary conditions on the basis
    set functions to ensure this are not checked explicitly. *The user
    must use appropriate basis sets.*
    
    The total potential energy surface contains a centrifugal contribution
    :math:`V_J = (\\hbar^2/2I) J(J+1)`, where :math:`J` is the total 
    angular momentum quantum number.
    
    """
    
    def __init__(self, bases, cs, pes = None, masses = None, J = 0, hbar = None):
        """

        Parameters
        ----------
        bases : list
            A list of :class:`~nitrogen.dvr.DVR` or
            :class:`~nitrogen.dvr.NDBasis` basis sets for active
            coordinates. Scalar elements will constrain the 
            corresponding coordinate to that fixed value.
        cs : CoordSys
            The coordinate system.
        pes : DFun or function, optional
            The potential energy surface, V(q). This accepts the 
            coordinates defined by `cs` as input. If None (default),
            no PES is used.
        masses : array_like, optional
            The atomic masses. If None (default), unit masses
            are used. 
        J : int, optional
            The total angular momentum quantum number :math:`J`. The
            default value is 0.
        hbar : scalar, optional
            The value of :math:`\\hbar`. If None, the default value in 
            standard NITROGEN units is used (``n2.constants.hbar``).

        Notes
        -----
        The coordinate system must position all atoms along the body-fixed
        :math:`z`-axis (accounting for the fixed values of constrained
        coordinates). 
        
        """
        
        ###################################
        # Construct a generic space-fixed 
        # Hamiltonian
        #
        
        ##########################################
        # First, construct the total direct product
        # grid
        Q = nitrogen.dvr.bases2grid(bases)
        # Q has as many axes as elements in bases, even
        # for multi-dimensional basis sets (whose quadrature grid
        # may not be a direct product structure). Fixed coordinates
        # (with bases[i] equal to a scalar) have a singleton axis
        # in Q.
        
        ##########################################
        # Parse hbar and masses
        #
        if hbar is None:
            hbar = nitrogen.constants.hbar 
        
        if not cs.isatomic:
            raise ValueError("The coordinate system must be atomic.")
        
        if masses is None:
            masses = [1.0] * cs.natoms

        if len(masses) != cs.natoms:
            raise ValueError("unexpected length of masses")
        
        ########################################
        # Evaluate potential energy function on 
        # quadrature grid 
        if pes is None:
            Vq = None # no PES to be used
        else:
            try: # Attempt DFun interface
                Vq = pes.f(Q, deriv = 0)[0,0]
            except:
                Vq = pes(Q) # Attempt raw function
        #
        #
        
        ########################################
        # Figure out which coordinates are active,
        # the shape of the FBR direct-product basis,
        # and which axis in the grid corresponds to 
        # each coordinate        
        axis_of_coord = [] # length = # of coord (cs.NQ)
        fbr_shape = []     # length = # of bases 
        NH = 1 
        vvar = []  # The ordered list of active coordinates 
        k = 0
        for ax,b in enumerate(bases):
            if isinstance(b, DVR): # A DVR basis is one-dimensional
                vvar.append(k)
                fbr_shape.append(b.num)
                NH *= b.num 
                axis_of_coord.append(ax) 
                k +=1 
            elif isinstance(b, NDBasis): # NDBasis is `nd` dimensional
                for i in range(b.nd):
                    vvar.append(k)
                    axis_of_coord.append(ax)
                    k += 1 
                fbr_shape.append(b.Nb)
                NH *= b.Nb
            else: # a scalar, not active 
                fbr_shape.append(1)
                axis_of_coord.append(ax)
                k += 1
        
        # FBR shape, including singleton non-active coordinates
        fbr_shape = tuple(fbr_shape)  
        
        # Check that there is at least one active coordinate        
        if len(vvar) == 0:
            raise ValueError("there must be an active coordinate!") 
        
        ########################################
        #
        # Calculate coordinate system metric functions
        # 
        # Given a collinear geometry (which is assumed)
        # g factors into a pure vibrational block
        # and a rotational block. We only need the 
        # (single) moment of inertia, I, from the rotational
        # block. This equals the g_xx = g_yy matrix element
        # (g_zz = 0). We will ultimately need the
        # logarithmic derivatives of the quantity
        # gvib^1/2 * I = (gvib * I^2)^1/2. We see that
        # the quantity in gvib * I^2 is just the determinant
        # of the full rovib g tensor excluding the z-axis
        # row and column.
        # 
        # So first, we calculate this truncated g metric
        # using rvar = 'xy'
        g = cs.Q2g(Q, deriv = 1, mode = 'bodyframe', 
                   vvar = vvar, rvar = 'xy', masses = masses)
        #
        # And then calculate its inverse and determinant
        #
        G,detg = nitrogen.dfun.sym2invdet(g, 1, len(vvar))
        #
        # G and detg contain the derivative arrays for G = inv(g) and det(g)
        # where det(g) = det(gvib) I**2
        # 
        # We only need to keep the vibrational block of G.
        # In packed storage, this is the first nv*(nv+1)/2 
        # elements (where nv = len(vvar))
        #
        nv = len(vvar)
        nG = (nv*(nv+1))//2 
        G = G[0][:nG] # need only value; lower triangle row-order
        #
        # Calculate the log. deriv. of gvib * I**2
        # Only do this for active coordinates (i.e. those in vvar) 
        gI2tilde = detg[1:] / detg[0]
        #
        # We also need the moment of inertia for the 
        # centrifugal potential
        if J == 0:
            Vc = None # No centrifugal potential
        else:
            nI = nG + nv
            I = g[0][nI] # the moment of inertia (over quadrature grid)
            Vc = (J * (J+1) * hbar**2 / 2.0)  / I
        
        
        ########################################
        #
        # Calculate the logarithmic derivatives of
        # the integration volume weight function 
        # defined by the basis sets. Evaluate over the 
        # quadrature grid Q.
        # (only necessary for active coordinates)
        rhotilde = [] 
        k = 0
        
        for b in bases: #
            if isinstance(b, DVR):
                # All DVR objects have unit weight function,
                # rho = 1.
                # So rhotilde_k = 0
                #
                rhotilde.append(np.zeros(Q.shape[1:])) 
                k += 1 
            
            elif isinstance(b, NDBasis):
                #
                # NDBasis objects provide their weight
                # function with the DFun wgtfun()
                # Evaluate wgtfun and its first derivatives
                # over the quadrature grid. It only takes
                # the coordinates belonging to this basis 
                # set as arguments
                #
                rho = b.wgtfun.f(Q[k:(k+b.nd)], deriv = 1) 
                # Note: using the *entire* quadrature grid is a big 
                # waste of effort because most of the arrays are the same value
                # One could slice-out the necessary coordinates and then
                # broadcast them back out to the entire quadrature grid,
                # but wgtfun is usually a simple, inexpensive function
                # so this is not a bottle-neck.
                #
                for i in range(b.nd): # for each coordinate in the basis
                    rhoi = rho[i+1][0] / rho[0][0] # calculate log. deriv.
                    rhotilde.append(rhoi) 
                    k += 1            
            # else, inactive coordinate; no entry in rhotilde.
            
        rhotilde = np.stack(rhotilde, axis = 0) # active only 
        
        # Calculate Gammatilde, the log deriv of the ratio
        # of the basis weight function and (gvib * I**2) ** 1/2
        #
        Gammatilde = rhotilde - 0.5 * gI2tilde  # active only
        
        ###################################
        #
        # Collect or construct the single
        # derivative operators for every
        # active coordinate.
        # 
        # Inactive coordinates will be included
        # in the list via None.
        #
        # For DVR objects, the derivative operator
        # is provided in the DVR representation
        # via DVR.D
        #
        # For NDBasis objects, we will construct
        # an effective operator that acts on the
        # the quadrature representation:
        #    1) it first transforms *back* to the 
        #       FBR representation, and then
        #    2) transform back to a quadrature
        #       evaluated using the derivative
        #       of the basis functions directly.
        #
        D = [] 
        for b in bases:
            if isinstance(b, DVR):
                D.append(b.D) 
            elif isinstance(b, NDBasis):
                # Evaluate the derivative of the basis functions
                # with respect to its coordinates on the basis set's
                # quadrature grid
                #
                dbas = b.basisfun.jac(b.qgrid)
                for i in range(b.nd):
                    # For each coordinate in the basis set
                    # 
                    quad2fbr = b.bas * np.sqrt(b.wgt)
                    dquad2fbr = dbas[:,i] * np.sqrt(b.wgt)
                    D.append(  dquad2fbr.T @ quad2fbr )       
            else:
                # an inactive coordinate, include None
                D.append(None)
         

        # Define the required LinearOperator attributes
        self.shape = (NH,NH)
        self.dtype = np.result_type(G, Gammatilde, Vq)
        
        # Additional attributes
        self.NH = NH        # The size of the Hamiltonian matrix
        self.bases = bases  # The basis sets
        self.Vq = Vq        # The PES quadrature grid
        self.Vc = Vc        # The centrifugal potential 
        self.G = G          # The inverse vibrational metric
        self.Gammatilde = Gammatilde # The pseudo-potential terms
        self.D = D          # The derivative operators 
        self.hbar = hbar    # The value of hbar.    
        self.fbr_shape = fbr_shape # The shape of the mixed DVR-FBR product basis 
        self.axis_of_coord = axis_of_coord # Grid axis of each coordinate
        self.J = J          # The total angular momentum
        
        return 
    
    def _matvec(self, x):
        """ The matrix-vector product function
        
        Input: x, 1D array
        
        """
        
        # Reshape x to a direct-product array
        # in the mixed DVR-FBR representation
        #
        x = np.reshape(x, self.fbr_shape) 
        
        # Convert x to from the mixed DVR-FBR
        # representation to the quadrature
        # representation
        xq = self._to_quad(x) 
        # Make a vector in the quadrature
        # representation to accumulate results
        yq = np.zeros_like(xq) 
        
        ##################################
        # Potential energy operator
        #
        if self.Vq is not None:
            yq += self.Vq * xq # diagonal product in quad. representation
        #
        if self.Vc is not None:
            yq += self.Vc * xq # centrifugal potential
        #
        #
        
    
        ##################################
        # Kinetic energy operator
        # 
        lactive = 0
        nd = len(self.D) # The number of coordinates (including inactive)
        for l in range(nd):
            # calculate dtilde_l acting on wavefunction,
            # result in the quadrature representation 
            
            if self.D[l] is None:
                continue # an in-active coordinate, no derivative to compute
            
            # Apply the derivative matrix to the appropriate index
            dl_x = dvrops.opO(xq, self.D[l], self.axis_of_coord[l]) 
            #
            # dtilde_l is the sum of the derivative 
            # and one-half Gammatilde_l
            #
            dtilde_l = dl_x + 0.5 * self.Gammatilde[lactive] * xq
            
            
            kactive = 0 
            for k in range(nd):
                
                if self.D[k] is None:
                    continue # inactive
                
                # Get the packed-storage index for 
                # G^{kactive, lactive}
                idx = nitrogen.linalg.packed.IJ2k(kactive,lactive)
                
                # Apply G^kl 
                Gkl_dl = self.G[idx] * dtilde_l
                
                # Now finish with (dtilde_k)
                # and put results in quadrature representation
                #
                # include the final factor of -hbar**2 / 2
                yq += self.hbar**2 * 0.25 * self.Gammatilde[kactive] * Gkl_dl 
                yq += self.hbar**2 * 0.50 * dvrops.opO(Gkl_dl, self.D[k].T, self.axis_of_coord[k]) 
                
                kactive += 1 
                
            lactive += 1
        
        # yq contains the complete 
        # matrix-vector result in the quadrature representation
        #
        # Convert this back to the mixed FBR-DVR representation
        # and reshape back to a 1-D vector
        #
        y = self._to_fbr(yq)
        return np.reshape(y, (-1,))
        
    def _to_quad(self, x, force_copy = False):
        
        """ convert the mixed FBR representation
        array to the quadrature array
        """
        for i,b in enumerate(self.bases):
            # i**th axis
            if isinstance(b, NDBasis):
                x = b.fbrToQuad(x, i)
        
        if force_copy:
            x = x.copy() 

        return x 

    def _to_fbr(self, x, force_copy = False):
        
        """ convert the quadrature array to 
        the mixed FBR representation array 
        """
        for i,b in enumerate(self.bases):
            # i**th axis
            if isinstance(b, NDBasis):
                x = b.quadToFbr(x, i)
        
        if force_copy:
            x = x.copy() 

        return x 