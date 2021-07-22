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
        rhotilde = nitrogen.dvr.calcRhoLogD(bases, Q)
        
        # Calculate Gammatilde, the log deriv of the ratio
        # of the basis weight function and the Euclidean metric
        # of the coordinate system (g^1/2)
        #
        Gammatilde = rhotilde - 0.5 * gtilde  # active only
        
        ###################################
        #
        # Collect or construct the single
        # derivative operators for every
        # coordinate.
        D = nitrogen.dvr.collectBasisD(bases)
         

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
        xq = nitrogen.dvr._to_quad(self.bases, x)
        
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
        y = nitrogen.dvr._to_fbr(self.bases, yq)
        return np.reshape(y, (-1,))
        
    
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
        # Construct a body-frame Hamiltonian
        # for a collinear molecule.
        
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
        rhotilde = nitrogen.dvr.calcRhoLogD(bases, Q)
        
        # Calculate Gammatilde, the log deriv of the ratio
        # of the basis weight function and (gvib * I**2) ** 1/2
        #
        Gammatilde = rhotilde - 0.5 * gI2tilde  # active only
        
        ###################################
        #
        # Collect or construct the single
        # derivative operators for every
        # vibrational coordinate.
        # 
        D = nitrogen.dvr.collectBasisD(bases)
         

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
        xq = nitrogen.dvr._to_quad(self.bases, x)
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
        y = nitrogen.dvr._to_fbr(self.bases, yq)
        
        return np.reshape(y, (-1,))
        
class NonLinear(LinearOperator):
    """
    A general curvilinear rovibrational Hamiltonian
    for non-linear molecules with total angular momentum :math:`J`.
    
    The vibrational kinetic energy operator (KEO) is constructed via
    the general curvilinear Laplacian for the body-fixed coordinate system 
    (`cs`) and the vibrational integration volume element :math:`\\rho` defined by the 
    basis set functions (`bases`). KEO matrix elements equal
    
    ..  math::
        
        \\langle r', \\Psi' \\vert \\hat{T} \\vert r, \\Psi \\rangle &= \\frac{\\hbar^2}{2} \\langle r' \\vert r \\rangle \\int dq\\, \\rho  (\\tilde{\\partial}_k \\Psi') G^{kl} (\\tilde{\\partial}_l \\Psi) 
    
        &\\qquad + \\frac{1}{2} \\langle r' \\vert J_\\alpha J_\\beta \\vert r \\rangle \\int dq\\, \\rho G^{\\alpha \\beta} \\Psi' \\Psi 
        
        &\\qquad - \\frac{\\hbar}{2} \\langle r' \\vert i J_\\alpha \\vert r \\rangle \\int dq\\, \\rho G^{\\alpha k} \\left[\\Psi'(\\tilde{\\partial}_k \\Psi)- (\\tilde{\\partial}_k \\Psi') \\Psi\\right]
    
    where :math:`\\tilde{\\partial}_k = \\partial_k + \\frac{1}{2} \\tilde{\\Gamma}_k`,
    :math:`\\tilde{\\Gamma}_k = (\\partial_k \\Gamma) / \\Gamma`, 
    :math:`\\Gamma = \\rho / (g^{1/2})`, 
    :math:`g` is the determinant
    of the full ro-vibrational metric tensor. This form of the KEO assumes
    certain surface terms arising from integration-by-parts
    are zero. The necessary boundary conditions on the basis
    set functions to ensure this are not checked explicitly. *The user
    must use appropriate basis sets.*
    
    """
    
    def __init__(self, bases, cs, pes = None, masses = None, J = 0, hbar = None,
                 Vmax = None, Vmin = None):
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
        Vmax,Vmin : scalar, optional
            Potential energy cut-off thresholds. The default is None.
        """
        
        ###################################
        # Construct a generic body-fixed
        # Hamiltonian for a non-linear molecule.
        
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
                
            # Apply PES min and max cutoffs
            if Vmax is not None:
                Vq[Vq > Vmax] = Vmax 
            if Vmin is not None:
                Vq[Vq < Vmin] = Vmin 
                
        #
        #
        
        ########################################
        # Figure out which coordinates are active,
        # the shape of the FBR direct-product basis,
        # and which axis in the grid corresponds to 
        # each coordinate        
        axis_of_coord = [] # length = # of coord (cs.NQ)
        fbr_shape = []     # length = # of bases 
        NV = 1 
        vvar = []  # The ordered list of active coordinates 
        k = 0
        for ax,b in enumerate(bases):
            if isinstance(b, DVR): # A DVR basis is one-dimensional
                vvar.append(k)
                fbr_shape.append(b.num)
                NV *= b.num 
                axis_of_coord.append(ax) 
                k +=1 
            elif isinstance(b, NDBasis): # NDBasis is `nd` dimensional
                for i in range(b.nd):
                    vvar.append(k)
                    axis_of_coord.append(ax)
                    k += 1 
                fbr_shape.append(b.Nb)
                NV *= b.Nb
            else: # a scalar, not active 
                fbr_shape.append(1)
                axis_of_coord.append(ax)
                k += 1
        
        # FBR shape, including singleton non-active coordinates
        fbr_shape = tuple(fbr_shape)  
        
        NJ = 2*J + 1 # The number of rotational wavefunctions 
        NH = NJ * NV # The total dimension of the Hamiltonian 
        
        # Check that there is at least one active coordinate        
        if len(vvar) == 0:
            raise ValueError("there must be an active coordinate!") 
        
        ########################################
        #
        # Calculate coordinate system metric functions
        # 
        g = cs.Q2g(Q, deriv = 1, mode = 'bodyframe', 
                   vvar = vvar, rvar = 'xyz', masses = masses)
        #
        # And then calculate its inverse and determinant
        #
        G,detg = nitrogen.dfun.sym2invdet(g, 1, len(vvar))
        #
        # G and detg contain the derivative arrays for G = inv(g) and det(g)
        # 
        # If J = 0, then we only need to keep the vibrational block of G.
        # In packed storage, this is the first nv*(nv+1)/2 
        # elements (where nv = len(vvar))
        #
        if J == 0:
            nv = len(vvar)
            nG = (nv*(nv+1))//2 
            G = G[0][:nG] # need only value; lower triangle row-order
        else:
            G = G[0]      # keep all elements
        
        #
        # Calculate the log. deriv. of g
        gtilde = detg[1:] / detg[0]
        #
        
        ########################################
        #
        # Calculate the logarithmic derivatives of
        # the integration volume weight function 
        # defined by the basis sets. Evaluate over the 
        # quadrature grid Q.
        # (only necessary for active coordinates)
        rhotilde = nitrogen.dvr.calcRhoLogD(bases, Q)
        
        # Calculate Gammatilde, the log deriv of the ratio
        # of the basis weight function rho and g**1/2
        #
        Gammatilde = rhotilde - 0.5 * gtilde  # active only
        
        ###################################
        #
        # Collect or construct the single
        # derivative operators for every
        # vibrational coordinate.
        # 
        D = nitrogen.dvr.collectBasisD(bases)
         
        ####################################
        #
        # Construct the angular momentum operators 
        # in the real-Wang representation
        # (i.e. matrix elements of i*J are purely real)
        iJ = nitrogen.angmom.iJbf_wr(J)      # iJ = (x,y,z)
        iJiJ = nitrogen.angmom.iJiJbf_wr(J)  # iJiJ[a][b] = [a,b]_+ anticommutator

        # Define the required LinearOperator attributes
        self.shape = (NH,NH)
        self.dtype = np.result_type(G, Gammatilde, Vq)
        
        # Additional attributes
        self.NH = NH        # The size of the Hamiltonian matrix
        self.bases = bases  # The basis sets
        self.Vq = Vq        # The PES quadrature grid
        self.G = G          # The inverse vibrational metric
        self.Gammatilde = Gammatilde # The pseudo-potential terms
        self.D = D          # The derivative operators 
        self.hbar = hbar    # The value of hbar.    
        self.fbr_shape = fbr_shape # The shape of the mixed DVR-FBR product basis 
        self.axis_of_coord = axis_of_coord # Grid axis of each coordinate
        self.J = J          # The total angular momentum
        self.iJ = iJ        # The angular momentum operators
        self.iJiJ = iJiJ    #  " " 
        self.nact = len(vvar) # The number of active coordinates 
        
        return  
    
    def _matvec(self, x):
        """ The matrix-vector product function
        
        Input: x, 1D array
        
        """
        
        # Reshape x to a direct-product array
        # in the mixed DVR-FBR representation
        # The first axis spans the rotational wavefunctions,
        # and the remaining span the vibrational mixed DVR-FBR
        # grid, which has shape self.fbr_shape
        #
        J = self.J 
        NJ = 2*J + 1 
        nact = self.nact # The number of active coordinates 
        hbar = self.hbar # hbar 
        
        x = np.reshape(x, (NJ,) + self.fbr_shape) 
        
        # Convert x to from the mixed DVR-FBR
        # representation to the quadrature
        # representation
        # (Prepend the list of bases with a dummy element,
        #  so that the rotational index is left unchanged)
        xq = nitrogen.dvr._to_quad([None] + self.bases, x)
        # Make a vector in the quadrature
        # representation to accumulate results
        yq = np.zeros_like(xq) 
        
        ##################################
        # Potential energy operator
        #
        if self.Vq is not None:
            # The PES is diagonal in the rotational index
            for r in range(NJ):
                yq[r] += self.Vq * xq[r]
        #
        #
        
        ##################################
        # Kinetic energy operator
        # 
        #
        # 1) Pure vibrational kinetic energy
        #    Diagonal in rotational index
        
        nd = len(self.D) # The number of coordinates (including inactive)
        for r in range(NJ):
            # Rotational block `r`
            lactive = 0
            for l in range(nd):
                # calculate dtilde_l acting on wavefunction,
                # result in the quadrature representation 
                
                if self.D[l] is None:
                    continue # an in-active coordinate, no derivative to compute
                
                # Apply the derivative matrix to the appropriate index
                dl_x = dvrops.opO(xq[r], self.D[l], self.axis_of_coord[l]) 
                #
                # dtilde_l is the sum of the derivative 
                # and one-half Gammatilde_l
                #
                dtilde_l = dl_x + 0.5 * self.Gammatilde[lactive] * xq[r]
                
                
                kactive = 0 
                for k in range(nd):
                    
                    if self.D[k] is None:
                        continue # inactive
                    
                    # Get the packed-storage index for 
                    # G^{kactive, lactive}
                    kl_idx = nitrogen.linalg.packed.IJ2k(kactive,lactive)
                    Gkl = self.G[kl_idx]
                    
                    # Apply G^kl 
                    Gkl_dl = Gkl * dtilde_l
                    
                    # Now finish with (dtilde_k)
                    # and put results in quadrature representation
                    #
                    # include the final factor of -hbar**2 / 2
                    yq[r] += hbar**2 * 0.25 * self.Gammatilde[kactive] * Gkl_dl 
                    yq[r] += hbar**2 * 0.50 * dvrops.opO(Gkl_dl, self.D[k].T, self.axis_of_coord[k]) 
                    
                    kactive += 1
                    
                lactive += 1
        #
        if J > 0:
            # Rotational and ro-vibrational terms are zero unless
            # J > 0
            #
            # 2) Pure rotational kinetic energy
            #
            #  -hbar**2/4  *  [iJa/hbar, iJb/hbar]_+  *  G^ab 
            #
            for a in range(3):
                for b in range(3):
                    # Because both [iJa,iJb]_+ and G^ab
                    # are symmetric with a <--> b, we only need 
                    # to loop over half
                    if (b > a):
                        continue 
                    if b == a:
                        symfactor = 1.0  # on the diagonal, count once!
                    else: # b < a 
                        symfactor = 2.0  # count both (b,a) and (a,b)
                    
                    # G^ab term
                    ab_idx = nitrogen.linalg.packed.IJ2k(nact + a, nact + b) 
                    Gab = self.G[ab_idx] 
                            
                    for rp in range(NJ):
                        for r in range(NJ):
                            # <rp | ... | r > rotational block
                            rot_me = self.iJiJ[a][b][rp,r] # the rotational matrix element
                            
                            if rot_me == 0:
                                continue # a zero rotational matrix element
                            
                            # otherwise, add contribution from
                            # effective inverse inertia tensor
                            yq[rp] += (symfactor * rot_me * (-hbar**2) * 0.25) * (Gab * xq[r])
                            
            #
            # 3) Rotation-vibration coupling
            #
            # -hbar**2 / 2 iJa/hbar * G^ak [psi' (dtildek psi) - (dtildek psi') psi]
            
            for a in range(3):
                for rp in range(NJ):
                    for r in range(NJ):
                        rot_me = self.iJ[a][rp,r] # the rotational matrix element  
                        if rot_me == 0:
                            continue 
                        
                        kactive = 0 
                        for k in range(nd):
                            #
                            # Vibrational coordinate k
                            #
                            if self.D[k] is None:
                                    continue # an in-active coordinate, no derivative to compute
                            
                            # calculate index of G^ak
                            ak_idx = nitrogen.linalg.packed.IJ2k(nact + a, kactive)
                            Gak = self.G[ak_idx] 
                            
                            # First, do the psi' (dtilde_k psi) term
                            dk_x = dvrops.opO(xq[r], self.D[k], self.axis_of_coord[k]) 
                            dtilde_k = dk_x + 0.5 * self.Gammatilde[kactive] * xq[r]
                            yq[rp] += (rot_me * (-hbar**2) * 0.50) * Gak * dtilde_k
                            
                            # Now, do the -(dtilde_k psi') * psi term
                            yq[rp] += (rot_me * (+hbar**2) * 0.25) * self.Gammatilde[kactive] * Gak * xq[r] 
                            yq[rp] += (rot_me * (+hbar**2) * 0.50) * dvrops.opO(Gak * xq[r], self.D[k].T, self.axis_of_coord[k])
                            
                            kactive += 1
        
        # yq contains the complete 
        # matrix-vector result in the quadrature representation
        #
        # Convert this back to the mixed FBR-DVR representation
        # and reshape back to a 1-D vector
        # (Prepend a dummy basis to keep rotational index unchanged)
        y = nitrogen.dvr._to_fbr([None] + self.bases, yq)
        
        return np.reshape(y, (-1,))
    
    
    
class AzimuthalLinear(LinearOperator):
    """
    A  Hamiltonian for linear molecules
    
    """
    
    def __init__(self, bases, cs, azimuth, pes = None, masses = None, J = 0, hbar = None,
                 Vmax = None, Vmin = None, Voffset = None):
        
        # For each basis, get the azimuthal quantum number
        # list 
        
        if len(azimuth) != len(bases):
            raise ValueError("azimuth must be same length as bases")
        
        # There should be one entry of Ellipsis in the azimuth list
        n = 0
        ellipsis_idx = 0 # The basis index for the Ellipsis factor
        for i,entry in enumerate(azimuth):
            if entry is Ellipsis:
                n += 1
                ellipsis_idx = i
        if n != 1:
            raise ValueError("azimuth must contain exactly one Ellipsis")
        
        # bases[i] must be a callable 
        if not callable(bases[ellipsis_idx]):
            raise ValueError("The ... bases entry must be callable")
        
        az_m = []  # Entry for each basis factor
        az_U = []  # Unitary transforms that convert from azimuthal 
                   # representation to mixed DVR-FBR
                   # (entries of None indicate identity)
                   
        # The signed-k (i.e. Condon-Shortley representation) rotational
        # basis functions are first
        #
        az_m.append(np.arange(-J,J+1))  # signed-k rotational factor
        az_U.append(None) #  Already in azimuthal representation 
        
        min_m = -J  # Keep track of Ellipsis quantum number range
        max_m = J 
        
        for i in range(len(bases)):
            if azimuth[i] is None: # This is not declared an azimuthal coordinate
                try:
                    az_m.append(np.array([0] * bases[i].num))
                except:
                    try:
                        az_m.append(np.array([0] * bases[i].Nb))
                    except:
                        az_m.append(np.array([0]))
                az_U.append(None) 
            elif azimuth[i] is Ellipsis:
                # This is the callable basis
                az_m.append(Ellipsis) # Place holder 
                az_U.append(None)     # Azimuthal representation is implied
            else:
                # This basis has been declared azimuthal
                # Attempt to find its azimuthal representation
                #
                # If it has a D (derivative) operator, use that
                try: 
                    D = bases[i].D
                except AttributeError:
                    # there is no D, attempt quadrature instead 
                    try:
                        
                        qgrid = bases[i].qgrid # the quadrature grid
                        wgt = bases[i].wgt # the quadrature weights over qgrid 
                    
                        # calculate derivatives w.r.t. the azimuthal coordinate only
                        dbas = bases[i].basisfun.f(qgrid,deriv=1, var = [azimuth[i][0]])
                        
                        f = dbas[0]
                        df = dbas[1] # d/dr 
                        
                        D = (f * wgt) @ (df.T)  
                        
                    except:
                        raise AttributeError(f"cannot get azimuthal representation for bases[{i:d}]")

                w,u = np.linalg.eigh(-1j * D) 
                w *= azimuth[i][1] # correct for right or left-handed sense
                # u transforms a vector in the the azimuthal representation
                # to the original FBR or DVR representation
                
                # in principle, w should be exact integer
                # assuming a proper basis has been chosen
                # (i.e. one that is closed under rotations of the azimuthal 
                #  coordinate) 
                w_int = np.rint(w).astype(np.int32) # round to nearest integer
                if np.max(np.absolute(w_int - w)) > 1e-10:
                    print("Warning: azimuthal quantum numbers are not quite integer!" + \
                          " (or perhaps were rounded unexpectedly)")
                
                az_m.append(w_int) 
                az_U.append(u) 
                
                min_m -= np.max(w_int)
                max_m -= np.min(w_int)
                
        # Calculate the conjugate tranpose of each az_U
        # This will transform from mixed DVR/FBR to azimuthal representation
        #
        az_UH = []
        for U in az_U:
            if U is None:
                az_UH.append(None)
            else:
                az_UH.append(U.conj().T) # conjugate transpose  
        
        #
        # We now need to determine what azimuthal quantum number range
        # is necessary for the Ellipsis basis, which takes all the slack
        #
        # m = k - sum(all other m's)
        #
        # The largest value of m is max(k) - sum( min(other m's) )
        # The smallest value is min(k) - sum(max(other m's)) 
        #
        # These have been kept track of with min_m and max_m above
        ellipsis_range = np.arange(min_m, max_m + 1)
        #
        # For each m in this range, bases[ellipsis_idx](m) returns
        # a basis specification. This may be a scalar, a DVR, or an NDBasis, etc.
        #
        # It is possible that a basis cannot be formed
        ellipsis_bases = []
        actual_ellipsis_range = []
        for m in ellipsis_range:
            try:
                b = bases[ellipsis_idx](m) 
                ellipsis_bases.append(b)
                actual_ellipsis_range.append(m)
            except:
                print(f"Note: azimuthal basis m = {m:d} is being skipped.")
    
        ellipsis_range = actual_ellipsis_range 
        if len(ellipsis_range) == 0:
            raise ValueError("No Ellipsis bases could be formed!")
        
        bfirst = ellipsis_bases[0] 
        size_of_ellipsis_m = []
        #
        # Check for consistency among the basis objects for every possible
        # value of m, and collect the size of the sub-basis for each m
        if np.isscalar(bfirst):
            # A scalar, all values must be equal 
            for i in range(len(ellipsis_bases)):
                if bfirst != ellipsis_bases[i]:
                    raise ValueError("For scalar Ellipsis basis, all values must be equal.")
                size_of_ellipsis_m.append(1)
        elif isinstance(bfirst, nitrogen.dvr.DVR):
            # A DVR
            for i in  range(len(ellipsis_bases)):
                if bfirst.basis != ellipsis_bases[i].basis:
                    raise ValueError("All Ellipsis bases must have the same DVR basis type.")
                if not np.all(bfirst.grid == ellipsis_bases[i].grid):
                    raise ValueError("All Ellipsis bases must have the same DVR grid.")
                size_of_ellipsis_m.append(ellipsis_bases[i].num)
        elif isinstance(bfirst, nitrogen.dvr.NDBasis):
            # An NDBasis
            for i in  range(len(ellipsis_bases)):
                if not np.all(bfirst.qgrid == ellipsis_bases[i].qgrid):
                    raise ValueError("All Ellipsis bases must have the same quadrature grid.")
                if not np.all(bfirst.wgt == ellipsis_bases[i].wgt):
                    raise ValueError("All Ellipsis bases must have the same quadrature weights.")
                size_of_ellipsis_m.append(ellipsis_bases[i].Nb)
        else: 
            # An unrecognized type
            raise ValueError("Unrecognized basis type for Ellipsis basis.")
        #
        #
        
        Nellip = sum(size_of_ellipsis_m) # The size of the concatenated ellipsis bases
        
        az_m[ellipsis_idx + 1] = np.repeat(ellipsis_range, np.array(size_of_ellipsis_m))
        
        az_grids = np.meshgrid(*az_m, indexing = 'ij') 
        
        total = az_grids[0] - sum(az_grids[1:]) # k - sum(m)
        sing_val_mask = (total == 0) 
        
        svm_1d = np.reshape(sing_val_mask, (-1,)) 
        NH = np.count_nonzero(svm_1d)  # the number of single-valued functions
        
        ########################################
        # Figure out which coordinates are active,
        # the shape of the FBR direct-product basis,
        # and which axis in the grid corresponds to 
        # each coordinate        
        axis_of_coord = [] # length = # of coord (cs.nQ)
        fbr_shape = []     # length = # of bases 
        NV = 1 
        vvar = []  # The ordered list of active coordinates 
        k = 0
        coord_k_is_ellip_coord = [None for i in range(cs.nQ)]
        for ax,b in enumerate(bases):
            
            # The ellipsis bases are treated differently 
            if ax == ellipsis_idx: 
                b = bfirst 
                if isinstance(b, DVR): # A DVR basis is one-dimensional
                    vvar.append(k)
                    #fbr_shape.append(b.num)
                    #NV *= b.num 
                    axis_of_coord.append(ax) 
                    coord_k_is_ellip_coord[k] = 0
                    k +=1 
                elif isinstance(b, NDBasis): # NDBasis is `nd` dimensional
                    for i in range(b.nd):
                        vvar.append(k)
                        axis_of_coord.append(ax)
                        coord_k_is_ellip_coord[k] = i
                        k += 1 
                    #fbr_shape.append(b.Nb)
                    #NV *= b.Nb
                else: # a scalar, not active 
                    #fbr_shape.append(1)
                    axis_of_coord.append(ax)
                    coord_k_is_ellip_coord[k] = 0
                    k += 1  
                fbr_shape.append(Nellip) # The size of this axis is the concatenated number of Ellipsis functions
                NV *= Nellip 
            else:
                #
                # A normal basis factor
                #
                if isinstance(b, DVR): # A DVR basis is one-dimensional
                    vvar.append(k)
                    fbr_shape.append(b.num)
                    NV *= b.num 
                    axis_of_coord.append(ax) 
                    k +=1 
                elif isinstance(b, NDBasis): # NDBasis is `nd` dimensional
                    for i in range(b.nd):
                        vvar.append(k)
                        axis_of_coord.append(ax)
                        k += 1 
                    fbr_shape.append(b.Nb)
                    NV *= b.Nb
                else: # a scalar, not active 
                    fbr_shape.append(1)
                    axis_of_coord.append(ax)
                    k += 1
        
        # FBR shape, including the concatenated ellipsis axis 
        fbr_shape = tuple(fbr_shape)  
        NJ = 2*J + 1 
        NDP = NJ * NV # The size of the rot-vib direct product basis
                      # (which contains all ellipsis basis functions, 
                      #  not selected by which actually occur in the working basis)
                      
        
        # Check that there is at least one active coordinate        
        if len(vvar) == 0:
            raise ValueError("there must be an active coordinate!") 
            
            
        ################################################    
        # Evaluate quadrature grid quantities
        bases_quad = [b for b in bases]
        # Use one representative Ellipsis basis. These must
        # all have the same quadrature grid (or fixed value) anyway
        #
        bases_quad[ellipsis_idx] = ellipsis_bases[0] 
        Q = nitrogen.dvr.bases2grid(bases_quad) 
        
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
                
            # Apply PES min and max cutoffs
            if Vmax is not None:
                Vq[Vq > Vmax] = Vmax 
            if Vmin is not None:
                Vq[Vq < Vmin] = Vmin     
            
            # Then apply offset 
            if Voffset is not None:
                Vq = Vq - Voffset 
        #
        #
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
        #
        # Calculate coordinate system metric functions
        # 
        g = cs.Q2g(Q, deriv = 1, mode = 'bodyframe', 
                   vvar = vvar, rvar = 'xyz', masses = masses)
        #
        # And then calculate its inverse and determinant
        #
        G,detg = nitrogen.dfun.sym2invdet(g, 1, len(vvar))
        #
        # G and detg contain the derivative arrays for G = inv(g) and det(g)
        # 
        # If J = 0, then we only need to keep the vibrational block of G.
        # In packed storage, this is the first nv*(nv+1)/2 
        # elements (where nv = len(vvar))
        #
        if J == 0:
            nv = len(vvar)
            nG = (nv*(nv+1))//2 
            G = G[0][:nG] # need only value; lower triangle row-order
        else:
            G = G[0]      # keep all elements
        
        #
        # Calculate the log. deriv. of g
        gtilde = detg[1:] / detg[0]
        #
        
        ########################################
        #
        # Calculate the logarithmic derivatives of
        # the integration volume weight function 
        # defined by the basis sets. Evaluate over the 
        # quadrature grid Q.
        # (only necessary for active coordinates)
        #
        # We can use bases_quad, because the rho element for 
        # the Ellipsis basis must be the same for each azimuthal component
        rhotilde = nitrogen.dvr.calcRhoLogD(bases_quad, Q)
        
        # Calculate Gammatilde, the log deriv of the ratio
        # of the basis weight function rho and g**1/2
        #
        Gammatilde = rhotilde - 0.5 * gtilde  # active only
        
        ###################################
        #
        # Collect or construct the single
        # derivative operators for every
        # vibrational coordinate.
        # 
        # Again, we use bases_quad *BUT* the entries for
        # the Ellipsis basis coordinates will be invalid !!! 
        D = nitrogen.dvr.collectBasisD(bases_quad)
        #
        # For the Ellipsis basis coordinates, we need to calculate
        # derivatives for each block separately
        D_ellipsis = [nitrogen.dvr.collectBasisD([b]) for b in ellipsis_bases]
        # D_ellipsis[i][j] is the D operator for the j**th ellipsis coordinate of the i**th block
        
        
        ####################################
        #
        # Construct the angular momentum operators 
        # in the signed-k (i.e. Condon-Shortley) representation
        #
        # 
        Jx,Jy,Jz = nitrogen.angmom.Jbf_cs(J)  # k ordering = -J, -J + 1, ... +J
        iJ = (1j * Jx, 1j * Jy, 1j * Jz)
        #
        # Calculate the iJ, iJ anticommutator
        iJiJ = tuple(tuple( iJ[a]@iJ[b] + iJ[b]@iJ[a] for b in range(3)) for a in range(3))
        
        # Define the required LinearOperator attributes
        self.shape = (NH,NH)
        self.dtype = np.result_type(1j)  # complex128 
        
        self.az_m = az_m
        self.az_U = az_U 
        self.az_UH = az_UH 
        self.sing_val_mask = sing_val_mask 
        self.fbr_shape = fbr_shape 
        self.NDP = NDP 
        self.NV = NV 
        self.svm_1d = svm_1d 
        self.J = J 
        self.bases = bases 
        self.ellipsis_bases = ellipsis_bases 
        self.ellipsis_range = ellipsis_range 
        self.size_of_ellipsis_m = size_of_ellipsis_m
        self.ellipsis_idx = ellipsis_idx 
        self.iJ = iJ
        self.iJiJ = iJiJ
        self.Vq = Vq 
        self.D = D 
        self.D_ellipsis = D_ellipsis 
        self.axis_of_coord = axis_of_coord 
        self.Gammatilde = Gammatilde 
        self.G = G 
        self.hbar = hbar 
        self.nact = len(vvar) # The number of active coordinates 
        self.coord_k_is_ellip_coord = coord_k_is_ellip_coord
        
        return 
    
    def _matvec(self, x):
        
        J = self.J 
        NJ = 2*J + 1 
        NH = self.shape[0] 
        hbar = self.hbar
        nact = self.nact
        
        x = x.reshape((NH,)) # reshape to (NH,) 
        
        # The working representation contains only a sub-set of 
        # functions of the direct-product azimuthal representation
        #
        # 1) Convert the 1D vector in the working representation
        # to a 1D vector in the direct-product azimuthal representation
        x_dp = np.zeros((self.NDP,), dtype = np.complex128) 
        x_dp[self.svm_1d] = x 
        #
        # Then reshape this into the direct-product azimuthal representation
        # grid shape (`fbr_shape`)
        # The leading axis is the signed-k rotational wavefunctions. The
        # remaining axes are the mixed DVR/FBR grid shape
        #
        x_dp = np.reshape(x_dp, (NJ,) + self.fbr_shape) 
        
        #
        # 2) Now transform from the declared azimuthal representations
        # to the mixed DVR/FBR representation
        #
        x_fbr = nitrogen.dvr.ops.opTensorO(x_dp, self.az_U) 
        
        #
        # 3) Now transform from the mixed representation to the
        # quadrature representation. Each Ellipsis basis is treated 
        # separately. They go to the same quadrature grid.
        xq = 0
        k = 0
        
        nellip_coord = len(self.D_ellipsis[0])
        dxq_ellip = [0 for i in range(nellip_coord)]
        # dxq_ellip[i] is the quad representation of the derivative with 
        # respect to the i**th Ellipsis coordinate 
        for i in range(len(self.ellipsis_range)):
            # For each ellipsis basis
            axis = self.ellipsis_idx + 1 # The location of the ellipsis basis
                                         # (remember that rotations are first)
            nblock = self.size_of_ellipsis_m[i] # the number of functions in this
                                                # ellipsis block                            
            x_fbr_block = np.take(x_fbr, np.arange(k,k+nblock), axis = axis)
            
            block_bases = [None] + self.bases
            block_bases[axis] = self.ellipsis_bases[i] 
            
            xq_block = nitrogen.dvr._to_quad(block_bases, x_fbr_block)
            
            xq = xq + xq_block
            
            for j in range(nellip_coord):
                if self.D_ellipsis[i][j] is not None:
                    dj_xq_block = dvrops.opO(xq_block, self.D_ellipsis[i][j], self.ellipsis_idx + 1)
                    dxq_ellip[j] += dj_xq_block
            
            k += nblock 
        
        # xq contains the quadrature representation of the input function
        # dxq_ellip[i] contains the quadrature representation of the derivative
        # of the input function w.r.t. the i**th Ellipsis coordinate. 
        # If that is inactive, then dxq_ellip[i] is 0 
        
        #########################
        # Initialize yq, the quadrature representation of the
        # result
        #
        yq = np.zeros_like(xq)
        yq_block = [np.zeros_like(xq) for m in self.ellipsis_range] # A block-specific
        #                                                    contribution
        #
        #########################
        # Potential energy operator
        #
        if self.Vq is not None:
            for r in range(NJ):
                yq[r] += self.Vq * xq[r] 
        #########################
        
        #########################
        #
        # Kinetic energy operator 
        #
        # First, parts *not* involving the Ellipsis
        # basis will be handled
        
        # 1) Pure vibrational kinetic energy
        #    Diagonal in rotational index
        
        nd = len(self.D) # The number of coordinates (including inactive)
        for r in range(NJ):
            # Rotational block `r`
            lactive = 0
            for l in range(nd):
                # calculate dtilde_l acting on wavefunction,
                # result in the quadrature representation 
                
                
                if self.D[l] is None:
                    continue # an in-active coordinate, no derivative to compute
                
                if self.axis_of_coord[l] == self.ellipsis_idx:
                    dl_x = dxq_ellip[self.coord_k_is_ellip_coord[l]][r] # the derivative w.r.t. correct Ellipsis coordinate
                else:
                    # Apply the derivative matrix to the appropriate index
                    dl_x = dvrops.opO(xq[r], self.D[l], self.axis_of_coord[l]) 
                #
                # dtilde_l is the sum of the derivative 
                # and one-half Gammatilde_l
                #
                dtilde_l = dl_x + 0.5 * self.Gammatilde[lactive] * xq[r]
                
                
                kactive = 0 
                for k in range(nd):
                    
                    if self.D[k] is None:
                        continue # inactive
                    
                    # Get the packed-storage index for 
                    # G^{kactive, lactive}
                    kl_idx = nitrogen.linalg.packed.IJ2k(kactive,lactive)
                    Gkl = self.G[kl_idx]
                    
                    # Apply G^kl 
                    Gkl_dl = Gkl * dtilde_l
                    
                    # Now finish with (dtilde_k)
                    # and put results in quadrature representation
                    #
                    # include the final factor of -hbar**2 / 2
                    yq[r] += hbar**2 * 0.25 * self.Gammatilde[kactive] * Gkl_dl 
                    
                    if self.axis_of_coord[k] == self.ellipsis_idx:
                        # Ellipsis coordinate, this is block specific 
                        for m in range(len(self.ellipsis_range)):
                            # Result for m**th block
                            Dk_op = self.D_ellipsis[m][self.coord_k_is_ellip_coord[k]] 
                            # !!! This should be rechecked for complex basis functions!!!
                            yq_block[m][r] += hbar**2 * 0.50 * dvrops.opO(Gkl_dl, Dk_op.T, self.axis_of_coord[k])
                    else:
                        yq[r] += hbar**2 * 0.50 * dvrops.opO(Gkl_dl, self.D[k].T, self.axis_of_coord[k]) 
                    
                    kactive += 1
                    
                lactive += 1
        #      
        if J > 0:
            # Rotational and ro-vibrational terms are zero unless
            # J > 0
            #
            # 2) Pure rotational kinetic energy
            #
            #  -hbar**2/4  *  [iJa/hbar, iJb/hbar]_+  *  G^ab 
            #
            for a in range(3):
                for b in range(3):
                    # Because both [iJa,iJb]_+ and G^ab
                    # are symmetric with a <--> b, we only need 
                    # to loop over half
                    if (b > a):
                        continue 
                    if b == a:
                        symfactor = 1.0  # on the diagonal, count once!
                    else: # b < a 
                        symfactor = 2.0  # count both (b,a) and (a,b)
                    
                    # G^ab term
                    ab_idx = nitrogen.linalg.packed.IJ2k(nact + a, nact + b) 
                    Gab = self.G[ab_idx] 
                            
                    for rp in range(NJ):
                        for r in range(NJ):
                            # <rp | ... | r > rotational block
                            rot_me = self.iJiJ[a][b][rp,r] # the rotational matrix element
                            
                            if rot_me == 0:
                                continue # a zero rotational matrix element
                            
                            # otherwise, add contribution from
                            # effective inverse inertia tensor
                            yq[rp] += (symfactor * rot_me * (-hbar**2) * 0.25) * (Gab * xq[r])
                            
            #
            # 3) Rotation-vibration coupling
            #
            # -hbar**2 / 2 iJa/hbar * G^ak [psi' (dtildek psi) - (dtildek psi') psi]
            
            for a in range(3):
                for rp in range(NJ):
                    for r in range(NJ):
                        rot_me = self.iJ[a][rp,r] # the rotational matrix element  
                        if rot_me == 0:
                            continue 
                        
                        kactive = 0 
                        for k in range(nd):
                            #
                            # Vibrational coordinate k
                            #
                            if self.D[k] is None:
                                continue # an in-active coordinate, no derivative to compute
                            
                            # calculate index of G^ak
                            ak_idx = nitrogen.linalg.packed.IJ2k(nact + a, kactive)
                            Gak = self.G[ak_idx] 
                            
                            # First, do the psi' (dtilde_k psi) term
                            if self.axis_of_coord[k] == self.ellipsis_idx:
                                dk_x = dxq_ellip[self.coord_k_is_ellip_coord[k]][r] # the derivative w.r.t. correct Ellipsis coordinate
                            else:   
                                dk_x = dvrops.opO(xq[r], self.D[k], self.axis_of_coord[k]) 
                            dtilde_k = dk_x + 0.5 * self.Gammatilde[kactive] * xq[r]
                            yq[rp] += (rot_me * (-hbar**2) * 0.50) * Gak * dtilde_k
                            
                            # Now, do the -(dtilde_k psi') * psi term
                            yq[rp] += (rot_me * (+hbar**2) * 0.25) * self.Gammatilde[kactive] * Gak * xq[r] 
                            
                            if self.axis_of_coord[k] == self.ellipsis_idx:
                                # Ellipsis coordinate, this is block specific 
                                for m in range(len(self.ellipsis_range)):
                                    # Result for m**th block
                                    Dk_op = self.D_ellipsis[m][self.coord_k_is_ellip_coord[k]] 
                                    # !!! This should be rechecked for complex basis functions!!!
                                    yq_block[m][rp] += (rot_me * (+hbar**2) * 0.50) * dvrops.opO(Gak * xq[r], Dk_op.T, self.axis_of_coord[k])
                            else:
                                yq[rp] += (rot_me * (+hbar**2) * 0.50) * dvrops.opO(Gak * xq[r], self.D[k].T, self.axis_of_coord[k])
                            
                            kactive += 1
        
        #######################################################
        #
        # 4) Convert from the quadrature representation to the
        # mixed DVR/FBR representation
        y_fbr = 0
        k = 0
        y_fbr_blocks = []
        axis = self.ellipsis_idx + 1
        for i in range(len(self.ellipsis_range)):
            nblock = self.size_of_ellipsis_m[i]
            block_bases = [None] + self.bases
            block_bases[axis] = self.ellipsis_bases[i]
            # project the quadrature representation 
            # to the i**th ellipsis basis
            y_fbr_block = nitrogen.dvr._to_fbr(block_bases, yq) # Block independent contribution 
            y_fbr_block += nitrogen.dvr._to_fbr(block_bases, yq_block[i]) # Block-specific contribution 
            
            y_fbr_blocks.append(y_fbr_block)
        
        y_fbr = np.concatenate(y_fbr_blocks, axis = axis)
        
        
        #
        # 5) Transform from mixed DVR/FBR representation
        # to the multi-valued azimuthal representation
        # (Steps 4 and 5 could be combined like 2 and 3.)
        y_dp = nitrogen.dvr.ops.opTensorO(y_fbr, self.az_UH)
        
        # 6) Extract the singled-valued basis function 
        # coefficients from the multi-valued azimuthal
        # representation. This is the final working representation
        #
        #
        y = (np.reshape(y_dp,(-1,)))[self.svm_1d] 
        
        return y
    
    
    