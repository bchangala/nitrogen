"""
nitrogen.ham
------------

Hamiltonian construction routines.

"""

import numpy as np
from nitrogen.dvr import DVR
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