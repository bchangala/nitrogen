# -*- coding: utf-8 -*-
"""
General purpose space-fixed and body-fixed curvilinear Hamiltonians
"""

__all__ = ['GeneralSpaceFixed', 'Collinear', 'NonLinear', 'AzimuthalLinear']

import numpy as np 
from scipy.sparse.linalg import LinearOperator
import nitrogen.constants
import nitrogen.basis 

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
            A list of :class:`~nitrogen.basis.GriddedBasis` for active
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
        Q = nitrogen.basis.bases2grid(bases)
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
        
        fbr_shape, NH = nitrogen.basis.basisShape(bases)
        axis_of_coord = nitrogen.basis.coordAxis(bases)
        subcoord_of_coord = nitrogen.basis.coordSubcoord(bases)
        vvar = nitrogen.basis.basisVar(bases) 
        isactive = [(i in vvar) for i in range(len(axis_of_coord))]
        
        # Check that there is at least one active coordinate        
        if len(vvar) == 0:
            raise ValueError("there must be an active coordinate!") 
        
        ########################################
        #
        # Calculate coordinate system metric functions
        #
        g = cs.Q2g(Q, deriv = 1, mode = 'simple', vvar = vvar, masses = masses)
        #G,detg = nitrogen.dfun.sym2invdet(g, 1, len(vvar))
        # G and detg contain the derivative arrays for G = inv(g) and det(g)
        G = nitrogen.linalg.packed.inv_sp(g[0])  # need only value; lower triangle row-order
        #
        # Calculate gtilde_k = (d detg)/dQ_k / detg
        # Only do this for active coordinates (i.e. those in vvar) 
        #gtilde = detg[1:] / detg[0]
        gtilde = [nitrogen.linalg.packed.trAB_sp(G, g[i+1]) for i in range(len(vvar))]
        gtilde = np.stack(gtilde)
        #
        
        ########################################
        #
        # Calculate the logarithmic derivatives of
        # the integration volume weight function 
        # defined by the basis sets. Evaluate over the 
        # quadrature grid Q.
        # (only necessary for active coordinates)
        rhotilde = nitrogen.basis.calcRhoLogD(bases, Q)
        
        # Calculate Gammatilde, the log deriv of the ratio
        # of the basis weight function and the Euclidean metric
        # of the coordinate system (g^1/2)
        #
        Gammatilde = rhotilde - 0.5 * gtilde  # active only
        
        # Define the required LinearOperator attributes
        self.shape = (NH,NH)
        self.dtype = np.result_type(G, Gammatilde, Vq)
        
        # Additional attributes
        self.NH = NH        # The size of the Hamiltonian matrix
        self.bases = bases  # The basis sets
        self.Vq = Vq        # The PES quadrature grid
        self.G = G          # The inverse metric
        self.Gammatilde = Gammatilde # The pseudo-potential terms
        self.hbar = hbar    # The value of hbar.    
        self.fbr_shape = fbr_shape # The shape of the mixed DVR-FBR product basis 
        self.axis_of_coord = axis_of_coord # Grid axis of each coordinate
        self.subcoord_of_coord = subcoord_of_coord # Intra-basis coordinate index of each coordinate
        self.isactive = isactive # The activity of each coordinate 
        
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
        xq = nitrogen.basis._to_quad(self.bases, x)
        
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
        nd = len(self.axis_of_coord) # The number of coordinates (including inactive)
        for l in range(nd):
            # calculate dtilde_l acting on wavefunction,
            # result in the quadrature representation 
            
            if not self.isactive[l]:
                continue # an in-active coordinate, no derivative to compute
            
            # Apply the derivative matrix to the appropriate index
            ax_l = self.axis_of_coord[l] # The axis of this coordinate
            sc_l = self.subcoord_of_coord[l] # The intra-basis sub-index of this coordinate
            dl_x = self.bases[ax_l].d_grid(xq, sc_l, ax_l)
            
            #
            # dtilde_l is the sum of the derivative 
            # and one-half Gammatilde_l
            #
            dtilde_l = dl_x + 0.5 * self.Gammatilde[lactive] * xq
            
            
            kactive = 0 
            for k in range(nd):
                
                if not self.isactive[k]:
                    continue # inactive
                ax_k = self.axis_of_coord[k] # The axis of this coordinate
                sc_k = self.subcoord_of_coord[k] # The intra-basis sub-index of this coordinate
                
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
                yq += self.hbar**2 * 0.50 * self.bases[ax_k].dH_grid(Gkl_dl, sc_k, ax_k)
                
                kactive += 1 
                
            lactive += 1
        
        # yq contains the complete 
        # matrix-vector result in the quadrature representation
        #
        # Convert this back to the mixed FBR-DVR representation
        # and reshape back to a 1-D vector
        #
        y = nitrogen.basis._to_fbr(self.bases, yq)
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
            A list of :class:`~nitrogen.basis.GenericDVR` or
            :class:`~nitrogen.basis.NDBasis` basis sets for active
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
        Q = nitrogen.basis.bases2grid(bases)
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
        
        fbr_shape, NH = nitrogen.basis.basisShape(bases)
        axis_of_coord = nitrogen.basis.coordAxis(bases)
        vvar = nitrogen.basis.basisVar(bases) 
        subcoord_of_coord = nitrogen.basis.coordSubcoord(bases)
        isactive = [(i in vvar) for i in range(len(axis_of_coord))]
        
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
        rhotilde = nitrogen.basis.calcRhoLogD(bases, Q)
        
        # Calculate Gammatilde, the log deriv of the ratio
        # of the basis weight function and (gvib * I**2) ** 1/2
        #
        Gammatilde = rhotilde - 0.5 * gI2tilde  # active only
         

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
        self.hbar = hbar    # The value of hbar.    
        self.fbr_shape = fbr_shape # The shape of the mixed DVR-FBR product basis 
        self.axis_of_coord = axis_of_coord # Grid axis of each coordinate
        self.J = J          # The total angular momentum
        self.subcoord_of_coord = subcoord_of_coord # Intra-basis coordinate index of each coordinate
        self.isactive = isactive # The activity of each coordinate 
        
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
        xq = nitrogen.basis._to_quad(self.bases, x)
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
        nd = len(self.axis_of_coord)
        for l in range(nd):
            # calculate dtilde_l acting on wavefunction,
            # result in the quadrature representation 
            
            if not self.isactive[l]:
                continue # an in-active coordinate, no derivative to compute
            
            # Apply the derivative matrix to the appropriate index
            ax_l = self.axis_of_coord[l] # The axis of this coordinate
            sc_l = self.subcoord_of_coord[l] # The intra-basis sub-index of this coordinate
            dl_x = self.bases[ax_l].d_grid(xq, sc_l, ax_l)
            
            #
            # dtilde_l is the sum of the derivative 
            # and one-half Gammatilde_l
            #
            dtilde_l = dl_x + 0.5 * self.Gammatilde[lactive] * xq
            
            
            kactive = 0 
            for k in range(nd):
                
                if not self.isactive[k]:
                    continue # inactive
                ax_k = self.axis_of_coord[k] # The axis of this coordinate
                sc_k = self.subcoord_of_coord[k] # The intra-basis sub-index of this coordinate
                
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
                yq += self.hbar**2 * 0.50 * self.bases[ax_k].dH_grid(Gkl_dl, sc_k, ax_k)
                
                kactive += 1 
                
            lactive += 1
        
        # yq contains the complete 
        # matrix-vector result in the quadrature representation
        #
        # Convert this back to the mixed FBR-DVR representation
        # and reshape back to a 1-D vector
        #
        y = nitrogen.basis._to_fbr(self.bases, yq)
        
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
            A list of :class:`~nitrogen.basis.GenericDVR` or
            :class:`~nitrogen.basis.NDBasis` basis sets for active
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
        Q = nitrogen.basis.bases2grid(bases)
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
        
        fbr_shape, NV = nitrogen.basis.basisShape(bases)
        axis_of_coord = nitrogen.basis.coordAxis(bases)
        vvar = nitrogen.basis.basisVar(bases) 
        subcoord_of_coord = nitrogen.basis.coordSubcoord(bases)
        isactive = [(i in vvar) for i in range(len(axis_of_coord))]
        
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
        G = nitrogen.linalg.packed.inv_sp(g[0]) # value only
        #
        gtilde = [nitrogen.linalg.packed.trAB_sp(G, g[i+1]) for i in range(len(vvar))]
        gtilde = np.stack(gtilde)
        
        # If J = 0, then we only need to keep the vibrational block of G.
        # In packed storage, this is the first nv*(nv+1)/2 
        # elements (where nv = len(vvar))
        #
        if J == 0:
            nv = len(vvar)
            nG = (nv*(nv+1))//2 
            G = G[:nG].copy() # need only value; lower triangle row-order
        else:
            pass 

        
        ########################################
        #
        # Calculate the logarithmic derivatives of
        # the integration volume weight function 
        # defined by the basis sets. Evaluate over the 
        # quadrature grid Q.
        # (only necessary for active coordinates)
        rhotilde = nitrogen.basis.calcRhoLogD(bases, Q)
        
        # Calculate Gammatilde, the log deriv of the ratio
        # of the basis weight function rho and g**1/2
        #
        Gammatilde = rhotilde - 0.5 * gtilde  # active only
         
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
        self.hbar = hbar    # The value of hbar.    
        self.fbr_shape = fbr_shape # The shape of the mixed DVR-FBR product basis 
        self.axis_of_coord = axis_of_coord # Grid axis of each coordinate
        self.J = J          # The total angular momentum
        self.iJ = iJ        # The angular momentum operators
        self.iJiJ = iJiJ    #  " " 
        self.nact = len(vvar) # The number of active coordinates 
        self.subcoord_of_coord = subcoord_of_coord # Intra-basis coordinate index of each coordinate
        self.isactive = isactive # The activity of each coordinate 
        
        return  
    #@profile 
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
        xq = nitrogen.basis._to_quad([None] + self.bases, x)
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
        
        nd = len(self.axis_of_coord) # The number of coordinates (including inactive)
        for r in range(NJ):
            # Rotational block `r`
            lactive = 0
            for l in range(nd):
                # calculate dtilde_l acting on wavefunction,
                # result in the quadrature representation 
                
                
                if not self.isactive[l]:
                    continue # an in-active coordinate, no derivative to compute
                
                # Apply the derivative matrix to the appropriate index
                ax_l = self.axis_of_coord[l] # The axis of this coordinate
                sc_l = self.subcoord_of_coord[l] # The intra-basis sub-index of this coordinate
                dl_x = self.bases[ax_l].d_grid(xq[r], sc_l, ax_l)
            
                #
                # dtilde_l is the sum of the derivative 
                # and one-half Gammatilde_l
                #
                dtilde_l = dl_x + 0.5 * self.Gammatilde[lactive] * xq[r]
                
                
                kactive = 0 
                for k in range(nd):
                    
                    if not self.isactive[k]:
                        continue # inactive
                    ax_k = self.axis_of_coord[k] # The axis of this coordinate
                    sc_k = self.subcoord_of_coord[k] # The intra-basis sub-index of this coordinate
                    
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
                    yq[r] += hbar**2 * 0.50 * self.bases[ax_k].dH_grid(Gkl_dl, sc_k, ax_k)
                    
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
                            if not self.isactive[k]:
                                continue # inactive
                            ax_k = self.axis_of_coord[k] # The axis of this coordinate
                            sc_k = self.subcoord_of_coord[k] # The intra-basis sub-index of this coordinate
                            
                            # calculate index of G^ak
                            ak_idx = nitrogen.linalg.packed.IJ2k(nact + a, kactive)
                            Gak = self.G[ak_idx] 
                            
                            # First, do the psi' (dtilde_k psi) term
                            dk_x = self.bases[ax_k].d_grid(xq[r], sc_k, ax_k)
                            
                            dtilde_k = dk_x + 0.5 * self.Gammatilde[kactive] * xq[r]
                            yq[rp] += (rot_me * (-hbar**2) * 0.50) * Gak * dtilde_k
                            
                            # Now, do the -(dtilde_k psi') * psi term
                            yq[rp] += (rot_me * (+hbar**2) * 0.25) * self.Gammatilde[kactive] * Gak * xq[r] 
                            yq[rp] += (rot_me * (+hbar**2) * 0.50) * self.bases[ax_k].dH_grid(Gak * xq[r], sc_k, ax_k)
                            
                            kactive += 1
        
        # yq contains the complete 
        # matrix-vector result in the quadrature representation
        #
        # Convert this back to the mixed FBR-DVR representation
        # and reshape back to a 1-D vector
        # (Prepend a dummy basis to keep rotational index unchanged)
        y = nitrogen.basis._to_fbr([None] + self.bases, yq)
        
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
        
        # The ellipsis entry must be a callable 
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
        
        min_m = -J  # Keep track of Ellipsis quantum number range;
        max_m = J   # initialize the range to [-J, J]
        
        for i in range(len(bases)):
            if azimuth[i] is None: # This is not declared an azimuthal coordinate
                if np.isscalar(bases[i]): # A scalar singleton factor
                    az_m.append(np.array([0]))
                else: # A generic basis
                    az_m.append(np.array([0] * bases[i].nb))
                az_U.append(None) 
            elif azimuth[i] is Ellipsis:
                # This is the callable "Ellipsis" basis
                az_m.append(Ellipsis) # Place holder 
                az_U.append(None)     # Azimuthal representation is implied
            else:
                #
                # Attempt to compute the azimuthal representation of this basis.
                # This is the representation that diagonalizes the derivative
                # operator.
                #
                try:
                    # Calculate the grid of the derivative of each basis function
                    df = bases[i].basis2grid_d(np.eye(bases[i].nb), azimuth[i][0], axis = 0)
                    D = bases[i].grid2basis(df, axis = 0)
                except:
                    raise RuntimeError(f"Attempt to build derivative operator for basis [{i:d}] failed.")
                    
                w,u = np.linalg.eigh(-1j * D) 
                w *= azimuth[i][1] # correct for right or left-handed sense / unit scaling
                #
                # u transforms a vector in the azimuthal representation
                # to the original FBR or DVR representation
                
                # in principle, w should be exact integers
                # assuming a proper basis has been chosen
                # (i.e. one that is closed under rotations of the azimuthal 
                #  coordinate) 
                #
                # Let's check that the eigenvalues are indeed close to integer
                # values 
                w_int = np.rint(w).astype(np.int32) # round to nearest integer
                if np.max(np.absolute(w_int - w)) > 1e-10:
                    print("Warning: azimuthal quantum numbers are not quite integer!"
                          " (or perhaps were rounded unexpectedly)")
                
                az_m.append(w_int) # Save the integer values
                az_U.append(u) 
                
                min_m -= np.max(w_int)
                max_m -= np.min(w_int)
                
        # Calculate the conjugate tranpose of each az_U
        # This will transform from original DVR/FBR basis to the azimuthal representation
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
        # These have been kept track of with min_m and max_m in the above
        # loops
        ellipsis_range = np.arange(min_m, max_m + 1)
        #
        # For each m in this range, bases[ellipsis_idx](m) returns
        # a basis specification. This must be a compatible GriddedBasis, i.e.
        # not a scalar. (It is possible that a basis cannot be formed, in which
        # case we will not include that azimuthal quantum number.)
        #
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
        
        #
        # Concatenate the individual blocks of basis functions
        # for the Ellipsis factor into a single generic basis set
        # and record the corresponding azimuthal quantum number for
        # every individual basis function
        ellipsis_basis_joined = nitrogen.basis.ConcatenatedBasis(ellipsis_bases) 
        size_of_ellipsis_m = [b.nb for b in ellipsis_bases]
        # Nellip = sum(size_of_ellipsis_m)
        az_m[ellipsis_idx + 1] = np.repeat(ellipsis_range, np.array(size_of_ellipsis_m))
        
        #
        # az_m now contains the azimuthal quantum numbers for each basis factor.
        # Non-azimuthal factors and fixed scalars have values of zero.
        # We now need to determine which combinations of direct products
        # have the correct quantum numbers 
        #
        
        az_grids = np.meshgrid(*az_m, indexing = 'ij') 
        
        total = az_grids[0] - sum(az_grids[1:]) # k - sum(m)
        sing_val_mask = (total == 0) 
        svm_1d = np.reshape(sing_val_mask, (-1,)) 
        NH = np.count_nonzero(svm_1d)  # the number of single-valued functions
        
        # The True values of svm_1d are the direct-product functions
        # in the azimuthal representation that have the correct 
        # combination of quantum numbers.
        
        ########################################
        # Analyze final direct-product basis shape
        # and activity
        #
        # First, form the final vibrational bases list including the 
        # concatenated Ellipsis basis. This is the direct-product
        # basis set in the DVR/FBR representation
        #
        bases_dp = [bases[i] if i != ellipsis_idx else ellipsis_basis_joined for i in range(len(bases))]
        
        fbr_shape, NV = nitrogen.basis.basisShape(bases_dp)
        axis_of_coord = nitrogen.basis.coordAxis(bases_dp)
        vvar = nitrogen.basis.basisVar(bases_dp)
        #coord_k_is_ellip_coord = [None for i in range(cs.nQ)]
        
        NJ = 2*J + 1 
        NDP = NJ * NV # The size of the rot-vib direct product basis
                      # (which contains all ellipsis basis functions, 
                      #  not selected by which actually occur in the working basis)
                      
        # Check that there is at least one active coordinate        
        if len(vvar) == 0:
            raise ValueError("there must be an active coordinate!") 
            
        ################################################    
        # Evaluate quadrature grid quantities
        
        Q = nitrogen.basis.bases2grid(bases_dp) 
        
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
        #G,detg = nitrogen.dfun.sym2invdet(g, 1, len(vvar))
        G = nitrogen.linalg.packed.inv_sp(g[0])
        #
        # Calculate the log. deriv. of det(g)
        #
        gtilde = [nitrogen.linalg.packed.trAB_sp(G, g[i+1]) for i in range(len(vvar))]
        gtilde = np.stack(gtilde)
        
        # If J = 0, then we only need to keep the vibrational block of G.
        # In packed storage, this is the first nv*(nv+1)/2 
        # elements (where nv = len(vvar))
        #
        if J == 0:
            nv = len(vvar)
            nG = (nv*(nv+1))//2 
            #G = G[0][:nG] # need only value; lower triangle row-order
            G = G[:nG].copy() 
        else:
            pass      # keep all elements
        

        
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
        rhotilde = nitrogen.basis.calcRhoLogD(bases_dp, Q)
        
        # Calculate Gammatilde, the log deriv of the ratio
        # of the basis weight function rho and g**1/2
        #
        Gammatilde = rhotilde - 0.5 * gtilde  # active only
        
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
        self.bases_dp = bases_dp
        self.ellipsis_idx = ellipsis_idx 
        self.iJ = iJ
        self.iJiJ = iJiJ
        self.Vq = Vq 
        self.axis_of_coord = axis_of_coord 
        self.Gammatilde = Gammatilde 
        self.G = G 
        self.hbar = hbar 
        self.nact = len(vvar) # The number of active coordinates 
        
        return 
    
    
    def _matvec(self, x):
        
        J = self.J 
        NJ = 2*J + 1 
        NH = self.shape[0] 
        hbar = self.hbar
        nact = self.nact
        eidx = self.ellipsis_idx
        
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
        x_fbr = nitrogen.basis.ops.opTensorO(x_dp, self.az_U) 
        
        #
        # 3) Now transform from the mixed representation to the
        # quadrature representation. Also calculate the 
        # quadrature grids of the derivatives 
        #
        # Remember that the first axis of x_fbr is the rotational index
        #
        
        # First transform the ellipsis basis to the quadrature grid
        # (this will reduce the size considerably)
        x_qe = self.bases_dp[eidx].basis2grid(x_fbr, axis = eidx + 1)
        # Now transform the remaining factors 
        xq = x_qe 
        for i,b in enumerate(self.bases_dp):
            if np.isscalar(b):
                pass 
            elif i != eidx:
                xq = b.basis2grid(xq, axis = i + 1)
            else:
                pass # ellipsis factor is already transformed
        # Now calculate derivatives
        dxq = [] 
        for i,b in enumerate(self.bases_dp):
            if np.isscalar(b):
                pass # Inactive, no entry in list 
            elif i != eidx:
                # A normal basis factor, for which we assume left-unitarity
                for k in range(b.nd):
                    dxq.append(b.d_grid(xq, k, axis = i+1))
            else:
                # The Ellipsis factor 
                for k in range(b.nd):
                    # Tranform from direct-product fbr to derivative-grid
                    dk = b.basis2grid_d(x_fbr, k, axis = eidx+1)
                    # Now transform rest again
                    for j,bj in enumerate(self.bases_dp):
                        if np.isscalar(bj):
                            pass
                        elif j != eidx:
                            dk = bj.basis2grid(dk, axis = j + 1)
                        else:
                            pass
                    dxq.append(dk) 
        #
        # Calculate the ellipsis derivatives in this way forces us
        # to perform basis2grid transformations for other coordinates
        # multiple times (instead of just doing it once with x_fbr)
        # This is still likely to be cheaper because there is usually
        # only 1 ellipsis coordinate, but transforming it from the
        # Concatenated FBR representation (which is large) to the
        # single quadrature grid (which is usually small) reduces the
        # array size considerably.
        #
        
        #
        # dxq now contains the quadrature representation 
        # of the derivative for each coordinate. Inactive
        # coordinates have an entry of None.
        #
            
        #########################
        # Initialize yq, the quadrature representation of the
        # result. Also initialize result arrays for quads
        # associated with each left/bra-side derivative.
        #
        yq = np.zeros_like(xq)
        dyq = [np.zeros_like(xq) for i in range(nact)]
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
        # 1) Pure vibrational kinetic energy
        #    Diagonal in rotational index
        
        for r in range(NJ):
            # Rotational block `r`
            for lact in range(nact):
                # calculate dtilde_l acting on wavefunction,
                # result in the quadrature representation 
                
                #
                # dtilde_l is the sum of the derivative 
                # and one-half Gammatilde_l
                #
                dtilde_l = dxq[lact][r] + 0.5 * self.Gammatilde[lact] * xq[r]
                
                for kact in range(nact):
                    
                    # Get the packed-storage index for 
                    # G^{kactive, lactive}
                    kl_idx = nitrogen.linalg.packed.IJ2k(kact, lact)
                    Gkl = self.G[kl_idx]
                    
                    # Apply G^kl 
                    Gkl_dl = Gkl * dtilde_l
                    
                    # Now finish with (dtilde_k)
                    # and put results in quadrature representation
                    #
                    # include the final factor of -hbar**2 / 2
                    yq[r] += (hbar**2 * 0.25) * self.Gammatilde[kact] * Gkl_dl 
                    dyq[kact][r] += (hbar**2 * 0.50) * Gkl_dl 

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
                        
                        
                        for kact in range(nact):
                            #
                            # Vibrational coordinate k
                            #
    
                            # calculate index of G^ak
                            ak_idx = nitrogen.linalg.packed.IJ2k(nact + a, kact)
                            Gak = self.G[ak_idx] 
                            
                            # First, do the psi' (dtilde_k psi) term

                            dtilde_k = dxq[kact][r] + 0.5 * self.Gammatilde[kact] * xq[r]
                            yq[rp] += (rot_me * (-hbar**2) * 0.50) * Gak * dtilde_k
                            
                            # Now, do the -(dtilde_k psi') * psi term
                            Gak_xq = Gak * xq[r] 
                            yq[rp] += (rot_me * (+hbar**2) * 0.25) * self.Gammatilde[kact] * Gak_xq
                            dyq[kact][rp] += (rot_me * (+hbar**2) * 0.50) * Gak_xq
                            
        #######################################################
        #
        # 4) Convert from the quadrature representation to the
        # mixed DVR/FBR representation
        #
        # There is first the simple contribution from yq
        # and then the contributions from all derivatives
        #
        # We work in the reverse order as the original basis-to-grid
        # transformation above.
        kact = 0

        y_fbr = 0
        for i,b in enumerate(self.bases_dp):
            if np.isscalar(b):
                pass # Inactive, no entry in dyq
            elif i != eidx:
                # A normal basis factor, which we assume 
                # is left-unitary for the grid transformation
                for k in range(b.nd):
                    # We can add this result to the normal
                    # yq quadrature result
                    yq += b.dH_grid(dyq[kact], k, axis = i+1)
                    kact += 1
            else:
                for k in range(b.nd): # For each ellipsis derivative
                    dk = dyq[kact]
                    for j,bj in enumerate(self.bases_dp):
                        if np.isscalar(bj):
                            pass
                        elif j != eidx:
                            dk = bj.grid2basis(dk, axis = j + 1)
                        else: 
                            pass # do last 
                    # finally, transform ellipsis grid onto derivative functions
                    y_fbr += b.grid2basis_d(dk, k, axis = eidx + 1)
                    
                    kact += 1
        
        #
        # yq contains the contributions from the non-derivative 
        # result and the derivatives of all non-ellipsis coordinates
        # 
        # y_fbr currently contains just the results from the ellipsis derivatives
    
        y_qe = yq 
        for i, b in enumerate(self.bases_dp):
            if np.isscalar(b):
                pass
            elif i != eidx:
                y_qe = b.grid2basis(y_qe, axis = i + 1)
               
            else:
                pass # the ellipsis axis will be transformed last 
        y_fbr += self.bases_dp[eidx].grid2basis(y_qe, axis = eidx + 1) 
               
        #
        # 5) Transform from mixed DVR/FBR representation
        # to the multi-valued azimuthal representation
        y_dp = nitrogen.basis.ops.opTensorO(y_fbr, self.az_UH)
        
        # 6) Extract the singled-valued basis function 
        # coefficients from the multi-valued azimuthal
        # representation. This is the final working representation
        #
        #
        y = (np.reshape(y_dp,(-1,)))[self.svm_1d] 
        
        return y