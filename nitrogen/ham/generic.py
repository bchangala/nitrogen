# -*- coding: utf-8 -*-
"""
General purpose space-fixed and body-fixed curvilinear Hamiltonians
"""

__all__ = ['GeneralSpaceFixed', 'Collinear', 'NonLinear', 'AzimuthalLinear',
           'AzimuthalLinearRT']

import numpy as np 
from scipy.sparse.linalg import LinearOperator
import nitrogen.constants
import nitrogen.basis 
import warnings

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
            A list of :class:`~nitrogen.basis.GriddedBasis` basis sets for active
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
            A list of :class:`~nitrogen.basis.GriddedBasis` basis sets for active
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
        # Determine which elements of G are strictly zero
        G_is_zero = [np.max(abs(G[i])) < 1e-10 for i in range(G.shape[0])]
        
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
        self.G_is_zero = G_is_zero # Zero-mask for G elements
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
                    
                    if not self.G_is_zero[kl_idx]: # If G element is non-zero
                        
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
                    
                    if not self.G_is_zero[ab_idx]: # If G element is non-zero
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
                            
                            if not self.G_is_zero[ak_idx]:
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
    
    @staticmethod
    def vectorRME(bases,fun,X,Y):
        """
        Evaluate reduced matrix elements of a lab-frame vector operator.

        Parameters
        ----------
        bases : list of GriddedBasis and scalar
            The direct-product basis set factors.
        fun : function
            A function that returns the :math:`xyz` body-frame
            components of the vector :math:`V` in terms of the
            coordinates of `bases`.
        X,Y : list of ndarray
            Each element is an array of vectors in 
            the `bases` basis set of a given value of :math:`J` 
            following conventions of the :class:`NonLinear` 
            Hamiltonian.

        Returns
        -------
        VXY : ndarray
            The scaled reduced matrix elements :math:`\\langle X || V || Y \\rangle`.
            See Notes for precise definition.

        Notes
        -----
        
        We define the standard (lab-frame) reduced matrix element by
        
        ..  math::
            
            \\langle Jm\\cdots | V_Q | J' m'\\cdots \\rangle = 
                \\langle J' m', 1 Q | J m \\rangle
                \\langle J \\cdots || V || J' \\cdots \\rangle
        
        This function returns the scaled value
        :math:`V_{JJ'} = \\sqrt{2J+1} \\langle J \\cdots || V || J' \\cdots \\rangle`.
        
        The RME itself is calculated as 
        
        ..  math::
            
            \\langle J \\cdots || V || J' \\cdots \\rangle = 
                \\sqrt{2J' + 1} \\sum_{q k k'} (-1)^{J + k' + q}
                \\left(\\begin{array}{ccc} J & J' & 1 \\\\ k  & -k' & q \\end{array} \\right) 
                \\langle \\Psi^{(J,k)} | V_{-q} | \\Psi^{(J',k')} \\rangle,
                
        where :math:`\\Psi^{(J,k)}` is the vibrational factor associated with 
        the signed-:math:`k` symmetric top basis function :math:`| J,k \\rangle`.
        
        Note that the scaled RME satisfies :math:`|V_{J' J}|^2 = |V_{JJ'}|^2` and 
        in general
        
        ..  math::
            
            |V|_{J J'}^2 = \sum_{Amm'} | \\langle Jm | V_A | J' m' \\rangle | ^2
            
        When :math:`V` is the electric dipole operator :math:`\\mu`, then 
        :math:`|V|_{JJ'}^2` equals the line strength, :math:`S_{JJ'}`.
        
        """
        
        # Get the basis FBR shape and total
        # number of vibrational basis functions, nb
        fbr_shape, nb = nitrogen.basis.basisShape(bases)
        
        #
        # Now parse the sizes and implied J-values of 
        # each block of vectors in X and Y
        #
        nX, JX = [],[]
        for x in X:
            nX.append( x.shape[1] )
            JX.append( ((x.shape[0] // nb) - 1)//2 )
        nY, JY = [],[]
        for y in Y:
            nY.append( y.shape[1] )
            JY.append( ((y.shape[0] // nb) - 1)//2 ) 
            
        # Construct the coordinate grid over which 
        # we need to evaluate the vector-valued function
        Q = nitrogen.basis.bases2grid(bases)
        
        # Evaluate the vector-valued function.
        #
        Vbf = fun(Q) 
        # Vbf has a shape of (3,) + Q.shape[1:] (the quadrature shape)
        
        # Vbf[0,1,2] are the body-fixed x,y,z axis components
        #
        # Construct the spherical components
        # Vq, q = 0, +1, -1
        #
        #   0  ... z
        #  +1  ... -(x + iy) / sqrt[2]
        #  -1  ... +(x - iy) / sqrt[2]
        #
        # Note that the ordering of the spherical components
        # allow normal array indexing
        Vq = [  Vbf[2],
               -(Vbf[0] + 1j*Vbf[1])/np.sqrt(2.0),
               +(Vbf[0] - 1j*Vbf[1])/np.sqrt(2.0)]
        
        dX,dY = sum(nX), sum(nY) # The total size of the reduced matrix 
        
        # Initialize the reduced matrix
        VXY = np.zeros((dX,dY), dtype = np.complex128)
        
        # 
        # Calculate <X||MU||Y> reduced matrix element
        # block-by-block, including extra scaling.
        
        for i in range(len(nX)):
            # For each X block,
            # Reshape block of vectors to
            # (2J+1, fbr_shape, nX[i]) 
            
            Xi = np.reshape(X[i], (2*JX[i]+1,) + fbr_shape + (nX[i],) )
            # 
            # Now transform the entire block to quadrature grid 
            # Use None's to leave rotational and index axes untouched
            xi = nitrogen.basis._to_quad([None] + bases + [None], Xi)
            
            # The rotational index is w.r.t the NonLinear Hamiltonian's
            # real Wang basis. Transform these to the standard
            # CS signed-k symmetric top basis.
            #
            UX = nitrogen.angmom.U_wr2cs(JX[i]) # The transformation matrix
            xi = np.tensordot(UX, xi, axes = 1) # Transfrom from Wang-Real to CS 
            # (the rotational index is first, so no axis swapping is needed
            #  after tensordot)
                
            for j in range(len(nY)):
                
                # Process the Y block of vectors the same way
                #
                Yj = np.reshape(Y[j], (2*JY[j]+1,) + fbr_shape + (nY[j],) )
                yj = nitrogen.basis._to_quad([None] + bases + [None], Yj)
                UY = nitrogen.angmom.U_wr2cs(JY[j])
                yj = np.tensordot(UY, yj, axes = 1) 
                
                # idx_x and idx_y will be the array indices
                # of the final VXY reduced matrix
                #
                # Each block starts at the position equal to the 
                # sum of the sizes of the previous blocks
                
                # bi and bj will index the vectors within each block
                
                idx_x = sum(nX[:i])
                for bi in range(nX[i]):
                    
                    idx_y = sum(nY[:j])
                    for bj in range(nY[j]):
                        
                        # Perform summation over body-fixed spherical 
                        # component q and the body-fixed projections
                        # k (of X) and k' (of Y)
                        #
                        for q in [0,1,-1]: # ordering here doesn't matter
                            for k in range(-JX[i], JX[i]+1):
                                for kp in range(-JY[j], JY[j]+1):
                                    
                                    # Check selection rules 
                                    if JX[i] < abs(JY[j]-1) or JX[i] > JY[j] + 1:
                                        # failed triangle rule
                                        continue 
                                    if kp - q != k :
                                        # failed z-component rule
                                        continue 
                                    
                                    # Calculate contribution to reduced
                                    # matrix element
                                    #
                                    factor = (-1)**(JX[i] + kp + q) * \
                                            nitrogen.angmom.wigner3j(2*JX[i], 2*JY[j], 2*1,
                                                                     2*k,    -2*kp   , 2*q)
                                    if factor == 0.0:
                                        continue # final check for zero
                                    
                                    # Compute quadrature sum of the vibrational
                                    # integral
                                    bra = xi[JX[i] + k , ..., bi]
                                    ket = yj[JY[j] + kp, ..., bj]
                                    mid = Vq[-q] 
                                    integral = np.sum(np.conj(bra) * mid * ket) 
                                    
                                    # Finally, 
                                    # include sqrt[2J+1] factor to symmetrize the 
                                    # reduced matrix ! 
                                    #
                                    VXY[idx_x, idx_y] += np.sqrt(2*JX[i]+1) * np.sqrt(2*JY[j] + 1) * factor * integral 
                        
                        idx_y += 1 
                    
                    idx_x += 1 
        
        # VXY is complete
        # return the reduced matrix elements
        
        return VXY 
        
        
class AzimuthalLinear(LinearOperator):
    """
    A general rovibrational Hamiltonian for linear molecules. 
    
    This Hamiltonian enables a fairly flexible treatment of linear
    molecules that accounts for the necessary rovibrational boundary conditions 
    related to linear geometries. The same kinetic energy operator
    is used as that of :class:`NonLinear` Hamiltonians. The requirements 
    for the basis functions are explained in more detail in the parameter notes
    below.
    
    """
     
    def __init__(self, bases, cs, azimuth, pes = None, masses = None, J = 0, hbar = None,
                 Vmax = None, Vmin = None, Voffset = None,
                 signed_azimuth = False ):
        """

        Parameters
        ----------
        bases : list
            A list of :class:`~nitrogen.basis.GriddedBasis` basis sets for active
            coordinates. Scalar elements will constrain the 
            corresponding coordinate to that fixed value.
        cs : CoordSys
            The coordinate system.
        azimuth : list
            The azimuthal designation of each element of `bases`. Each element
            must be one of None, Ellipsis, or a two-element tuple. See Notes
            for details.
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
        Voffset : scalar, optional
            A potential energy offset. This will be subtracted
            from the surface value. The default is None.
        signed_azimuth : bool, optional
            If True, then the Ellipsis basis functions depend on the
            sign of the azimuthal quantum number. If False, then 
            the sign is ignored. The default is False.

        Notes
        -----
        
        Basis functions are constructed as products of factors supplied
        in the `bases` parameter. Each (non-scalar) element represents
        a single, possibly multi-dimensional, :class:`~nitrogen.basis.GriddedBasis`
        basis set. Together with symmetric-top rotational wavefunctions,
        the total product is 
        
        ..  math::
        
            \\Phi = f^{(m_1)}_i g^{(m_2)}_j h^{(m_3)}_k \\cdots \\vert J,k\\rangle
        
        where :math:`i,j,k,\\ldots` are the basis function indices. Each
        factor is labeled with an additional *azimuthal quantum number*,
        :math:`m_i`, assigned automatically (see below). The linear boundary
        conditions are enforced by selecting only basis functions for which
        
        ..  math::
        
            k - \\sum_i m_i = 0.
        
        The azimuthal quantum numbers for each basis factor are assigned according
        to the `azimuth` list, which has one element for each element in `bases`.
        
        A basis factor can be assigned in one of three ways:
            
        1. For factors/coordinates not relevant to the linear boundary conditions (typically
        radial distances) the appropriate azimuthal quantum number is simply
        :math:`m=0` for every basis function of that factor. This is indicated
        with an `azimuth` element of ``None``. 
        
        2. For factors that involve internal rotation coordinates,
        the azimuthal quantum number corresponds to the vibrational angular momentum 
        for internal rotation about the body-fixed :math:`z` axis. Currently,
        only one coordinate from a given basis factor can be identified 
        as an internal rotation coordinate. This is specified with an `azimuth`
        element of a two-element tuple ``(i, a)``. ``i`` is the coordinate
        index of the internal rotation index for that basis factor (i.e. ``i`` = 0
        means the first coordinate in the function, ``i`` = 1 the second, etc.)
        ``a`` is a scaling parameter which defines the handedness and units
        of the coordinate. The sign of ``a`` is positive for right-handed
        rotation about :math:`z` and negative for left-handed rotation.
        Its magnitude is equal to the geometric period of the internal 
        rotation coordinate (in whatever units it is defined
        in) divided by :math:`2\\pi`.
        
        The azimuthal quantum numbers are automatically determined by 
        calculating the matrix representation of the operator :math:`-i a \\partial`,
        where :math:`\\partial` is the partial derivative with respect to the
        coordinate identified by the ``azimuth`` entry. To work properly, 
        the set of basis functions must be closed under :math:`\\partial` (i.e.
        a unitary transformation produces exact eigenfunctions) and the grid
        representation must itself be quasi-unitary. The eigenfunctions
        of :math:`-i a \\partial` are referred to as the *azimuthal representation*, 
        and this is the working representation of the linear Hamiltonian.
        The corresponding eigenvalues are the azimuthal quantum numbers 
        :math:`m`. 
        
        3. There will be one special coordinate, :math:`\\theta`,
        that behaves as a generalized
        polar coordinate (e.g. the bond angle of a triatomic molecule). 
        The boundary conditions on this coordinate as it approaches
        linear geometries are related to the "pure rotational" angular 
        momentum component
        
        ..  math::
            
            m^* = k - \\sum_{i'} m_{i'}
        
        where the sum includes all azimuthal quantum numbers other than
        that associated with the polar coordinate. Usually, the polar
        coordinate should have an integration volume element that
        goes like :math:`\\sim \\theta` near linear geometries and 
        its basis functions should go like :math:`\\sim \\theta^{m^*}`.
        Associated Legendre polynomials and 2D radial harmonic oscillator 
        wavefunctions are two such examples.
        
        The user is required to explicitly provide separate sets of 
        basis functions for every possible (integer) value of 
        :math:`m^*`. That is, the element of `bases` for the polar
        coordinate is not just a single :class:`~nitrogen.basis.GriddedBasis`,
        but a function of signature ``f(m)`` that returns a :class:`~nitrogen.basis.GriddedBasis`
        with appropriate boundary conditions. Each of these different
        basis sets must have equivalent grids and quadrature rules. 
        
        To indicate that a given factor contains the generalized
        polar coordinate, the corresponding element in `azimuth` is
        ``...`` (``Ellipsis``). One and only one factor must be designated
        as the polar coordinate.
        
        
        """
        
        # Process Azimuthal basis factors 
        #
        # Define the azimuthal quantum number of the rotational 
        # basis functions, which is just -k
        k_azimuth = -np.arange(-J,J+1) # -k for basis order k = -J,...,+J
        
        az_m, az_U, az_UH, sing_val_mask, svm_1d, NH, bases_dp, ellipsis_idx  = \
            AzimuthalLinear._process_generic_azimuthal_basis(bases, azimuth, k_azimuth, signed_azimuth)
        
        
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
        # Determine which elements of G are strictly zero
        G_is_zero = [np.max(abs(G[i])) < 1e-10 for i in range(G.shape[0])]
        
        
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
        self.G_is_zero = G_is_zero
        self.hbar = hbar 
        self.nact = len(vvar) # The number of active coordinates 
        
        return 
    
    @staticmethod 
    def _process_generic_azimuthal_basis(bases, azimuth, sre_quantum_number, signed_azimuth):
        """
        Process an AzimuthalLinear basis function set.

        Parameters
        ----------
        bases : list
            The vibrational basis set specification
        azimuth : list
            The azimuthal designations of `bases`.
        sre_quantum_number : array_like
            The effective azimuthal quantum number for a general (spin-electronic-)rotational
            basis factor.
        signed_azimuth : bool
            If True, the Ellipsis basis functions depend on the sign of 
            the azimuthal quantum number. If False, its sign is ignored.


        Returns
        -------
        az_m : list
            The azimuthal quantum numbers for each basis factor, including
            the rotational factor, which is first.
        az_U : list
            The azimuthal-to-DVR/FBR unitary transformation matrices for 
            each basis factor. An entry of None indicates identity.
        az_UH : list 
            The conjugate transpose of each element of az_U. 
            These transform from the DVR/FBR representation to the
            azimuthal representation 
        sing_val_mask : ndarray
            The direct-product-basis mask indicating which 
            basis set functions are single-valued and included.
        svm_1d : ndarray
            A 1D-shaped version of `sing_val_mask`
        NH : ndarray
            The number of non-zero entries of `svm_1d`, i.e. 
            the size of the working basis set.
        bases_dp : list
            The set of direct-product basis factors. The
            entry for the Ellipsis basis function is the concatenation 
            of all basis sets needed for each azimuthal quantum number.
        ellipsis_idx : integer
            The index of the Ellipsis factor.
            
        
        Notes
        -----
        The value of `sre_quantum_number` should be :math:`-k` for 
        standard symmetric top rotational basis functions, where :math:`k`
        is the projection along the body-fixed :math:`z` axis. For 
        linear molecules with orbital angular momentum, it should be
        the value of :math:`-k + \\Lambda` for each case (b) basis
        function.
        
        """
        
        # For each basis, get the azimuthal quantum number
        # list 
        
        if len(azimuth) != len(bases):
            raise ValueError("azimuth must be same length as bases")
        
        # There should be one entry of Ellipsis in the azimuth list
        n = 0
        for i,entry in enumerate(azimuth):
            if entry is Ellipsis:
                n += 1
                ellipsis_idx = i # The basis index for the Ellipsis factor
        if n != 1:
            raise ValueError("azimuth must contain exactly one Ellipsis")
        
        # The ellipsis entry must be a callable 
        if not callable(bases[ellipsis_idx]):
            raise ValueError("The ... bases entry must be callable")
        
        az_m = []  # Entry for each basis factor
        az_U = []  # Unitary transforms that convert from azimuthal 
                   # representation to mixed DVR-FBR
                   # (entries of None indicate identity)
                   
        # 
        # sre_quantum_number contains the azimuthal quantum number
        # of the generic (spin-electronic-)rotational factor.
        #
        # E.g., for standard symmetric top rotational basis functions
        # it equals the negative of the Jz component.
        #
        #
        #
        az_m.append(np.array(sre_quantum_number))  # generic spin-rotation factor
        az_U.append(None) #  Already in azimuthal representation 
        
        # Keep track of the Ellipsis quantum number range needed
        # Initialize these limits to the min/max of -az_m[0] (Note *negative*)
        #
        min_m = min(-az_m[0])
        max_m = max(-az_m[0]) 
        
        # For each basis factor, ...
        for i in range(len(bases)):
            if azimuth[i] is None: # This is not declared an azimuthal coordinate
                #
                # Its aximuthal quantum number is always 0 
                # and it is already in its azimuthal representation
                #
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
        # m = -az_m[0] - sum(all other m's)
        #
        # Remember, +az_m[0] equals k or (k-Lambda), etc.
        #
        # The largest value of m is max(-az_m[0]) - sum( min(other m's) )
        # The smallest value is min(-az_m[0]) - sum(max(other m's)) 
        #
        # These have been kept track of with min_m and max_m in the above
        # loops
        ellipsis_range = np.arange(min_m, max_m + 1)
        
        # If the sign of the ellipsis quantum number does not matter
        # then we can truncate the range further
        if not signed_azimuth:
            max_abs = max(abs(min_m), abs(max_m))
            ellipsis_range = np.arange(0, max_abs + 1) 
            print("The Ellipsis basis is not sign dependent.")
        else:
            print("The Ellipsis basis is sign dependent.")
        
        print("Attempting to calculate Ellipsis bases over m = ")
        print(ellipsis_range)
            
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
        # The rotational entry carries an additional negative sign.
        # Non-azimuthal factors and fixed scalars have values of zero.
        # We now need to determine which combinations of direct products
        # have the correct quantum numbers 
        #
        
        az_grids = np.meshgrid(*az_m, indexing = 'ij') 
        
        non_ellip_total = np.zeros_like(az_grids[0])
        for i in range(len(az_grids)):
            if i != (ellipsis_idx + 1):
                non_ellip_total += -az_grids[i] # note negative sign
        az_e = az_grids[ellipsis_idx + 1] 
        #
        # non_ellip_total is equal to -az_m[0] - Sum' m 
        # where the Sum' is over all non-ellipsis azimuthal 
        # quantum numbers
        # 
        # az_e is the ellipsis factor azimuthal quantum number
        #
        if signed_azimuth:
            sing_val_mask = (az_e == non_ellip_total)
        else:
            sing_val_mask = (az_e == abs(non_ellip_total)) 
            
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
        
        
        return az_m, az_U, az_UH, sing_val_mask, svm_1d, NH, bases_dp, ellipsis_idx
    
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
                    
                    if not self.G_is_zero[kl_idx]:
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
                    
                    if not self.G_is_zero[ab_idx]:
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
                            
                            if not self.G_is_zero[ak_idx]:
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
    
    @staticmethod
    def vectorRME(bases, azimuth, signed_azimuth, fun, X, Y, JX, JY):
        """
        Evaluate reduced matrix elements of a lab-frame vector operator.

        Parameters
        ----------
        bases : list
            The basis set specification.
        azimuth : list
            The azimuthal designations.
        signed_azimuth : bool
            If True, Ellipsis functions are dependent on the sign of the 
            azimuthal quantum number.
        fun : function
            A function that returns the :math:`xyz` body-frame
            components of the vector :math:`V` in terms of the
            coordinates of `bases`.
        X,Y : list of ndarray
            Each element is an array of vectors in 
            the `bases` basis set of a given value of :math:`J` 
            following conventions of the :class:`AzimuthalLinear` 
            Hamiltonian.
        JX, JY : list of integer
            The :math:`J` value for each block of `X` or `Y`.

        Returns
        -------
        VXY : ndarray
            The scaled reduced matrix elements :math:`\\langle X || V || Y \\rangle`.
            See Notes to :func:`NonLinear.vectorRME` for precise definition.

        See Also
        --------
        NonLinear.vectorRME : similar function for :class:`NonLinear` Hamiltonians
        
        """
        
        # Process Azimuthal basis sets for each value of J 
        
        bases_dp_list = [] 
        NH_list = []
        NDP_list = [] 
        svm_1d_list = []
        fbr_shape_list = []
        az_U_list = [] 
        
        Jmax = max( max(JX), max(JY) )
        for j in range(Jmax + 1):
            
            k_azimuth = -np.arange(-j,j+1) # -k for basis order k = -J,...,+J
            az_m, az_U, az_UH, sing_val_mask, svm_1d, NH, bases_dp, ellipsis_idx  = \
                AzimuthalLinear._process_generic_azimuthal_basis(bases, azimuth, k_azimuth, signed_azimuth)
            
            fbr_shape, NV = nitrogen.basis.basisShape(bases_dp)
            NDP = NV * (2*j+1)              
            
            bases_dp_list.append(bases_dp)      # The direct product basis set
            NH_list.append(NH)                  # The working basis size for each J
            NDP_list.append(NDP)                # The size of the direct product rovib azimuthal basis set
            svm_1d_list.append(svm_1d)
            fbr_shape_list.append(fbr_shape)
            az_U_list.append(az_U)

        # Generate the quadrature grid using the 
        # J = 0 basis set. The quadrature grid should be 
        # the same for every value of J anyway.
        Q = nitrogen.basis.bases2grid(bases_dp_list[0]) 
        
        #
        # Now parse the sizes of
        # each block of vectors in X and Y
        #
        nX = [x.shape[1] for x in X]
        nY = [y.shape[1] for y in Y]
       
        
        # Evaluate the vector-valued function.
        #
        Vbf = fun(Q) 
        # Vbf has a shape of (3,) + Q.shape[1:] (the quadrature shape)
        
        # Vbf[0,1,2] are the body-fixed x,y,z axis components
        #
        # Construct the spherical components
        # Vq, q = 0, +1, -1
        #
        #   0  ... z
        #  +1  ... -(x + iy) / sqrt[2]
        #  -1  ... +(x - iy) / sqrt[2]
        #
        # Note that the ordering of the spherical components
        # allow normal array indexing
        Vq = [  Vbf[2],
               -(Vbf[0] + 1j*Vbf[1])/np.sqrt(2.0),
               +(Vbf[0] - 1j*Vbf[1])/np.sqrt(2.0)]
        
        dX,dY = sum(nX), sum(nY) # The total size of the reduced matrix 
        
        # Initialize the reduced matrix
        VXY = np.zeros((dX,dY), dtype = np.complex128)
        
        # 
        # Calculate <X||MU||Y> reduced matrix element
        # block-by-block, including extra scaling.
        
        def block2quad(Z, JZ):
            # transform a block of eigenvectors to its quadrature
            # representation 
            #
            # Transformation steps:
            # 1) Working (single-valued) representation to direct-product azimuthal
            # 2) Azimuthal to mixed DVR/FBR
            # 3) mixed DVR/FBR to quadrature 
            #
            nz = Z.shape[1] # The number of vectors in this block
            
            # Z has shape (NH, nz)
            
            # 1) Convert to direct-product azimuthal representation
            Z_dp = np.zeros((NDP_list[JZ], nz), dtype = np.complex128)
            Z_dp[svm_1d_list[JZ],:] = Z
            Z_dp = np.reshape(Z_dp, (2*JZ+1,) + fbr_shape_list[JZ] + (nz,))
            
            # 2) Transform to mixed DVR/FBR representation 
            #    Include a [None] element to leave last index the same
            #
            Z_fbr = nitrogen.basis.ops.opTensorO(Z_dp, az_U_list[JZ] + [None])
            
            # 3) Transform to quadrature grid
            #    Do ellipsis index first to save space
            Z_qe = bases_dp_list[JZ][ellipsis_idx].basis2grid(Z_fbr, axis = ellipsis_idx + 1)
            Zq = Z_qe 
            for i,b in enumerate(bases_dp_list[JZ]):
                if np.isscalar(b):
                    pass
                elif i != ellipsis_idx:
                    Zq = b.basis2grid(Zq, axis = i + 1) 
                else:
                    pass
            
            return Zq 
            
        
        for i in range(len(nX)):
            
            # Transform X[i] block to quadrature grid 
            xi = block2quad(X[i], JX[i])  # (NJ, quad_shape, nX[i])
            
            for j in range(len(nY)):
                
                # Transform Y[j] block to quadrature grid 
                yj = block2quad(Y[j], JY[j])
                
                
                # idx_x and idx_y will be the array indices
                # of the final VXY reduced matrix
                #
                # Each block starts at the position equal to the 
                # sum of the sizes of the previous blocks
                
                # bi and bj will index the vectors within each block
                
                idx_x = sum(nX[:i])
                for bi in range(nX[i]):
                    
                    idx_y = sum(nY[:j])
                    for bj in range(nY[j]):
                        
                        # Perform summation over body-fixed spherical 
                        # component q and the body-fixed projections
                        # k (of X) and k' (of Y)
                        #
                        for q in [0,1,-1]: # ordering here doesn't matter
                            for k in range(-JX[i], JX[i]+1):
                                for kp in range(-JY[j], JY[j]+1):
                                    
                                    # Check selection rules 
                                    if JX[i] < abs(JY[j]-1) or JX[i] > JY[j] + 1:
                                        # failed triangle rule
                                        continue 
                                    if kp - q != k :
                                        # failed z-component rule
                                        continue 
                                    
                                    # Calculate contribution to reduced
                                    # matrix element
                                    #
                                    factor = (-1)**(JX[i] + kp + q) * \
                                            nitrogen.angmom.wigner3j(2*JX[i], 2*JY[j], 2*1,
                                                                     2*k,    -2*kp   , 2*q)
                                    if factor == 0.0:
                                        continue # final check for zero
                                    
                                    # Compute quadrature sum of the vibrational
                                    # integral
                                    bra = xi[JX[i] + k , ..., bi]
                                    ket = yj[JY[j] + kp, ..., bj]
                                    mid = Vq[-q] 
                                    integral = np.sum(np.conj(bra) * mid * ket) 
                                    
                                    # Finally, 
                                    # include sqrt[2J+1] factor to symmetrize the 
                                    # reduced matrix ! 
                                    #
                                    VXY[idx_x, idx_y] += np.sqrt(2*JX[i]+1) * np.sqrt(2*JY[j] + 1) * factor * integral 
                        
                        idx_y += 1 
                    
                    idx_x += 1 
        
        # VXY is complete
        # return the reduced matrix elements
        
        return VXY 
    
class AzimuthalLinearRT(LinearOperator):
    """
    A general quasi-diabatic spin-rovibronic Hamiltonian for linear molecules.
    
    This Hamiltonian is an extension of :class:`AzimuthalLinear`. The differences
    introduced by the addition of spin-electronic degrees of freedom are
    described in the Notes.
    
    """
     
    def __init__(self, bases, cs, azimuth, 
                 pes = None, masses = None, JJ1 = 1, hbar = None,
                 signed_azimuth = False,
                 NE = 1, Lambda = None, SS1 = None,
                 Li = None, LiLj_ac = None,
                 pesorder = 'LR', ASO = None):
        """

        Parameters
        ----------
        bases : list
            A list of :class:`~nitrogen.basis.GriddedBasis` basis sets for active
            coordinates. Scalar elements will constrain the 
            corresponding coordinate to that fixed value.
        cs : CoordSys
            The coordinate system.
        azimuth : list
            The azimuthal designation of each element of `bases`. Each element
            must be one of None, Ellipsis, or a two-element tuple. See Notes
            for details.
        pes : DFun or function, optional
            The diabatic potential energy matrix, V(q). 
            The function returns the `NE`\ (`NE`\ +1)/2
            elements of the *lower triangle* and the matrix
            is assumed to be Hermitian. 
            The accepts the coordinates defined by `cs` as input. 
            If None (default), no PES is used.
        masses : array_like, optional
            The atomic masses. If None (default), unit masses
            are used. 
        JJ1 : int, optional
            The value of :math:`2J+1`, where :math:`J` is the total angular 
            momentum quantum number. The default value is 1.
        hbar : scalar, optional
            The value of :math:`\\hbar`. If None, the default value in 
            standard NITROGEN units is used (``n2.constants.hbar``).
        signed_azimuth : bool, optional
            If True, then the Ellipsis basis functions depend on the
            sign of the azimuthal quantum number. If False, then 
            the sign is ignored. The default is False.
        NE : int, optional
            The number of electronic states. The default is 1.
        Lambda : array_like of integers, optional
            The electronic angular momentum component about the
            linear (:math:`z`) axis (in units of :math:`\\hbar`). 
            If None (default), all 
            states as assumed to have :math:`\\Lambda = 0`.
        SS1 : array_like of integers, optional
            The spin multiplicity, :math:`2S+1`, of each electronic
            state. If None (default), singlet states will be assumed.
        Li : (3,NE,NE) array_like, optional
            The matrix elements of the body-fixed electronic orbital
            angular momentum, :math:`L_i`, :math:`i=x,y,z`.
            If None (default), :math:`L_z` will be determined
            from `Lambda` and :math:`L_{x,y}` will be assumed to be zero.
        LiLj_ac : (3,3,NE,NE) array_like, optional
            The matrix elements of the anti-commutators, 
            :math:`[L_i,L_j]_+ = L_i L_j + L_j L_i`,
            :math:`i,j=x,y,z`. If None (default), these will be
            approximated from `Li`. 
        pesorder : {'LR', 'LC'}, optional
            V matrix electronic state ordering. The `pes` function returns
            the *lower* triangle of the diabatic potential matrix. If 'LR'
            (default), the values are returned in row major order. If 'LC'
            the values are returned in column major order.
        ASO : (3,3) array_like, optional
            The effective spin-orbit coupling tensor. If None (default),
            this is ignored.

        Notes
        -----
        
        Basis functions are constructed similarly as those of 
        :class:`AzimuthalLinear`. The rotational factor is replaced
        with a Hund's case (b) function
        
        ..  math::
            
            \\vert J N k S \\alpha \\Lambda \\rangle,
        
        where :math:`J`, :math:`N`, and :math:`S` have their usual meaning.
        :math:`k` is the body-fixed projection of :math:`\\mathbf{N}`. The
        electronic states are labeled by an index :math:`\\alpha`, and each
        electronic state has a signed integer :math:`z`-projection of the 
        electronic orbital
        angular momentum :math:`\\Lambda` at linear geometries. 
        
        The rotational kinetic energy operator is modified by replacing
        :math:`J_\\alpha` with :math:`N_\\alpha - L_\\alpha`. Derivatives
        of the electronic orbital angular momentum matrix elements with
        respect to nuclear coordinates are ignored, consistent with the
        quasi-diabatic ansatz. 
        
        The linear boundary
        conditions are enforced by selecting only basis functions for which
        
        ..  math::
        
            k - \\Lambda - \\sum_i m_i = 0.
        
        The azimuthal quantum numbers for each basis factor are assigned according
        to the `azimuth` list in the same way as :class:`AzimuthalLinear`.
        
        The electronic orbital angular momentum matrix elements
        are supplied by optional parameters. They are assumed to be 
        constant.
        Geometry-dependent quenching functions may be added in the future.
        
        
        Spin orbit interactions are currently included with an effective
        :math:`\\Delta S = 0` interaction term 
        
        ..  math::
            
            (A_{SO})_{\\alpha \\beta} L_\\alpha S_\\beta
        
        Unless spin interactions are included, there is no need to use 
        non-zero spin, and it is most efficient to treat all states as
        singlets. 
        
        
        """
        
        print("***************************************")
        print("Preparing AzimuthalLinearRT Hamiltonian")
        print("")
        
        # Construct the case (b) spin-rotation-electronic 
        # basis set list.
        Lambda, SS1, sre_basis_list  = AzimuthalLinearRT._process_sre_basis(JJ1, NE, Lambda, SS1)
        Nsre = sre_basis_list.shape[0] # The total number of sre functions
        #
        # The sre azimuthal quantum number is -k + Lambda
        #
        sre_azimuthal = -sre_basis_list[:,3] + sre_basis_list[:,1] 
        
        # Now process the entire sre-vibrational basis 
        az_m, az_U, az_UH, sing_val_mask, svm_1d, NH, bases_dp, ellipsis_idx  = \
            AzimuthalLinear._process_generic_azimuthal_basis(bases, azimuth, sre_azimuthal, signed_azimuth)
            
        fbr_shape, NV = nitrogen.basis.basisShape(bases_dp)
        axis_of_coord = nitrogen.basis.coordAxis(bases_dp)
        vvar = nitrogen.basis.basisVar(bases_dp)
        #coord_k_is_ellip_coord = [None for i in range(cs.nQ)]
        
        NDP = Nsre * NV # The size of the [spin-rot-elec]-vib direct product basis
                        # (which contains all ellipsis basis functions, 
                        #  not selected by which actually occur in the working basis)
                      
        # Check that there is at least one active coordinate        
        if len(vvar) == 0:
            raise ValueError("there must be an active coordinate!") 
            
        print(f"Total basis set size = {NH:d}")
            
        ################################################    
        # Evaluate quadrature grid quantities
        
        Q = nitrogen.basis.bases2grid(bases_dp) 
        
        ########################################
        # Evaluate potential energy function on 
        # quadrature grid 
        if pes is None:
            Vij = None # No PES 
        else:
            #
            # Calculate the lower triangle
            # of the Hermitian V matrix
            #
            try: # Attempt DFun interface
                Vtri = pes.f(Q, deriv = 0)[0]
            except:
                Vtri = pes(Q) # Attempt raw function
            
            idx = 0 
            
            Vij = [[None for j in range(NE)] for i in range(NE)]
            
            
            if pesorder == 'LR':
                # The function provides the lower triangle
                # in row major order
                for i in range(NE):
                    for j in range(i+1):
                        
                        Vij[i][j] = Vtri[idx]
                        if i != j:
                            Vij[j][i] = np.conj( Vij[i][j] ) # Hermitian
                        
                        idx = idx + 1 
                        
            elif pesorder == 'LC':
                # Column major order
                for j in range(NE):
                    for i in range(j,NE):
                        
                        Vij[i][j] = Vtri[idx]
                        if i != j:
                            Vij[j][i] = np.conj( Vij[i][j] ) # Hermitian
                        
                        idx = idx + 1
                        
            else:
                raise ValueError("Invalid pesorder")
            
        
        
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
        # Determine which elements of G are strictly zero
        G_is_zero = [np.max(abs(G[i])) < 1e-10 for i in range(G.shape[0])]
        
        #
        # Calculate the log. deriv. of det(g)
        #
        gtilde = [nitrogen.linalg.packed.trAB_sp(G, g[i+1]) for i in range(len(vvar))]
        gtilde = np.stack(gtilde)
        
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
        # in the multi-state case (b) basis
        #
        #
        Ni = nitrogen.angmom.caseb_multistate_N(sre_basis_list[:,0], # elec. index 
                                                sre_basis_list[:,2], # N
                                                sre_basis_list[:,3], # k
                                                sre_basis_list[:,4], # 2S+1
                                                sre_basis_list[:,5]) # 2J+1
        
        Si = nitrogen.angmom.caseb_multistate_S(sre_basis_list[:,0], # elec. index 
                                                sre_basis_list[:,2], # N
                                                sre_basis_list[:,3], # k
                                                sre_basis_list[:,4], # 2S+1
                                                sre_basis_list[:,5]) # 2J+1
        
        #
        # Construct body-fixed L_i operators and 
        # their products 
        #
        # We are supplied with Li and {Li,Lj} in 
        # the Ne x Ne electronic representation
        #
        # If Li is not supplied, assume that 
        # L_z is Lambda and the orthogonal components
        # are zero
        if Li is None: 
            print("Li is being constructed from Lambda")
            Lz = np.diag(1.0 * np.array(Lambda)) # Lambda 
            Lx = np.zeros((NE,NE))
            Ly = np.zeros((NE,NE))
            Li = [Lx, Ly, Lz]
        Li_e = np.array(Li) # Pure electronic representation
        if not np.allclose(np.diag(Li_e[2]), Lambda):
            warnings.warn("Lz appears inconsistent with Lambda")
            
        if LiLj_ac is None:
            print("{Li,Lj} is being constructed from Li")
            # Calculate anti-commutator assuming closure
            # [Li, Lj]_+ = LiLj + LjLi
            LiLj_ac = [ 
                [ Li[i] @ Li[j] + Li[j] @ Li[i] for j in range(3)] 
                    for i in range(3)]
        LiLj_ac_e = np.array(LiLj_ac) # Pure electronic representation 
        
        #
        # Expand these to the full Nsre x Nsre
        # spin-rot-electronic representation
        # These operators are diagonal in J, N
        # k, and S.
        # 
        Li, LiLj_ac = nitrogen.angmom.caseb_multistate_L(Li_e, LiLj_ac_e, 
                                                         sre_basis_list[:,0], # elec. index 
                                                         sre_basis_list[:,2], # N
                                                         sre_basis_list[:,3], # k
                                                         sre_basis_list[:,4], # 2S+1
                                                         sre_basis_list[:,5]) # 2J+1
        
        ###########################
        # Process ASO tensor
        if ASO is None:
            ASO = np.zeros((3,3))
        ASO = np.array(ASO)
        if ASO.shape != (3,3):
            raise ValueError("ASO must be a constant (3,3) array")
        print("")
        print("The ASO spin-orbit tensor is ")
        print(ASO)
        print("")
        #
        ###########################
        
        
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
        
        self.sre_basis_list = sre_basis_list 
        self.Nsre = Nsre
        self.JJ1 = JJ1 
        self.Ni = Ni 
        self.Si = Si 
        self.Li = Li 
        self.LiLj_ac = LiLj_ac 
        self.ASO = ASO 
        
        self.bases_dp = bases_dp
        self.ellipsis_idx = ellipsis_idx 
        
        self.Vij = Vij 
        
        self.axis_of_coord = axis_of_coord 
        self.Gammatilde = Gammatilde 
        self.G = G 
        self.G_is_zero = G_is_zero 
        self.hbar = hbar 
        self.nact = len(vvar) # The number of active coordinates 
        
        return 
    
    @staticmethod 
    def _process_sre_basis(JJ1, NE, Lambda, SS1):
                                                     
        """
        see AzimuthalLinear._process_basis for more information
        
        Returns
        -------
        Lambda : array_like
            The value of :math:`\\Lambda` for each of the `NE` electronic states
        SS1 : array_like
            The value of :math:`2S+1` for each of the `NE` electronic states.
        sre_basis_list : ndarray
            The case (b) quantum numbers. The column order is 
            electronic index, Lambda, N, k, 2S+1, 2J+1.
        """
        
        ######################################################
        # Generate spin-rotation-electronic basis set list
        # and quantum numbers
        #
        if NE < 1:
            raise ValueError("NE must be a positive integer")
        
        # Defaults:
        # Lambda --> All Sigma states
        # S --> All singlet states
        if Lambda is None:
            Lambda = (0,) * NE # The z-axis component of L
        if SS1 is None:
            SS1 = (1,) * NE    # The spin multiplicity of each state 
        
        if len(Lambda) != NE:
            raise ValueError('Lambda must have NE entries')
        if len(SS1) != NE:
            raise ValueError('SS1 must have NE entries')
        
        # Check that the half-integer or integer
        # character of J and S are consistent. Their multiplicities
        # must both be even or odd, so their sum must be even
        #
        if JJ1 < 1:
            raise ValueError("J multiplicity must be a positive integer")
            
        for ss1 in SS1:
            if (JJ1 + ss1) % 2 == 1:
                raise ValueError("J and S are not compatible")
            if ss1 < 1:
                raise ValueError("S multiplicity must be a positive integer")
        #
        # Now, construct the list of case (b) quantum numbers
        # for each electronic state
        sre_basis_list = [] 
        # The column order is 
        # electronic index, Lambda, N, k, 2S+1, 2J+1
        #
        for iE in range(NE):
            
            ss1 = SS1[iE] # 2S + 1 for this electronic state
            
            # Calculate the allowed range of the N
            # quantum number, which is strictly integer
            Nmin = abs(JJ1 - ss1) // 2
            Nmax = (JJ1 + ss1) // 2 - 1 
            
            for Nval in range(Nmin, Nmax + 1):
                # For each value of N, we have (2*N+1)
                # values of k, the signed z-axis projection
                # (using the usual "anomalous" convention)
                
                for k in range(-Nval, Nval+1):
                    qns = [iE, Lambda[iE], Nval, k, ss1, JJ1]
                    sre_basis_list.append(qns)
                    
        sre_basis_list = np.array(sre_basis_list)
        # The number of spin-rot-electronic basis functions
        Nsre = sre_basis_list.shape[0]
        
        
        print("Spin-rotation-electronic basis")
        print("------------------------------")
        print(f" Total # of sre functions = {Nsre:d}")
        print("------------------------------")
        print(" E-idx Lambda N   k    S    J ")
        print("------------------------------")
        eidx_prev = -1 
        for i in range(Nsre):
            qn = sre_basis_list[i]
            Sstr = f" {(qn[4]-1)//2:2d}" if qn[4] % 2 == 1 else f"{(qn[4]-1)/2:4.1f}"
            Jstr = f" {(qn[5]-1)//2:2d}" if qn[5] % 2 == 1 else f"{(qn[5]-1)/2:4.1f}"
            
            if eidx_prev != qn[0]:
                print(f"  {qn[0]:2d}    {qn[1]:2d}   ",end="")
            else:
                print("             ", end = "")
                
            print(f"{qn[2]:2d}  {qn[3]:2d}  {Sstr:4s} {Jstr:4s} ")
            
            eidx_prev = qn[0]
            
        print("------------------------------")
        
        return Lambda, SS1, sre_basis_list 
    
    def toDirectProduct(self, x):
        """
        Transform a vector in the working basis representation to the
        direct product azimuthal basis .
        
        Parameters
        ----------
        x : (NH,...) ndarray
            A set of vectors in the working representationg.

        Returns
        -------
        y : (Nsre,) + fbr_shape + (...)
            The vectors transformed to the direct product azimuthal representation

        """
        
        # Transform a vector in the working basis representation 
        #
        base_shape = x.shape[1:]
        
        x_dp = np.zeros((self.NDP,)+base_shape, dtype = x.dtype)
        x_dp[self.svm_1d,...] = x 
        x_dp = np.reshape(x_dp, (self.Nsre,) + self.fbr_shape + base_shape) 
        
        return x_dp
    
    def toSREV(self, x):
        """
        Transform a vector in the working basis representation to the
        SRE basis and a vibrational factor.

        Parameters
        ----------
        x : (NH,...) ndarray
            A set of vectors in the working representation.

        Returns
        -------
        y : (Nsre,NV,...)
            The vectors transformed to the direct product azimuthal representation

        """
        
        # Transform a vector in the working basis representation 
        #
        base_shape = x.shape[1:]
        
        x_dp = np.zeros((self.NDP,)+base_shape, dtype = x.dtype)
        x_dp[self.svm_1d,...] = x 
        x_dp = np.reshape(x_dp, (self.Nsre,self.NV) + base_shape) 
        
        return x_dp
        
    def calcSREop(self, X, SRE_op):
        """
        Calculate the matrix elements of a SRE operator using vectors
        in the working representation
        
        Parameters
        ----------
        X : (NH,n) ndarray
            A set of vectors in the working representation.
        SRE_op : (Nsre, Nsre) ndarray
            The matrix representation of an operator in the SRE basis
        
        Returns
        -------
        M : (n,n) ndarray
            The matrix elements of the operator.
        """
        
        # First, transform the working vectors in the SRE-V direct product
        Y = self.toSREV(X)
        # Y has shape (Nsre, NV, n)
        
        #  n = X.shape[1] # The number of vectors 
        
        OY = np.tensordot(SRE_op, Y, axes = 1) # shape (Nsre, NV, n)
        
        M = np.einsum('abi,abj',np.conj(Y), OY) # Contract 
        
        return M 
    
    def calcSREpop(self, X):
        """
        Calculate the population of each SRE basis function of vectors
        in the working representation

        Parameters
        ----------
        X : (NH,n) ndarray
            A set of vectors in the working representation.

        Returns
        -------
        C : (Nsre,n) ndarray
            The weights of each vector in the SRE space.

        """
        
        # First, transform the working vectors in the SRE-V direct product
        Y = self.toSREV(X)
        # Y has shape (Nsre, NV, n)
        
        # Square and sum along the vibrational index
        C = np.sum(np.abs(Y)**2, axis = 1)
        
        return C 
    
    def calcVibDensity(self, X):
        """
        Calculate the vibrational density in mixed DVR/FBR basis
        set factor

        Parameters
        ----------
        X : (NH,n) ndarray
            A set of vectors in the working representation

        Returns
        -------
        None.

        """
        
        # First, transform the working representation 
        # to the direct product azimuthal basis 
        #
        x_dp = self.toDirectProduct(X)
        #
        # Then, then transform to the mixed DVR/FBR basis
        #
        x_fbr = nitrogen.basis.ops.opTensorO(x_dp, self.az_U + [None]) 
        #
        # x_fbr: [SRE index, // mixed DVR/FBR indices // , ...]
        nfactors = len(self.fbr_shape)
        
        vib_dens = [] 
        for i in range(nfactors):
            axes = np.arange(x_fbr.ndim - 1)
            axes = axes[axes != i+1] 
            # Sum over all indices except that of the factor of interest
            # and the final index
            #
            vib_dens.append(np.sum(abs(x_fbr)**2, axis = tuple(axes)))
        
        return vib_dens 
        
        
    def _matvec(self, x):
        
        Nsre = self.Nsre
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
        #
        # The leading axis is the spin-rotation-electronic factor. The
        # remaining axes are the mixed DVR/FBR vibrational grid shape
        #
        x_dp = np.reshape(x_dp, (Nsre,) + self.fbr_shape) 
        
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
        # Remember that the first axis of x_fbr is the spin-rot-elec. index
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
        if self.Vij is not None:
            for sre1 in range(Nsre):
                qn1 = self.sre_basis_list[sre1,:]
                
                for sre2 in range(Nsre):
                    qn2 = self.sre_basis_list[sre2,:]
                    
                    # Loop through all spin-rot-elec blocks
                    # The diabatic potential is diagonal in
                    # the N, k, S and J quantum numbers
                    #
                    # qn = [e-idx, Lambda, N, k, SS1, JJ1]
                    # 
                    if np.any( qn1[2:6] != qn2[2:6] ):
                        continue # Zero 
                    
                    # Apply the diabatic coupling surface
                    yq[sre1] += self.Vij[qn1[0]][qn2[0]] * xq[sre2]

        #########################
        
        #########################
        #
        # Kinetic energy operator 
        #
        # 1) Pure vibrational kinetic energy
        #    Diagonal in spin-rotational-electronic index
        #    (by quasi-diabatic ansatz)
        
        for sre in range(Nsre):
            # Spin-rotation-elec. block `sre`
            for lact in range(nact):
                # calculate dtilde_l acting on wavefunction,
                # result in the quadrature representation 
                
                #
                # dtilde_l is the sum of the derivative 
                # and one-half Gammatilde_l
                #
                dtilde_l = dxq[lact][sre] + 0.5 * self.Gammatilde[lact] * xq[sre]
                
                for kact in range(nact):
                    
                    # Get the packed-storage index for 
                    # G^{kactive, lactive}
                    kl_idx = nitrogen.linalg.packed.IJ2k(kact, lact)
                    
                    if not self.G_is_zero[kl_idx]:
                        Gkl = self.G[kl_idx]
                        
                        # Apply G^kl 
                        Gkl_dl = Gkl * dtilde_l
                        
                        # Now finish with (dtilde_k)
                        # and put results in quadrature representation
                        #
                        # include the final factor of -hbar**2 / 2
                        yq[sre] += (hbar**2 * 0.25) * self.Gammatilde[kact] * Gkl_dl 
                        dyq[kact][sre] += (hbar**2 * 0.50) * Gkl_dl 

        #      
        # Rotational-electronic angular momentum terms
        #
        # As part of the diabatic ansatz, we ignore
        # vibrational derivatives of the matrix elements of
        # the electronic angular momentum. 
        #
        # 2) Pure N-L kinetic energy
        #
        #  -hbar**2/4  *  [i(Na-La)/hbar, i(Nb-Lb)/hbar]_+  *  G^ab 
        #
        # We further assume that L_i is closed within the
        # electronic space, i.e. that the matrix representation
        # of its products equal the product of its matrix
        # representation. This is already true for N_a.
        #
        #
        #
        for a in range(3):
            for b in range(3):
                # Because both [i(N-L)a,i(N-L)b]_+ and G^ab
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
                
                if not self.G_is_zero[ab_idx]:
                    
                    Gab = self.G[ab_idx] 
                
                    #
                    # {Na-La, Nb-Lb} = 
                    #  {Na, Nb} + {La, Lb} - 2(NaLb + NbLa)
                    # 
                    # (N and L commute)
                    # N has closure
                    N_ac = self.Ni[a] @ self.Ni[b] + self.Ni[b] @ self.Ni[a]
                    L_ac = self.LiLj_ac[a,b]
                    cross = self.Ni[a] @ self.Li[b] + self.Ni[b] @ self.Li[a] 
                    
                    anticom = -(N_ac + L_ac - 2*cross)  # {1j(Na-La), 1j(Nb-Lb)}
                                                        # (minus sign from 1j*1j)
                            
                    for srep in range(Nsre):
                        for sre in range(Nsre):
                            #
                            # < sre' | ... |  sre > spin-rot-elec block
                            #
                            sre_me = anticom[srep,sre]
                            
                            if sre_me == 0:
                                continue # a zero spin-rot-elec matrix element 
                            # otherwise, add contribution from
                            # effective inverse inertia tensor
                            yq[srep] += (symfactor * sre_me * (-hbar**2) * 0.25) * (Gab * xq[sre])
                        
        #
        # 3) Rotation-vibration coupling
        #
        # -hbar**2 / 2 (i(Na-La)/hbar) * G^ak [psi' (dtildek psi) - (dtildek psi') psi]
        
        for a in range(3):
            for srep in range(Nsre):
                for sre in range(Nsre):
                    
                    #
                    # Calculate the spin-rot-elec matrix element
                    # of i*(Na-La)/hbar
                    #
                    sre_me = 1j * (self.Ni[a][srep,sre] - self.Li[a][srep,sre])
                    if sre_me == 0:
                        continue 
                    
                    for kact in range(nact):
                        #
                        # Vibrational coordinate k
                        #

                        # calculate index of G^ak
                        ak_idx = nitrogen.linalg.packed.IJ2k(nact + a, kact)
                        
                        if not self.G_is_zero[ak_idx]:
                            
                            Gak = self.G[ak_idx] 
                            
                            # First, do the psi' (dtilde_k psi) term
    
                            dtilde_k = dxq[kact][sre] + 0.5 * self.Gammatilde[kact] * xq[sre]
                            yq[srep] += (sre_me * (-hbar**2) * 0.50) * Gak * dtilde_k
                            
                            # Now, do the -(dtilde_k psi') * psi term
                            Gak_xq = Gak * xq[sre] 
                            yq[srep] += (sre_me * (+hbar**2) * 0.25) * self.Gammatilde[kact] * Gak_xq
                            dyq[kact][srep] += (sre_me * (+hbar**2) * 0.50) * Gak_xq
                            
        #####################################
        # Spin terms
        #
        # A_SO spin-orbit coupling interaction
        for a in range(3):
            for b in range(3):
                #
                # ASO_a,b  * L_a * S_b  term
                #
                LaSb = self.Li[a] @ self.Si[b] # The L.S operator in sre basis
                
                for srep in range(Nsre):
                    for sre in range(Nsre):
                        
                        #
                        # Calculate the spin-orbit matrix element
                        #
                        sre_me = self.ASO[a,b] * LaSb[srep,sre] 
                        
                        if sre_me == 0:
                            continue 
                        
                        yq[srep] += sre_me * xq[sre] 
        
        #
        #####################################
        
        
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
    
    @staticmethod
    def vectorRME(bases, azimuth, signed_azimuth, NE, Lambda, SS1, fun, X, Y, JJ1X, JJ1Y,
                  funorder = 'LR'):
        """
        Evaluate reduced matrix elements of a lab-frame vector operator.

        Parameters
        ----------
        bases : list
            The basis set specification.
        azimuth : list
            The azimuthal designations.
        signed_azimuth : bool
            If True, Ellipsis functions are dependent on the sign of the 
            azimuthal quantum number.
        NE : integer
            The number of electronic states 
        Lambda : array_like
            The Lambda values for each electronic state
        SS1 : array_like
            The 2S+1 values for each electronic state
        fun : function
            A function that returns the electronic matrix elements
            of the :math:`xyz` body-frame
            components of the vector :math:`V` in terms of the
            coordinates of `bases`. The return shape should be
            (3,NE*(NE+1)/2,...)
            Note, we assume :math:`V` is Hermitian in the
            electronic basis. 
        X,Y : list of ndarray
            Each element is an array of vectors in 
            the `bases` basis set of a given value of :math:`J` 
            following conventions of the :class:`AzimuthalLinear` 
            Hamiltonian.
        JJ1X, JJ1Y : list of integer
            The :math:`2J+1` value for each block of `X` or `Y`.
        funorder : {'LR', 'LC'}, optional
            :math:`V` matrix electronic state ordering. The `fun` function returns
            the *lower* triangle of the diabatic vector-operator matrix. If 'LR'
            (default), the values are returned in row major order. If 'LC'
            the values are returned in column major order.

        Returns
        -------
        VXY : ndarray
            The scaled reduced matrix elements :math:`\\langle X || V || Y \\rangle`.
            See Notes to :func:`NonLinear.vectorRME` for precise definition of
            the scaling.

        See Also
        --------
        NonLinear.vectorRME : similar function for :class:`NonLinear` Hamiltonians
        
        """
        
        # Process Azimuthal basis sets for each value of J 
        
        bases_dp_list = [] 
        NH_list = []
        NDP_list = [] 
        svm_1d_list = []
        fbr_shape_list = []
        az_U_list = [] 
        Nsre_list = [] 
        sre_basis_list_list = []
        
        JJ1max = max( max(JJ1X), max(JJ1Y) )
        
        # 2J+1 is either 1, 3, 5, ...
        # or 2, 4, 6, ...
        # The starting value is 2 - (JJ1max % 2)
        jj1_list = list(range(2-(JJ1max%2), JJ1max + 2, 2))
        
        for jj1 in jj1_list:
            
            Lambda, SS1, sre_basis_list  = AzimuthalLinearRT._process_sre_basis(jj1, NE, Lambda, SS1)
            Nsre = sre_basis_list.shape[0] # The total number of sre functions
            #
            # The sre azimuthal quantum number is -k + Lambda
            #
            sre_azimuthal = -sre_basis_list[:,3] + sre_basis_list[:,1] 
            
            az_m, az_U, az_UH, sing_val_mask, svm_1d, NH, bases_dp, ellipsis_idx  = \
                AzimuthalLinear._process_generic_azimuthal_basis(bases, azimuth, sre_azimuthal, signed_azimuth)
            
            fbr_shape, NV = nitrogen.basis.basisShape(bases_dp)
            NDP = NV * Nsre           
            
            bases_dp_list.append(bases_dp)      # The direct product basis set
            NH_list.append(NH)                  # The working basis size for each J
            NDP_list.append(NDP)                # The size of the direct product sre-vib azimuthal basis set
            svm_1d_list.append(svm_1d)
            fbr_shape_list.append(fbr_shape)
            az_U_list.append(az_U)
            Nsre_list.append(Nsre)
            sre_basis_list_list.append(sre_basis_list)
        
        # We need the direction cosine reduced matrix elements between
        # each jj1 block
        
        dircos = [[None for iidx in range(len(jj1_list))] for jidx in range(len(jj1_list))]
        
        for iidx in range(len(jj1_list)):
            for jidx in range(len(jj1_list)):
                # 0                 1       2  3   4     5
                # electronic index, Lambda, N, k, 2S+1, 2J+1.
                
                dircos[iidx][jidx] = nitrogen.angmom.caseb_multistate_dircos(
                    sre_basis_list_list[iidx][:,2],
                    sre_basis_list_list[iidx][:,3],
                    sre_basis_list_list[iidx][:,4],
                    sre_basis_list_list[iidx][:,5],
                    sre_basis_list_list[jidx][:,2],
                    sre_basis_list_list[jidx][:,3],
                    sre_basis_list_list[jidx][:,4],
                    sre_basis_list_list[jidx][:,5])
        
        # Generate the quadrature grid using the 
        # first JJ1 basis set. The quadrature grid should be 
        # the same for every value of J anyway.
        Q = nitrogen.basis.bases2grid(bases_dp_list[0]) 
        
        #
        # Now parse the sizes of
        # each block of vectors in X and Y
        #
        nX = [x.shape[1] for x in X]
        nY = [y.shape[1] for y in Y]
       
        
        ###################################
        # Evaluate vector-valued matrix elements
        # over th equadrature grid 
        Vtri = fun(Q) # The lower triangle, in row of column-major order Vtri[xyz,idx,...]
        idx = 0 
        Vaij = [[[None for j in range(NE)] for i in range(NE)] for a in range(3)]
        if funorder == 'LR':
            # The function provides the lower triangle
            # in row major order
            for i in range(NE):
                for j in range(i+1):
                    
                    for a in range(3):
                        Vaij[a][i][j] = Vtri[a,idx]
                        if i != j:
                            Vaij[a][j][i] = np.conj( Vaij[a][i][j] ) # Hermitian
                    
                    idx = idx + 1 
                    
        elif funorder == 'LC':
            # Column major order
            for j in range(NE):
                for i in range(j,NE):
                    
                    Vaij[a][i][j] = Vtri[a,idx]
                    if i != j:
                        Vaij[a][j][i] = np.conj( Vaij[a][i][j] ) # Hermitian
                    
                    idx = idx + 1
                    
        else:
            raise ValueError("Invalid funorder")
            
        
        
        
        # Vaij[0,1,2] are the body-fixed x,y,z axis components
        #
        # Construct the spherical components
        # Vq, q = 0, +1, -1
        #
        #   0  ... z
        #  +1  ... -(x + iy) / sqrt[2]
        #  -1  ... +(x - iy) / sqrt[2]
        #
        # Note that the ordering of the spherical components
        # allow normal array indexing
        
        Vq = [ [[  Vaij[2][i][j]                                  for j in range(NE)] for i in range(NE)],
               [[-(Vaij[0][i][j] + 1j*Vaij[1][i][j])/np.sqrt(2.0) for j in range(NE)] for i in range(NE)],
               [[+(Vaij[0][i][j] - 1j*Vaij[1][i][j])/np.sqrt(2.0) for j in range(NE)] for i in range(NE)] ]
        
        dX,dY = sum(nX), sum(nY) # The total size of the reduced matrix 
        
        # Initialize the reduced matrix
        VXY = np.zeros((dX,dY), dtype = np.complex128)
        
        # 
        # Calculate <X||MU||Y> reduced matrix element
        # block-by-block, including extra scaling.
        
        def block2quad(Z, jidx):
            # transform a block of eigenvectors to its quadrature
            # representation 
            #
            # Transformation steps:
            # 1) Working (single-valued) representation to direct-product azimuthal
            # 2) Azimuthal to mixed DVR/FBR
            # 3) mixed DVR/FBR to quadrature 
            #
            nz = Z.shape[1] # The number of vectors in this block
            
            # Z has shape (NH, nz)
            
            # 1) Convert to direct-product azimuthal representation
            Z_dp = np.zeros((NDP_list[jidx], nz), dtype = np.complex128)
            Z_dp[svm_1d_list[jidx],:] = Z
            Z_dp = np.reshape(Z_dp, (Nsre_list[jidx],) + fbr_shape_list[jidx] + (nz,))
            
            # 2) Transform to mixed DVR/FBR representation 
            #    Include a [None] element to leave last index the same
            #
            Z_fbr = nitrogen.basis.ops.opTensorO(Z_dp, az_U_list[jidx] + [None])
            
            # 3) Transform to quadrature grid
            #    Do ellipsis index first to save space
            Z_qe = bases_dp_list[jidx][ellipsis_idx].basis2grid(Z_fbr, axis = ellipsis_idx + 1)
            Zq = Z_qe 
            for i,b in enumerate(bases_dp_list[jidx]):
                if np.isscalar(b):
                    pass
                elif i != ellipsis_idx:
                    Zq = b.basis2grid(Zq, axis = i + 1) 
                else:
                    pass
            
            return Zq 
            
        # iidx and jidx are the indices in the
        # precalculated sre basis set list
        for i in range(len(nX)):
            
            # Transform X[i] block to quadrature grid 
            iidx = jj1_list.index(JJ1X[i])
            xi = block2quad(X[i], iidx)  # (Nsre, quad_shape, nX[i])
            
            for j in range(len(nY)):
                
                # Transform Y[j] block to quadrature grid 
                jidx = jj1_list.index(JJ1Y[j])
                yj = block2quad(Y[j], jidx)
                
                
                # idx_x and idx_y will be the array indices
                # of the final VXY reduced matrix
                #
                # Each block starts at the position equal to the 
                # sum of the sizes of the previous blocks
                
                # bi and bj will index the vectors within each block
                
                idx_x = sum(nX[:i])
                for bi in range(nX[i]):
                    
                    idx_y = sum(nY[:j])
                    for bj in range(nY[j]):
                        
                        # Perform summation over body-fixed spherical 
                        # component q and the body-fixed projections
                        # k (of X) and k' (of Y)
                        #
                        for q in [0,1,-1]: # ordering here doesn't matter
                        
                            for p in range(Nsre_list[iidx]):
                                for pp in range(Nsre_list[jidx]):
                                    
                                    factor = (-1)**q * dircos[iidx][jidx][q,p,pp]
                                    if factor == 0.0:
                                        continue # check sre selection rule
                                    
                                    # Get the electronic state indices
                                    eidx_i = sre_basis_list_list[iidx][p,0]
                                    eidx_j = sre_basis_list_list[jidx][pp,0]
                                    
                                    # Compute quadrature sum of the vibrational
                                    # integral
                                    bra = xi[p , ..., bi] # The bra vibrational factor
                                    ket = yj[pp, ..., bj] # The ket vibrational factor 
                                    mid = Vq[-q][eidx_i][eidx_j] # The electronic matrix element, note -q
                                    integral = np.sum(np.conj(bra) * mid * ket) 
                                    
                                    # Finally, 
                                    # include sqrt[2J+1] factor to scale
                                    #
                                    VXY[idx_x, idx_y] += np.sqrt(JJ1X[i]) * factor * integral 
                        
                        idx_y += 1 
                    
                    idx_x += 1 
        
        # VXY is complete
        # return the reduced matrix elements
        
        return VXY 