"""
core rovibrational Hamiltonians and operators
for tensor and configuration (VSCF/VMP2/VCI) methods
"""

import nitrogen 
import nitrogen.tensor as tensor 
import numpy as np 

__all__ = ['NonLinearTO'] 



class NonLinearTO(tensor.TensorOperator):
    """
    
    A simple, full-dimensional quadrature Hamiltonian
    tensor operator for non-linear molecules
    
    Attributes
    ----------
    H : TensorOperator
        The wrapped Hamlitonian tensor operator
    Crot : (3,3) nested list
        The TensorOperator coefficients of the pure rotational
        kinetic energy operator
    Crv : (3,) list
        The TensorOperator coefficients of the rotation-vibration
        kinetic energy operator
    
    """
    
    def __init__(self, bases, cs, pes, masses):
        
        """
        
        Parameters
        ----------
        bases : list of GriddedBasis or scalar
            The basis set factors
        cs : CoordSys
            The coordinate system
        pes : DFun or function
            The potential energy surface 
        masses : list
            The atomic masses 
        
        """
        
        # Construct the full quadrature grid
        # 
        # This method returns Q with singleton axes
        # for scalar entries of `bases`
        Q = nitrogen.basis.bases2grid(bases)
        #
        # We need to squeeze the singleton quadrature
        # grid axes out, because they are not included 
        # as tensor indices
        #
        scalar_axes = [i + 1 for i in range(len(bases)) if np.isscalar(bases[i])]
        Q = np.squeeze(Q, axis = tuple(scalar_axes))
        
        
        # Evaluate the potential energy 
        # over the squeezed quadrature grid
        #
        # A DFun or normal function can be used
        #
        try: # Attempt DFun
            Vq = pes.f(Q, deriv = 0)[0,0]
        except:
            Vq = pes(Q) # Attempt raw function
            
        # Get the active coordinates indices
        # (i.e. those for which we need derivatives)
        vvar = nitrogen.basis.basisVar(bases) 
        
        #
        # Calculate the metric tensor and its first derivatives
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
        
        # Partition the vibration, rotation-vibration, and rotational
        # blocks of G
        
        # Store views into each element
        nv = len(vvar)
        
        Gvib = [[G[nitrogen.linalg.packed.IJ2k(i,j)]      for i in range(nv)] for j in range(nv)]
        Grv  = [[G[nitrogen.linalg.packed.IJ2k(nv + i, j)] for i in range(3)] for j in range(nv)]
        Grot = [[G[nitrogen.linalg.packed.IJ2k(nv + i, nv + j)] for i in range(3)] for j in range(3)]
    
        # Calculate the integration volume weight
        # logarithmic derivatives
        rhotilde = nitrogen.basis.calcRhoLogD(bases, Q)
         
        # Calculate the final derivative values
        Gammatilde = rhotilde - 0.5 * gtilde  # active only
        
        ######################################
        # Collect the quadrature transformation matrices
        # for each basis set factor
        #
        quad_transform = nitrogen.basis.collectBasisQuadrature(bases)
        #
        # and the transformations of the derivative functions
        deriv_transform, factorOfCoord = nitrogen.basis.collectBasisDerivativeQuadrature(bases)
        
        dquad_transforms = [] 
        for i in range(nv):
            # All factors not involving this coordinate use their
            # normal quadrature transformation
            trans = [item for item in quad_transform]
            
            # Now replace the transformation for the approriate
            # index with the derivative transformation for that coordinate
            trans[factorOfCoord[i]] = deriv_transform[i] 
            
            dquad_transforms.append(trans)
        
        # Form the tensor operators for term in the Hamiltonian
        #
        terms = [] 
        #
        # Potential energy operator 
        terms.append(tensor.QuadratureOperator(Vq, quad_transform, quad_transform))
        
        # Kinetic energy operator
        # 
        # hbar**2/2  [dk + 1/2 Gamma_k](psi) * Gkl * [dl + 1/2 Gamma_l][psi]
        #
        hbar = nitrogen.constants.hbar 
        
        # 1) dk * Gkl * dl type terms
        for i in range(nv):
            for j in range(nv):
                grid = hbar**2/2.0 * Gvib[i][j]
                left = dquad_transforms[i]
                right = dquad_transforms[j]
                term = tensor.QuadratureOperator(grid, left, right)
                terms.append(term)
        
        # 2) Single derivative (U_l) terms
        for i in range(nv):
            #
            # Calculate hbar**2/2 * Ui 
            Ui = 0
            for j in range(nv):
                Ui += hbar**2 / 2.0 * (1/2 * Gammatilde[j]) * Gvib[i][j]
            
            terms.append(tensor.QuadratureOperator(Ui, quad_transform, dquad_transforms[i]))
            terms.append(tensor.QuadratureOperator(Ui, dquad_transforms[i], quad_transform))
        
        # 3) Scalar term (VT) 
        #
        VT = 0.0 
        for i in range(nv):
            for j in range(i+1):
                
                factor = hbar**2 / 8.0 
                if i != j:
                    factor *= 2.0 
                    
                VT += factor * Gammatilde[i] * Gammatilde[j] * Gvib[i][j]
        terms.append(tensor.QuadratureOperator(VT, quad_transform, quad_transform))
            
        # The total Hamiltonian 
        H = tensor.DirectSumOperator(*terms)
        
        
        # Also calculate the rovib, pure rot tensor operators
        # for later use 
        #
        # Crot[a][b] is the coefficient of [iJa/hbar, iJb/hbar]_+
        #
        # Crot[a][b] = Crot[b][a]
        
        Crot = [[
            tensor.QuadratureOperator(-0.25 * hbar**2 * Grot[a][b], quad_transform, quad_transform)
            for a in range(3)]
            for b in range(3)]
        #
        # 
        # Crv[k][a] is a coefficient of (iJa/hbar)
        #
        # Note that diagonal vibrational matrix elements of these terms 
        # are zero by anti-symmetry
        #
        Crv = []
        for a in range(3):
            terms = [] 
            for k in range(nv):
                terms.append(tensor.QuadratureOperator(+0.5 * hbar**2 * Grv[k][a], dquad_transforms[k], quad_transform) )
                terms.append(tensor.QuadratureOperator(-0.5 * hbar**2 * Grv[k][a], quad_transform, dquad_transforms[k]) )
            
            Crv.append(tensor.DirectSumOperator(*terms))

        super().__init__(H.shape, H.dtype)
        
        self.H = H
        self.Crot = Crot 
        self.Crv = Crv 
        
        return 
    
    def _contract(self, network):
        return self.H._contract(network)
    
    def asConfigurationOperator(self, config_funs, labels = None):
        return self.H.asConfigurationOperator(config_funs, labels = labels) 