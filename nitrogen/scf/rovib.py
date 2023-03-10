"""
core rovibrational Hamiltonians and operators
for tensor and configuration (VSCF/VMP2/VCI) methods
"""

import nitrogen 
import nitrogen.tensor as tensor 
import numpy as np 

__all__ = ['NonLinearTO', 
           'rovib_MP2'] 



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
    
    def getRotConfigOp(self, config_funs, labels = None):
        
        CrotCO = [[self.Crot[a][b].asConfigurationOperator(config_funs, labels = labels) for a in range(3)] for b in range(3)]
        CrvCO = [self.Crv[a].asConfigurationOperator(config_funs, labels = labels) for a in range(3)]
        
        return CrotCO, CrvCO 
    
    
def rovib_MP2(Hvib, Crot, Crv, mp2_max, target = None, excitation_fun = None, printlevel = 1):
    """
    Single-state second-order rotational effective Hamiltonian.

    Parameters
    ----------
    Hvib : ConfigurationOperator
        The configuration representation pure vibrational Hamiltonian.
    Crot : nested list of ConfigurationOperator
        The coefficients of :math:`[iJ_a/\\hbar, iJ_b/\\hbar]_+`.
    Crv : list of ConfigurationOperator
        The coefficients of :math:`[iJ_a/\\hbar]`.
    mp2_max : scalar
        The maximum excitation of the perturbative block.
    target : array_like, optional
        The target configuration. The default is [0, 0, ...] 
    excitation_fun : function, optional
        The configuration excitation function. If None, then the sum
        of configuration indices is used. See :func:`~nitrogen.scf.config_table`.
    printlevel : int, optional
        Printed output level. The default is 1. 

    Returns
    -------
    Evib : float
        The MP2 vibrational energy 
    sigma : ndarray
        The quadratic coefficients
    tau : ndarray
        The quartic coefficients
    
    Notes
    -----
    `Crot` is assumed to be a symmetric tensor, ``Crot[a][b] = Crot[b][a]``.
    The matrix elements of `Crv` are assumed to be real and anti-symmetric
    within the vibrational basis. 
    
    The rotational arrays are defined by 
    
    ..  math::
        
        H_\\text{rot} = \\frac{1}{2} \\sigma_{\\alpha \\beta} J_\\alpha J_\\beta + 
            \\frac{1}{4} \\tau_{\\alpha \\beta \\gamma \\delta} J_\\alpha J_\\beta J_\\gamma J_\\delta
    
    """
    
    #################################
    # First, calculate the configurations that define
    # perturbative space
    index_range = Hvib.shape
    n = len(index_range)    
    cfg = nitrogen.scf.config_table(mp2_max, n, fun = excitation_fun, index_range = index_range) 
    #
    # Now locate the position of the target configuration
    #
    if target is None:
        target = [0 for i in range(n)]
    target = np.array(target) 
    if len(target) != cfg.shape[1]:
        raise ValueError("Target array has unexpected length.")
    istarget = (cfg == target).all(axis=1)
    # 
    cfg_mp2 = cfg[~istarget] 
    cfg_target = cfg[istarget] 
    #
    
    # Calculate zeroth-order energies
    E0 = Hvib.block(cfg_target, ket_configs = 'diagonal')[0]
    Ei = Hvib.block(cfg_mp2, ket_configs = 'diagonal')
    
    # Calculate off-diagonal block
    Hi0 = Hvib.block(cfg_mp2, ket_configs = cfg_target)[:,0] 
    #
    if printlevel >= 1:
        print("")
        print("-----------------------")
        print( " Simple MP2 energy")
        print(f" Target = {cfg_target[0]}")
        print(f" N(MP2) = {cfg_mp2.shape[0]}")
        print("-----------------------")
        print(f" E0 + E1 = {E0:10.4f}") # Zeroth + first-order energy
    #
    c1 = Hi0 / (E0 - Ei) # First-order amplitudes
    e2 = np.abs(Hi0)**2 / (E0 - Ei) # Second-order energy corrections
    E2 = np.sum(e2)  # The total second-order 
    # 
    #
    if printlevel >= 1:
        print(f" E2      = {E2:10.4f}")
        print("-----------------------")
        print(f" Total E = {E0+E2 : 10.4f}") 
    
    if printlevel >= 1: # Print contribution report 
        idx = np.argsort(-np.abs(c1))
        print("")
        print("-------------------------------------")
        print(" MP2 report  (c1 ; e2 )           ")
        print("-------------------------------------")
        for i in idx[:10]:
            with np.printoptions(formatter={'int':'{:2d}'.format}):
                print(f"{cfg_mp2[i]} ... {c1[i]:10.2E}  ;  {e2[i]:10.2E} ")
        print("-------------------------------------")
    
    Evib = E0 + E2 
    
    # Now calculate the effective rotational Hamiltonian
    #
    sigma = np.zeros((3,3))
    tau = np.zeros((3,3,3,3))
    
    # The first-order contribution comes from Trot only 
    C1 = np.zeros((3,3))
    for a in range(3):
        for b in range(a+1):
            C1[a,b] = Crot[a][b].block(cfg_target, ket_configs = 'diagonal')[0] 
            C1[b,a] = C1[a,b] # Crot is symmetric
            
    # Now add these contributions to `sigma`
    for a in range(3):
        for b in range(3): # Note full loop range
            # 
            # Cab [iJa, iJb]_+ = -Cab*(JaJb + JbJa)
            #
            # (The factor of 2 here is from the factor of
            #  1/2 in the definition of sigma)
            sigma[a,b] -= 2 * C1[a,b]
            sigma[b,a] -= 2 * C1[a,b]

    # sigma is currently symmetric and contains just the first-order 
    # contributions. Its eigenvalues (times 1/2) are the first-order rotational constants 
    B1,_ = np.linalg.eigh(0.5 * sigma) 
    MHz = 29979.2458 
    if printlevel >= 1: # Print first-order rotational information
        print("")
        print("First-order rotational tensor (cm^-1):     ")
        for a in range(3):
            for b in range(a+1):
                print(f" {0.5*sigma[a,b]:12.3E} ", end = "")
            print("")
        print("")
        print("The first-order rotational constants are ")
        lab = ['C','B','A']
        for a in range(2,-1,-1):
            print(f" {lab[a]:s} = {B1[a]:7.3f} cm^-1 =  {MHz * B1[a]:11.3f} MHz")
        print("")
    
    ###################################################
    # Second-order rotational Hamiltonian
    # 
    # Calculate off-diagonal rot/rovib matrix elements
    C2 = np.zeros((3,3,cfg_mp2.shape[0]))
    c2 = np.zeros((3,cfg_mp2.shape[0]))
    
    
    for a in range(3):
        for b in range(a+1): # Crot is symmetric 
            C2[a,b] = Crot[a][b].block(cfg_mp2, ket_configs = cfg_target)[:,0]
            np.copyto(C2[b,a], C2[a,b]) # symmetric
        
        c2[a] = Crv[a].block(cfg_mp2, ket_configs = cfg_target)[:,0] # Note this is <v'|...|target>
        # We assume c2 changes sign for <target|...|v'>
    
    #
    # There are several types of terms from the second-order sum
    # 
    # We organize these by the total (raw) power of J
    #
    # The 0th degree terms is just the VMP2 vibrational energy 
    # contribution. This was calculated above.
    #
    # The 1st degree terms come from the rot-vib first-order corrections.
    # These are zero by the rot-vib operator being real and anti-symmetric
    #
    # We continue with 2nd, 3rd, and 4th degree terms
    #
    denom = 1.0 / (E0 - Ei) # The energy denominator 
    eps_abc = nitrogen.math.levi3() # The epsilon tensor 
    
    for a in range(3):
        for b in range(3): 
            #
            # "Coriolis contribution"
            # The 1 + 1 2nd degree term
            #
            #    c_vv'[a] * c_v'v [b] (iJa) (iJb)
            #  = -c_v'v[a] * c_v'v[b] (-1 * JaJb)   note sign change in `c`
            #  = c_v'v[a] * c_v'v[b] * JaJb
            sigma[a,b] += 2 * sum(denom * c2[a] * c2[b]) 
            
            # Vibrational contribution
            # (the 2 + 0 and 0 + 2 2nd degree terms)
            #
            #   (V_vv' Cab_v'v + Cab_vv' V_v'v) [iJa,iJb]_+
            # = -2 * V_v'v * Cab_v'v * (JaJb + JbJa)
            #
            val = -4 * sum(denom * Hi0 * C2[a,b])
            sigma[a,b] += val 
            sigma[b,a] += val 
            
            #
            # The nominal cubic terms can actually be reduced
            # to quadratic terms (the cubic parts exactly cancel
            # by antisymmetry of the rot-vib vibrational operator)
            #
            #
            for c in range(3):
                val = 2 * sum(denom * c2[c] * C2[a,b]) 
                for d in range(3):
                    # This is expressly symmetric in sigma[a,b] <> sigma [b,a]
                    sigma[a,d] += eps_abc[c,b,d] * val 
                    sigma[d,a] += eps_abc[c,b,d] * val 
                    
                    sigma[d,b] += eps_abc[c,a,d] * val 
                    sigma[b,d] += eps_abc[c,a,d] * val 
            
            #
            # Finally, the 4th degree terms. These are the centrifugal
            # quartic distortion
            #
            # (The factor of 4 below comes from the factor of 1/4 in the 
            #  definition of tau)
            #
            for c in range(3):
                for d in range(3):
                    val = 4 * sum(denom * C2[a,b] * C2[c,d]) 
                    tau[a,b,c,d] += val 
                    tau[a,b,d,c] += val 
                    tau[b,a,c,d] += val 
                    tau[b,a,d,c] += val 
    
    # 
    # The final sigma tensor is symmetric
    #
    # The final tau tensor is symmetric w.r.t permutation of the first
    # pair of indices, the second pair of indices, or the first with the
    # second pair of indices.
    #
    
    return Evib, sigma, tau 
    #
    ##################################