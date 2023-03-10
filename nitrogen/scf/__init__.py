"""
nitrogen.scf
------------

General self-consistent field methods.

"""

import nitrogen
import nitrogen.tensor as tensor
import numpy as np 
import time


from . import rovib 
from .rovib import * 

__all__ = [] 
__all__ += rovib.__all__
__all__ += ['calcSCF', 'thermalSCF', 'calcRho', 'config_table',
            'Heff_ci_mp2', 'simple_MP2', 'Heff_aci_mp2', 'scfStability',
            'singlesConfigs', 'doublesConfigs']


def calcSCF(H, labels = None, init_wfs = None, target = None, 
            sorting = 'energy', tol = 1e-10, maxiter = 100, printlevel = 0):
    """
    Calculate the generalized self-consistent field solution
    for a TensorOperator `H`.

    Parameters
    ----------
    H : TensorOperator
        The Hamiltonian as a tensor operator in the primitive basis 
        representation.
    labels : list of lists, optional
        Each element of list is a list of labels one factor of the product
        wavefunction. The labels 0, 1, 2, ... correspond to the axes 
        of `H`. The default is [[0], [1], [2], ...]
    init_wfs : list of ndarrays, optional
        The initial wavefunction factors. If None, default values will
        be used.
    target : array_like, optional
        The SCF target configuration. The default [0, 0, ...]
    sorting : {'energy'}
        The SCF factor sorting mode. This defines the meaning of the 
        target index values. The default is 'energy'.
    tol : float, optional
        The SCF energy convergence tolerance. The default is 1e-10.
    maxiter : int, optional
        The maximum number of SCF iterations. The default is 100.
    printlevel : int, optional
        Output print level. The default is 0.

    Returns
    -------
    e_scf : float
        The final SCF energy
    wfs : list of ndarrays
        The final SCF wavefunctions
    ve : list of ndarrays
        ve[i] are the 1-mode SCF energies for factor i.
    vwfs : list of ndarrays
        vwfs[i] is the array of virtual SCF wavefunctions for factor i

    """
    
    ##################
    # Default labels: [ [0], [1], [2], ... for each axis of shape]
    if labels is None:
        labels = [[i] for i in range(len(H.shape))]
    ##################
    # Check that each label is used once
    if list(np.unique(sum(labels,[]))) != [i for i in range(len(H.shape))]:
        raise ValueError("labels must use 0, 1, 2, ... once each")
    ##################
    
    ##################
    # Default init_wfs
    if init_wfs is None:
        init_wfs = []
        for i,lab in enumerate(labels):
            wfshape = tuple([H.shape[j] for j in lab]) # The shape of this factor 
            wf = np.ones(wfshape, dtype = np.float64)  # Uniform initial wavefuncion
            init_wfs.append(wf)
    ##################
    # Continue with init_wfs processing
    # 1) Explicitly normalize each factor 
    try:
        for i in range(len(init_wfs)):
            init_wfs[i] = init_wfs[i] / np.sqrt( np.sum(abs(init_wfs[i]) ** 2))
    except:
        raise ValueError("Invalid init_wfs")
    ##################
    
    ##################
    # Parse target vector
    if target is None: # Default: [0, 0, 0  ... for each factor]
        target = np.zeros((len(labels),) ,dtype = np.uint16)
    #
    target = np.array(target) 
    if len(target) != len(labels):
        raise ValueError("target array has an unexpected size")
    #
    ##################
    
    ########################
    # Parse sorting strategy 
    if sorting == 'energy':
        pass  # OK
    else:
        raise ValueError("Unexpected sorting string")
    ########################
    
    ########################
    #
    # Begin SCF iterations
    #
    cnt = 0 
    wfs = [wf for wf in init_wfs]
    e_scf = np.inf #dummy start value
    vwfs = [None for i in range(len(init_wfs))] # The complete virtual wfs
    ve = [None for i in range(len(init_wfs))]   # The complete SCF 1-mode energies
    
    while cnt < maxiter:
        #
        # Perform an SCF sweep for each wavefunction factor
        #
        for i in range(len(wfs)):
            
            # Construct the TensorNetwork for the bra / ket,
            # which includes all factors except the i**th
            bra_tensors = [wfs[j] for j in range(len(wfs)) if j != i]
            # Construct the labels for the TensorNetwork
            # This requires mapping 0,1,2,... -> -1,-2,-3,...
            bra_labels = [[ (-l-1) for l in labels[j]] for j in range(len(labels)) if j != i]
            
            bra = tensor.TensorNetwork(bra_tensors, bra_labels)
            ket = bra # Same on each side 
            
            # Calculate the effective Hamiltonian 
            Heff = H.contract(tensor.interleaveNetworks(bra, ket))
           
            nd = len(Heff.shape) # The number of axes of Heff
            nh = np.prod(Heff.shape[::2])
            # nh = the square side length of matricized Heff
            
            # Reshape and diagonalize Heff 
            h = np.transpose(Heff, list(range(0,nd,2)) + list(range(1,nd,2)))
            h = h.reshape((nh,nh))
            w,u = np.linalg.eigh(h) 
            # Currently no sorting to perform, 'energy' is default 
            wfs[i] = (u[:,target[i]]).reshape(wfs[i].shape)
            vwfs[i] = (u.T).reshape((nh,) + wfs[i].shape)
            ve[i] = w 
        
        # After one sweep through each factor, calculate the scf energy
        #
        e = w[target[-1]]  # the current scf energy 
        err = np.abs(e - e_scf) 
        if printlevel > 0 :
            print(f"SCF iteration {cnt:d} ... E = {e:+10.6e} ... delta = {err:10.6e}")
        e_scf = e 
        cnt += 1 
        
        # Check for convergence or max 
        if err < tol: 
            if printlevel > 0:
                print("SCF convergence reached.")
            break # Exit SCF loop 
    #########################################
    #
    if cnt >= maxiter:
        print(f"Warning: SCF loop reached maxiter = {maxiter:d} iterations"
              " and exited!")
    
    return e_scf, wfs, ve, vwfs
    
def thermalSCF(H, beta, labels = None, init_density = None,
               tol = 1e-10, maxiter = 100, printlevel = 0):
    """
    Calculate the thermal self-consistent field via the
    Bogolyubov free energy variational principle for the 
    Hamiltonian TensorOperator `H`.

    Parameters
    ----------
    H : TensorOperator
        The Hamiltonian as a tensor operator in the primitive basis 
        representation.
    beta : scalar
        The value of :math:`\\beta = 1/kT`.
    labels : list of lists, optional
        Each element of list is a list of labels one factor of the product
        wavefunction. The labels 0, 1, 2, ... correspond to the axes 
        of `H`. The default is [[0], [1], [2], ...]
    init_density : list of ndarrays, optional
        The initial thermal density operator for each factor.
    tol : float, optional
        The SCF free energy convergence tolerance. The default is 1e-10.
    maxiter : int, optional
        The maximum number of SCF iterations. The default is 100.
    printlevel : int, optional
        Output print level. The default is 0.
        
    Returns
    -------
    F_scf : float
        The thermal SCF free energy.
    rhos : list of ndarrays
        The thermal SCF density operators.
    ve : list of ndarrays
        ve[i] are the 1-mode SCF energies for factor i.
    vwfs : list of ndarrays
        vwfs[i] is the array of virtual SCF wavefunctions for factor i


    """
    
    ##################
    # Default labels: [ [0], [1], [2], ... for each axis of shape]
    if labels is None:
        labels = [[i] for i in range(len(H.shape))]
    ##################
    # Check that each label is used once
    if list(np.unique(sum(labels,[]))) != [i for i in range(len(H.shape))]:
        raise ValueError("labels must use 0, 1, 2, ... once each")
    ##################
    
    ##################
    # Check beta
    if beta < 0:
        raise ValueError("beta must be non-negative")
    
    ##################
    # Default init_density
    # Use high-temperature limit --> identity
    if init_density is None:
        init_density = []
        for i,lab in enumerate(labels):
            wfshape = tuple([H.shape[j] for j in lab]) # The shape of this factor's wavefunction
            N = np.prod(wfshape) # The total primitive dimension of this factor
            rho = np.eye(N).reshape(wfshape+wfshape) # Identity operator
            init_density.append(rho)
    ##################
    # Continue with init_density processing
    # Explicitly normalize each density operator to unit trace
    #
    try:
        for i in range(len(init_density)):
            lab = [j+1 for j in range(len(labels[i]))] 
            trace = np.einsum(init_density[i], lab + lab, optimize = True) 
            init_density[i] = init_density[i] / trace 
    except:
        raise ValueError("Invalid init_density")
    #
    # Note that the rho matrices have their axes ordered as
    # rho_ijk,i'j'k'...
    # where i and i', refer to the same degree of freedom.
    #
    # This is different than the (implicit) index order of the
    # TensorOperator Hamiltonian, which is H_ii',jj',kk',...
    #
    ##################
    
    
    ########################
    #
    # Begin thermal SCF iterations
    #
    cnt = 0 
    rhos = [rho for rho in init_density]
    F_scf = np.inf #dummy start value
    vwfs = [None for i in range(len(init_density))] # The complete virtual wfs
    ve = [None for i in range(len(init_density))]   # The complete SCF 1-mode energies
    
    while cnt < maxiter:
        #
        # Perform an SCF sweep for each rho factor
        #
        for i in range(len(rhos)):
            
            # Construct the TensorNetwork for the direct product
            # thermal density operator which includes all factors except the i**th
            rho_tensors = [rhos[j] for j in range(len(rhos)) if j != i]
            # Construct the labels for the TensorNetwork
            rho_labels = [ ([-2*k - 1 for k in labels[j]] + [-2*k - 2 for k in labels[j]])
                          for j in range(len(rhos)) if j != i]
            # Construct the TensorNetwork representing the dirct product
            # of all rho factors
            rho = tensor.TensorNetwork(rho_tensors, rho_labels)
            
            # Calculate the thermally averaged effective Hamiltonian 
            Heff = H.contract(rho)
            
            
            nd = len(Heff.shape) # The number of axes of Heff
            nh = np.prod(Heff.shape[::2])
            # nh = the square side length of matricized Heff
            
            # Reshape and diagonalize Heff 
            h = np.transpose(Heff, list(range(0,nd,2)) + list(range(1,nd,2)))
            h = h.reshape((nh,nh))
            w,u = np.linalg.eigh(h) 
            
            # Construct 1-group thermal density operator
            R,_ = calcRho(w,u,beta) # R is matricized
            rhos[i] = R.reshape(rhos[i].shape)
            
            vwfs[i] = (u.T).reshape((nh,) + rhos[i].shape[:len(rhos[i].shape)//2])
            ve[i] = w 
        
        # After a full sweep through each factor, 
        # check the convergence of the free energy
        F0 = _calcF0(ve, beta)  
        Ebar = 0.0 
        for e in ve:
            ebar = _calcEthermal(e,beta) # should all be equal when converged
            Ebar += ebar
        Ebar /= len(ve) 
        #
        #
        F = F0 - (len(ve) - 1.0) * Ebar
        
        err = np.abs(F - F_scf) 
        if printlevel > 0 :
            print(f"SCF iteration {cnt:d} ... F = {F:+10.6e} ... delta = {err:10.6e}")
        F_scf = F 
        cnt += 1 
        
        # Check for convergence or max 
        if err < tol: 
            if printlevel > 0:
                print("Thermal SCF convergence reached.")
            break # Exit SCF loop 
    #########################################
    #
    if cnt >= maxiter:
        print(f"Warning: SCF loop reached maxiter = {maxiter:d} iterations"
              " and exited!")

    return F_scf, rhos, ve, vwfs

def calcRho(w,u,beta):
    """ Calculate the normalized thermal density operator,
        :math:`\\rho = \\exp[-\\beta H]`
        
    Parameters
    ----------
    w : (n,) ndarray
        The energy eigenvalues of :math:`H`.
    u : (...,n) ndarrays
        The orthonormal eigenvectors of :math:`H`.
    beta : scalar
        The value of :math:`\\beta = 1/kT`. If `beta` is
        equal to ``np.inf``, then only the lowest energy
        eigenvectors is used. Behavior is undefined for
        exactly degenerate ground state.
    
    Returns
    -------
    rho : (...,...) ndarray
        The thermal density operator, :math:`\\rho`, normalized to unit trace,
        :math:`\\mathrm{Tr}[\\rho] = 1`.
    F : scalar
        The Helmholtz free energy. 
    """
    
    # Sort eigensystem by energy first 
    I = np.argsort(w)
    w = w[I]
    u = u[...,I]
    E0 = w[0] # The lowest energy 
    
    uT = np.moveaxis(u,-1,0) # move the eigen index to front for "u^T"
    
    if np.isposinf(beta): # Positive infinity 
        #rho = np.multiply.outer(u[:,0], u[:,0]) 
        rho = np.tensordot(u[...,0:1],uT[0:1,...], axes = 1)
        F = E0
        # As u[:,0] is already normalized to unity, 
        # so is the trace of rho
    else:
        # beta is finite 
        # Calculate the Boltzmann factors relative to exp(-beta * E0)
        z = np.exp(-beta * (w - E0)) # strictly >= 1.0 
        Zbar = np.sum(z)  # The partition function relative to exp(-beta * E0)
        rho = np.tensordot(u , np.tensordot(np.diag(z),  uT, axes = 1),
                           axes = 1) / Zbar
        F = (-np.log(Zbar) / beta) + E0 # The Helmholtz free energy : -kT * log(Z)
    
    return rho, F

def _calcF0(ve,beta):
    
    F0 = 0.0
    for e in ve:
        F0 += e[0] - (_calcLogZBar(e,beta) / beta)
    return F0 

def _calcLogZBar(e, beta):
    """ e : ndarray of energies """
    
    if np.isposinf(beta):
        return 1.0 # Ignores possible degeneracy !!! 
    
    e = np.sort(e)
    e0 = e[0] 
    zbar_minus_one = 0.0
    for i in range(1,len(e)):
        zbar_minus_one += np.exp(-beta * (e[i]-e0))
    return np.log1p(zbar_minus_one) 

def _calcEthermal(e, beta):
    """ e : ndarray of energies """
    
    e = np.sort(e)
    e0 = e[0]
    
    if np.isposinf(beta):
        return e0
    
    wgt = np.exp(-beta * (e-e0))
    ebar = np.sum( e * wgt) / np.sum(wgt) 
    return ebar
    
def config_table(maxf, n, sort = True, fun = None, index_range = None, minf = None):
    """
    Tabulate a list of configurations in `n` indices 
    with fun(config) <= `maxf`. 

    Parameters
    ----------
    maxf : scalar
        The maximum excitation.
    n : int
        The number of indices.
    sort : boolean
        If True (default), the configuration table is sorted 
        by the excitation function.
    fun : function, optional
        The excitation function. If None (default), the sum
        of indices is used. 
    index_range : array_like
        The dimension of each index. The default is infinite.
    minf : scalar
        The minimum excitation. The default is negative infinity.
        Allowed configurations are *strictly greater than* `minf`.

    Returns
    -------
    ndarray


    Notes
    -----
    Custom exciations functions `fun` take an argument
    ''configs'' with is a (..., `n`) array_like containing
    an array of configuration index vectors. It must return
    a (...)-shaped array with the excitation value. 
    The excitation value must be monotonically increasing with
    an increment of one or more configuration indices. 
    
    """
    
    if fun is None:
        fun = lambda x : np.sum(x, axis = -1)  # sum over last axis
    if index_range is None:
        index_range = [np.inf for i in range(n)] # Maximum index values 
    if minf is None:
        minf = -np.inf # Negative infinity 
        
    #######################################################
    # Loop once to count how many configurations there are.
    # Then repeat and actually record them.
    #
    nconfig = 0 
    for record in [False, True]:
        
        if record:
            table = np.zeros((nconfig, n), dtype = np.int32) 
        
        config = np.zeros((n,), dtype = np.uint32) 
        nconfig = 0 
        if fun(config) > maxf: # The [0 0 0 ... ] configuration is already
        # greater than the maximum excitation. Return an empty table.
            return np.zeros((0,n))
        # Otherwise, continue
        if fun(config) > minf: # Strictly greater than minf !
            nconfig += 1 # Count [0 0 0 0 ...]
            if record:
                np.copyto(table[0,:], config)
            
        while True:
            #
            # Attempt to increment the current configuration `config`
            #
            # 
            next_config = config.copy()
            found_valid = False 
            for j in range(n):
                # Try to increment index j, starting from the left.
                next_config[j] += 1 
                if next_config[j] < index_range[j] and fun(next_config) <= maxf:
                    # This config is okay.
                    if fun(next_config) > minf: # Strictly greater than minf!
                        nconfig += 1 
                    config = next_config
                    found_valid = True 
                    break 
                else: # next_config is out-of-bounds
                    # Set this index to zero and move on to the next one
                    next_config[j] = 0 
            
            if not found_valid: # We were not able to find a new valid configuration
                break 
            elif found_valid and record and fun(next_config) > minf:
                np.copyto(table[nconfig-1, :], next_config)
    
    if sort:
        # First sort table by configuration indices
        for j in range(n-1, -1, -1):
            I = np.argsort(-table[:,j], kind = 'stable')
            table = table[I,:] # descending sort 
        # Then sort by the excitation number of each configuration
        # This will leave the final table sorted by excitation first,
        # and then by each column left-to-right
        I = np.argsort(fun(table), kind = 'stable')
        table = table[I,:] # Sort table by the excitation number 
    
    return table 
    
            
        
def Heff_ci_mp2(Hcfg, ci_max, mp2_max = None, neff = None, excitation_fun = None,
                printlevel = 1):
    """
    Calculate hybrid VCI + VMP2 effective Hamiltonian 

    Parameters
    ----------
    Hcfg : ConfigurationOperator
        The configuration representation Hamiltonian.
    ci_max : scalar
        The maximum excitation of the variational block.
    mp2_max : scalar, optional
        The maximum excitation of the perturbative block. If None (default),
        no perturbative corrections will be made.
    neff : int, optional
        The size of the effective Hamiltonian. If None (default), then the
        entire variational space will be used.
    excitation_fun : function, optional
        The configuration excitation function. If None, then the sum
        of configuration indices is used. See :func:`~nitrogen.scf.config_table`.
    printlevel : int, optional
        Printed output level. The default is 1. 
        
    Returns
    -------
    Heff : ndarray
        The effective Hamiltonian 

    """
    
    #################################
    # First, calculate the configurations that define
    # the variational and perturbative blocks
    index_range = Hcfg.shape
    n = len(index_range)
    
    cfg0 = config_table(ci_max, n, fun = excitation_fun, index_range = index_range) 
    
    if mp2_max is None:
        cfg1 = np.empty((0, n))
    else:
        cfg1 = config_table(mp2_max, n, fun = excitation_fun, index_range = index_range, 
                            minf = ci_max)
    if neff is None:
        neff = cfg0.shape[0]
    else:
        neff = min(cfg0.shape[0], neff) 
    if neff < 1:
        raise ValueError("The size of the effective Hamiltonian is zero!") 
    #
    # cfg0 contains the variational space
    # cfg1 contains the perturbative space
    print( "-------------------------------")
    print(f"Size of Block 0 = {cfg0.shape[0]:d}")
    print(f"Size of Block 1 = {cfg1.shape[0]:d}")
    print( "-------------------------------")
    ##################################
    
    ##################################
    # Calculate the variational block
    # assuming a symmetric Hamiltonian 
    if printlevel >=1 : tic = time.perf_counter(); print("Calculating H00: ", end = "")
    H00 = Hcfg.block(cfg0, ket_configs = 'symmetric') 
    if printlevel >=1 : toc = time.perf_counter(); print(f"{toc-tic:.3f} s")
    #
    # Calculate the perturbative block
    if printlevel >=1 : tic = time.perf_counter(); print("Calculating H10: ", end = "")
    H10 = Hcfg.block(cfg1, ket_configs = cfg0)
    if printlevel >=1 : toc = time.perf_counter(); print(f"{toc-tic:.3f} s")
    # and its energies
    if printlevel >=1 : tic = time.perf_counter(); print("Calculating E1: ", end = "")
    E1 = Hcfg.block(cfg1, ket_configs = 'diagonal') 
    if printlevel >=1 : toc = time.perf_counter(); print(f"{toc-tic:.3f} s")
    #
    ##################################
    
    ##################################
    # Diagonalize the variational block
    w0,u0 = np.linalg.eigh(H00) 
    # Keep only the lowest `neff` eigenvectors 
    w0 = w0[:neff]
    u0 = u0[:, :neff] 
    #
    Heff = np.diag(w0)
    if cfg1.shape[0] == 0: # there is no perturbative block to consider
        return Heff
    #
    # Transform the perturbative block
    H10p = H10 @ u0 
    # 
    # H10p[i,j] is the matrix element between the i**th perturbative
    # configuration and the j**th eigenvector of the variational block 
    #
    deltaE = np.subtract.outer(E1, w0) 
    # deltaE[i,j] equals E1[i] - w0[j]
    #
    ideltaE = 1.0 / deltaE # the energy denominator 
    
    for i in range(neff):
        for j in range(i+1):
            # Calculate the second-order contribution
            # to Heff[i,j]
            
            h2 = np.sum(-0.5 * H10p[:,i] * H10p[:,j] * (ideltaE[:,i] + ideltaE[:,j]))
            Heff[i,j] += h2
            if i != j:
                Heff[j,i] += h2 
            
            #for m in range(H10p.shape[0]):
            #    c = H10p[m,i] * H10p[m,j]
            #    Heff[i,j] += 0.5 * c * (-ideltaE[m,i] - ideltaE[m,j])
            
    
    return Heff 

def simple_MP2(Hcfg, mp2_max, target = None, excitation_fun = None, printlevel = 1):
    """
    Simple single-state second-order perturbation theory 

    Parameters
    ----------
    Hcfg : ConfigurationOperator
        The configuration representation Hamiltonian.
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
    e2 : float
        The MP2 energy 
        
    Notes
    -----
    
    The zeroth order Hamiltonian is by default the diagonal of `Hcfg`. 

    """
    
    #################################
    # First, calculate the configurations that define
    # perturbative space
    index_range = Hcfg.shape
    n = len(index_range)    
    cfg = config_table(mp2_max, n, fun = excitation_fun, index_range = index_range) 
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
    E0 = Hcfg.block(cfg_target, ket_configs = 'diagonal')[0]
    Ei = Hcfg.block(cfg_mp2, ket_configs = 'diagonal')
    # Calculate off-diagonal block
    Hi0 = Hcfg.block(cfg_mp2, ket_configs = cfg_target)[:,0] 
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
    E2 = np.sum(e2) 
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
    
    return E0 + E2 
    #
    ##################################
    
def Heff_aci_mp2(Hcfg, target_cfgs, mp2_max, tol = 1e-1, excitation_fun = None,
                 max_iter = 20, target_char_tol = 0.20):
    """
    Iterative-adaptive CI effective Hamiltonian with second-order 
    corrections.

    Parameters
    ----------
    Hcfg : ConfigurationOperator
        Hamiltonian in the SCF modal configuration representation.
    target_cfgs : array_like
        The target configurations.
    mp2_max : scalar
        The maximum excitation of the perturbative corrections.
    tol : float, optional
        The first-order mixing coefficient threshold for inclusion
        in the iterative variational block. The default is 1e-1.
    excitation_fun : function, optional
        The configuration excitation function. If None, then the sum
        of configuration indices is used. See :func:`~nitrogen.scf.config_table`.
    max_iter : int, optional
        The maximum number of iterations. The default is 20.
    target_char_tol : float, optional
        The minimum required target state character to be used
        for iterative expansion stage. The default is 0.20.

    Returns
    -------
    Heff : ndarray
        The effective Hamiltonian 

    """
    
    print("------------------")
    print("Iterative CI-MP2  ")
    print("------------------")
    
    #################################
    # First, calculate the configurations that define
    # the initial variational and perturbative blocks
    index_range = Hcfg.shape
    n = len(index_range) # The number of indices
    
    #################################
    # Format the initial target space
    # 1) target_cfgs will be a permanent record of the desired
    #    target configurations
    #
    # 2) cfg0 is a running list of the adaptive CI space 
    #    The first `ntarget` entries are always the target space
    # 
    # 3) cfg1 is a running list of the adaptive MP2 space 
    target_cfgs = np.array(target_cfgs).reshape((-1,n)) 
    ntarget = target_cfgs.shape[0] # The number of targets
    cfg0 = target_cfgs.copy()
    #
    # Calculate the initial perturbative space
    cfg1 = config_table(mp2_max, n, fun = excitation_fun, index_range = index_range)
    # Remove all target configurations from initial perturbative space
    for i in range(cfg0.shape[0]): 
        istarget = (cfg1 == cfg0[i]).all(axis=1)
        cfg1 = cfg1[~istarget] 
    #
    # cfg0 contains the variational space
    # cfg1 contains the perturbative space
    print( "-------------------------------")
    print(f"Size of initial CI space = {cfg0.shape[0]:d}")
    print(f"Size of initial MP2 space = {cfg1.shape[0]:d}")
    print(f"Mixing tolerance = {tol:E}")
    print(f"Target threshold = {target_char_tol:E}")
    print( "-------------------------------")
    ##################################
    
    #
    # Initialize the CI diagonal block and CI-MP2 off-diagonal block
    print("Initializing ...")
    tic = time.perf_counter(); print("Calculating H00: ", end = "")
    H00 = Hcfg.block(cfg0, ket_configs = 'symmetric') 
    toc = time.perf_counter(); print(f"{toc-tic:.3f} s")
    #
    # Calculate the perturbative block
    tic = time.perf_counter(); print("Calculating H10: ", end = "")
    H10 = Hcfg.block(cfg1, ket_configs = cfg0)
    toc = time.perf_counter(); print(f"{toc-tic:.3f} s")
    # and its energies
    tic = time.perf_counter(); print("Calculating E1: ", end = "")
    E1 = Hcfg.block(cfg1, ket_configs = 'diagonal') 
    toc = time.perf_counter(); print(f"{toc-tic:.3f} s")
    
    cnt = 0 # The iteration count 

    while True:
        print("------------------")
        print(f"Iteration {cnt+1:d}")
        print("")
        
        ##################################
        # At the start of each loop, we have a valid H00, H10, and E1
        #
        # 1) Diagonalize H00
        # 2) Mark all eigenvectors as ``target-like'' that
        #    have sufficient overlap with target_cfgs.
        # 3) Mark all configurations in MP2 space that have sufficient
        #    mixing coefficient with any single target-like
        #    
        #    3a) If none or max_iter reached, then exit 
        #
        # 4) Add these configurations to the CI space; compute
        #    the necessary additions to the H10 block 
        
        # 1) 
        w,U = np.linalg.eigh(H00) 
        
        # 2)
        is_target_like = np.sum(U[:ntarget,:]**2, axis=0) >= target_char_tol 
        if np.sum(is_target_like) < 1:
            raise ValueError("Error: There are no CI eigenvectors with " + 
                             "sufficient target character.")
        
        # 3)
        # Transform H10 to H10' (i.e. w.r.t. the CI eigenvectors)
        H10p =  H10 @ U
        # Compute first-order amplitudes
        Delta = np.reshape(w,(1,-1)) - np.reshape(E1, (-1,1))
        # (note that Delta is len(w) x len(E1) in shape via broadcasting)
        c1 = H10p / Delta # First-order amplitudes
        # flag any configurations with sufficient mixing to any target-like state
        add_to_ci = np.max(np.absolute(c1[:,is_target_like]),axis = 1) >= tol 
        
        if sum(add_to_ci) == 0:
            print("There are zero configurations to add. Exiting loop.")
            break
        if cnt >= max_iter:
            print("Warning: max_iter reached! Exiting loop.")
            break 
        #
        # 4) Add new configurations to CI space 
        #
        new_cfg0 = cfg1[add_to_ci,:]   # New configs to add to CI space
        old_cfg1 = cfg1[~add_to_ci,:]  # Configs to keep in MP2 space 
        print(f"{sum(add_to_ci):d} config(s) are being added to the "+
              "CI space:")
        print("-------------------------")
        print(new_cfg0)
        print("-------------------------")
        
        
        tic = time.perf_counter(); print("Calculating new blocks: ", end = "")
        # The new H0 is built up as
        # 
        #  H00_old | Ha.T
        #  ---------------
        #    Ha    | Hb
        #
        H00_old = H00.copy() 
        Ha = H10[add_to_ci,:] 
        Hb = Hcfg.block(new_cfg0, ket_configs = 'symmetric') 
        
        H00 = np.block( [[H00_old, Ha.T], [Ha, Hb]] ).copy()
        
        # The new H10 is built up as 
        # 
        #  Hc     |  Hd
        Hc = H10[~add_to_ci,:]
        Hd = Hcfg.block(old_cfg1, ket_configs = new_cfg0) 
        
        H10 = np.block( [[Hc,Hd]]).copy()
        
        # The new E1 is just a cut
        E1 = E1[~add_to_ci].copy() 
        
        toc = time.perf_counter(); print(f"{toc-tic:.3f} s")
        
        #
        # H00, H10, and E1 are now correct for the
        # new CI space
        #
        # Update cfg0 and cfg lists
        cfg0 = np.concatenate((cfg0, new_cfg0), axis = 0)
        cfg1 = old_cfg1.copy() 
        
        cnt += 1
        ###########################################
        
        
    #########################################################
    # Perform final MP2 correction to effective CI Hamiltonian
    # via diagonalize-then-perturb 
    # 
    print("-----------------------")
    print("Calculating final Heff-")
    print(f"The final CI space contains {H00.shape[0]:d} configuration(s).")
    
    w,U = np.linalg.eigh(H00)
    
    is_target_like = np.sum(U[:ntarget,:]**2, axis=0) >= target_char_tol 
    if np.sum(is_target_like) < 1:
        raise ValueError("Error: There are no CI eigenvectors with " + 
                         "sufficient target character.")
    nh = np.sum(is_target_like)
    print(f"There are {nh:d} target-like state(s).")
    print("----------------------")  
    
    w = w[is_target_like]    # Use only target-like CI eigenvectors
    U = U[:,is_target_like]  #  "   " 
    
    H10p = H10 @ U
    
    deltaE = np.subtract.outer(E1, w) 
    # deltaE[i,j] equals E1[i] - w0[j]
    #
    ideltaE = 1.0 / deltaE # the energy denominator 
    
    Heff = np.diag(w) # The zeroth order Hamiltonian is the variational block
                      # It has been diagonalized, so there is no off-diagonal
                      # zeroth- or first-order contribution.
    for i in range(len(w)):
        for j in range(i+1):
            # Calculate the second-order contribution
            # to Heff[i,j]
            
            h2 = np.sum(-0.5 * H10p[:,i] * H10p[:,j] * (ideltaE[:,i] + ideltaE[:,j]))
            Heff[i,j] += h2
            if i != j:
                Heff[j,i] += h2 
    
    return Heff  

def scfStability(Hcfg, target_cfg = None):
    """
    Calculate the stability of a mean field solution
    using the configuration representation Hamiltonian.

    Parameters
    ----------
    Hcfg : ConfigurationOperator
        Hamiltonian in the SCF modal configuration
        representation
    target_cfg : array_like
        The target configuration. If None, this is assumed
        to be [0, 0, ...]

    Returns
    -------
    w : ndarray
        The eigenvalues of the SCF stability matrix
    u : ndarray
        The eigenvectors of the SCF stability matrix
        
    """
    
    shape = Hcfg.shape # The configuration dimensions
    nf = len(shape)    # The number of factors
    
    if target_cfg is None:
        target_cfg = [0 for i in range(nf)]
    target_cfg = np.array(target_cfg) 
    
    # A list of lists for the blocks of the D2L second-order
    # sensitivity matrix
    D2L = [[None for n in range(nf)] for m in range(nf)]
    
    print("----------------------")
    print("SCF stability analysis")
    print("----------------------")
    
    # Calculate the reference energy (the Lagrange multiplier)
    #
    lam_ref = Hcfg.block(target_cfg, target_cfg)[0,0]
    #
    # 
    # Calculate the diagonal blocks of D2L
    #
    for n in range(nf):
        S_cfgs = singlesConfigs(target_cfg, n, shape[n], exclude=True)
        Lam_n = Hcfg.block(S_cfgs, ket_configs = 'diagonal')
        D2L[n][n] = np.diag(Lam_n - lam_ref)
        
    #
    #
    # Calculate the off-diagonal blocks of D2L
    for n in range(nf):
        for m in range(n):
            # n > m 
            Sn = singlesConfigs(target_cfg, n, shape[n], exclude = True)
            Sm = singlesConfigs(target_cfg, m, shape[m], exclude = True)
            D  = doublesConfigs(target_cfg, (n,m), (shape[n], shape[m]), exclude = True)
            
            print(f"Block ({n:d},{m:d}):")
            # Calculate the D/0 term
            tic = time.perf_counter(); print(" D0 term ... ", end = "")
            HD0 = Hcfg.block(D, target_cfg)
            toc = time.perf_counter(); print(f"{toc-tic:.3f} s")
            # Calculate the S/S term
            tic = time.perf_counter(); print(" SS term ... ", end = "")
            HSS = Hcfg.block(Sn, Sm)
            toc = time.perf_counter(); print(f"{toc-tic:.3f} s")
            
            D2L[n][m] = HD0.reshape((shape[n]-1, shape[m]-1)) + HSS 
            
            D2L[m][n] = D2L[n][m].copy().T 
    #
    # D2L is complete 
    rows = [np.concatenate(tuple(row), axis = 1) for row in D2L]
    D2L = np.concatenate(tuple(rows), axis = 0)
    
    # Calculate the eigenvalues of D2L
    w,u = np.linalg.eigh(D2L)
    
    print("")
    print("The lowest stability eigenvalues are:")
    for i in range(min(len(w),5)):
        print(f" w[{i:d}] = {w[i]:+.4f}")
    
    return w,u
        
def singlesConfigs(config, i, ni, exclude=False):
    """
    Calculate singly excited configurations.

    Parameters
    ----------
    config : 1-d array_like
        The reference configuration.
    i : int
        The index to excite.
    ni : int
        The dimension of index `i`.
    exclude : boolean, optional
        Exclude the reference configuration `config` from the
        singly excited list. The default is False.

    Returns
    -------
    ndarray
        The singly excited configurations for index `i`.

    """
    
    config = np.array(config) 
    if i < 0 or i >= len(config):
        raise ValueError("Invalid index `i`")
    
    if exclude:
        ncfg = ni - 1
    else: 
        ncfg = ni
        
    singles = np.tile(config, (ncfg, 1)) 
    
    if exclude:
        for k in range(ni):
            if k < config[i]:
                singles[k,i] = k 
            elif k == config[i]:
                pass  # Do not include reference configuration
            else: # k > config[i]
                singles[k-1,i] = k 
    else:
        for k in range(ni):
            singles[k,i] = k 
    
    return singles 


def doublesConfigs(config, ij, ninj, exclude=False):
    """
    Calculate doubly excited configurations.

    Parameters
    ----------
    config : 1-d array_like
        The reference configuration.
    ij : tuple
        A tuple (i,j) with the two indices to excite.
    ninj : tuple
        A tuple (ni,nj) with the dimensions of the two indices to excite.
    exclude : boolean, optional
        Exclude the reference and singly excited configurations from the
        doubly excited list. The default is False.

    Returns
    -------
    ndarray
        The doubly excited configurations.

    """    
    
    # Unpack arguments
    i,j = ij
    ni,nj = ninj

    config = np.array(config) 
    if i < 0 or i >= len(config):
        raise ValueError("Invalid index `ij`[0]")
    if j < 0 or j >= len(config):
        raise ValueError("Invalid index `ij`[1]")
    if i == j :
        raise ValueError("`ij` elements must be distinct")
    
    if exclude:
        ncfg = (ni - 1) * (nj - 1)  # strict doubles only
    else: 
        ncfg = ni * nj # ref, singles and doubles
        
    doubles = np.tile(config, (ncfg, 1)) 
    
    idx = 0 
    for k in range(ni):
        if exclude and k == config[i]:
            continue
        for l in range(nj):
            if exclude and l == config[j]:
                continue 
            doubles[idx,i] = k 
            doubles[idx,j] = l 
            idx += 1

    
    return doubles   

