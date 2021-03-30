"""
nitrogen.scf
-------------

General self-consistent field methods.

"""


import nitrogen.tensor as tensor  
import numpy as np 
import time



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
            vwfs[i] is the array of virtual SCF wavefunction for factor i

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
                init_wfs[i] = init_wfs[i] / np.sqrt( np.sum(init_wfs[i] ** 2))
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
        e_scf = 1e99 #dummy start value
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
                nh = 1
                nd = len(Heff.shape) # The number of axes of Heff
                for j in range(0,len(Heff.shape),2):
                    nh *= Heff.shape[j]
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
    
            
        
def Heff_ci_mp2(Hcfg, ci_max, mp2_max = None, neff = None, fun = None,
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
    fun : function, optional
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
    
    cfg0 = config_table(ci_max, n, fun = fun, index_range = index_range) 
    
    if mp2_max is None:
        cfg1 = np.empty((0, n))
    else:
        cfg1 = config_table(mp2_max, n, fun = fun, index_range = index_range, 
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
