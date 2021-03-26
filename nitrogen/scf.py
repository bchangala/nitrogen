"""
nitrogen.scf
-------------

General self-consistent field methods.

"""


import nitrogen.tensor as tensor  
import numpy as np 


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
                Heff = H.contract(bra, ket)
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