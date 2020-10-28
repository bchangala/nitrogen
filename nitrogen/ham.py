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

from scipy.sparse.linalg import LinearOperator

def hdpdvr_bfJ(dvrs, cs, pes, masses, Jlist = 0):
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
    pes : function
        A potential energy function f(Q)
    masses : array_like
        Masses.
    Jlist : int or array_like
        Total angular momentum value(s).
    
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
    V = pes(Q)
    

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
