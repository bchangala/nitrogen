"""
nitrogen.vpt

ho_core.py

Core harmonic oscillator routines

"""

import numpy as np 

import nitrogen.autodiff.forward as adf


__all__ = ['a_ap_matrix', 'q_matrix', 'p_matrix',
           'cubic_firstorder_vacuum']


def a_ap_matrix(n):
    """
    Calculate the matrix representations 
    of the annihilation and creation
    operators, :math:`a` and :math:`a^\\dagger`.

    Parameters
    ----------
    n : int
        The number of states. The last state is
        :math:`\\vert n-1 \\rangle`.

    Returns
    -------
    a : ndarray
        The (`n`, `n`) matrix representation of 
        :math:`a`.
    ap : ndarray
        The (`n`, `n`) matrix representation of
        :math:`a^\\dagger`.

    """
    
    a = np.zeros((n,n))
    ap = np.zeros((n,n))
    
    for i in range(n-1):
        a[i,i+1] = np.sqrt(i+1)
        ap[i+1,i] = np.sqrt(i+1)
    
    
    return a,ap 
    
def q_matrix(n, m = 1):
    """
    Calculate the matrix representation 
    of :math:`q^m`, where :math:`q = \\sqrt{\\frac{1}{2}}(a + a^\\dagger)`.

    Parameters
    ----------
    n : int
        The number of states. The last state is
        :math:`\\vert n-1 \\rangle`.
    power : int, optional
        The power of :math:`q`. The default is 1. Must be 
        a non-negative integer.

    Returns
    -------
    q : ndarray
        The (`n`,`n`) matrix representation of :math:`q^m`.

    """
    
    if m < 0 :
        raise ValueError('m must be a non-negative integer')
    
    if m == 0:
        q = np.eye(n)
    else:
        # m is a positive integer
        
        ntemp = n + (m - 1) # Use a large enough temp matrix size
                            
        a,ap = a_ap_matrix(ntemp)
        qtemp = np.sqrt(0.5) * (a + ap) 
        
        q = qtemp
        for i in range(m-1):
            q = q @ qtemp
            
        q = q[:n,:n].copy()
        
    return q

def p_matrix(n, m = 1):
    """
    Calculate the matrix representation 
    of :math:`p^m`, where :math:`p = -i \\sqrt{\\frac{1}{2}}(a - a^\\dagger)`.

    Parameters
    ----------
    n : int
        The number of states. The last state is
        :math:`\\vert n-1 \\rangle`.
    power : int, optional
        The power of :math:`p`. The default is 1. Must be 
        a non-negative integer.

    Returns
    -------
    p : ndarray
        The (`n`,`n`) matrix representation of :math:`p^m`.

    """
    
    if m < 0 :
        raise ValueError('m must be a non-negative integer')
    
    if m == 0:
        p = np.eye(n)
    else:
        # m is a positive integer
        
        ntemp = n + (m - 1) # Use a large enough temp matrix size
                            
        a,ap = a_ap_matrix(ntemp)
        ptemp =  -1j * np.sqrt(0.5) * (a - ap) 
        
        p = ptemp
        for i in range(m-1):
            p = p @ ptemp
            
        p = p[:n,:n].copy()
        
    return p

def cubic_firstorder_vacuum(V,n):
    """
    Calculate the first-order coefficients 
    of the vacuum ground state from cubic
    anharmonic force constants.

    Parameters
    ----------
    V : ndarray
        The scaled derivative array.
    n : int
        The number of coordinates.

    Returns
    -------
    c : ndarray
        The first order coefficients.
        
    Notes
    -----
    The coefficient array `c` has the same
    format and sorting as derivative arrays, where
    the quantum numbers replace the deriative
    multi-index.

    """
    
    # Calculate the multi-index table up through
    # cubic terms. This is also the vib. quantum
    # table for states up to 3 quanta
    idxtab = adf.idxtab(3,n)
    nd = idxtab.shape[0] 
    
    if len(V) < nd: 
        raise ValueError("V must contain at least cubic derivatives") 
    
    nck = adf.ncktab(n+2, min(n,2)) # Create binomial coefficient table 
    
    #
    # We assume the force constants are in 
    # normal coordinate form, i.e. the gradients
    # are zero and the hessian is diagonal
    #
    # Extract harmonic frequencies 
    w = np.zeros((n,)) 
    idx = 1 + n
    for i in range(n):
       w[i] = 2 * V[idx] # (i i) derivative
       idx += (n-i)
    
    # 
    # We now consider first-order coefficients
    # from all cubic force constants
    #
    
    n2 = adf.nderiv(2,n) # The position in the table where cubics start
    n3 = nd
    
    # Initialize the list of coefficients
    c = np.zeros((nd,)) 
    
    # Loop through each cubic force constant 
    midx = np.zeros((n,), dtype = np.uint32)
    
    for idx in range(n2,n3):
        
        fidx = idxtab[idx,:] # The force constant multi-index 
        
        
        if max(fidx) == 3:
            # Case (1)
            # phi_iii type
            
            i = np.where(fidx == 3)[0][0] 
            
            # Contribution (1a)
            # | 1_i >
            #
            midx[i] = 1 # Set multi-index 
            c[adf.idxpos(midx, nck)] += -V[idx] * np.sqrt(1.125) / w[i] 
            
            # Contribution (1b)
            # | 3_i >
            #
            midx[i] = 3 
            c[adf.idxpos(midx, nck)] += -V[idx] * np.sqrt(0.75) / (3*w[i])
            
            
            midx[i] = 0 # Reset multi-index
        
        elif max(fidx) == 2:
            # Case (2)
            # phi_ijj type
            #
            
            i = np.where(fidx == 1)[0][0]
            j = np.where(fidx == 2)[0][0] 
            
            # Contribution (2a)
            # | 1_i 0_j >
            #
            midx[i] = 1
            c[adf.idxpos(midx,nck)] += -V[idx] * np.sqrt(0.125) / w[i]
            
            # Contribution (2b)
            # | 1_i 2_j >
            #
            midx[j] = 2 
            c[adf.idxpos(midx,nck)] += -V[idx] * np.sqrt(0.25) / (w[i] + 2*w[j])
            
            midx[i] = 0  # Reset multi-index
            midx[j] = 0 
            
        else: # max(fidx) == 3
            # Case (3)
            # phi_ijk type 
            
            i,j,k = np.where(fidx == 1)[0] 
            
            # Contribution (3a) 
            # | 1_i 1_j 1_k >
            midx[i] = 1
            midx[j] = 1
            midx[k] = 1 
            c[adf.idxpos(midx,nck)] += -V[idx] * np.sqrt(0.125) / (w[i] + w[j] + w[k])
            
            midx[i] = 0 
            midx[j] = 0 
            midx[k] = 0 
        
    # c now contains all first-order contributions
    
    return c 
            