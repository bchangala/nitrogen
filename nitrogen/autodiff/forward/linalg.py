"""
nitrogen.autodiff.forward.linalg
--------------------------------

Linear algebra routines for forward-type automatic differentiaion

"""
import numpy as np 


import nitrogen.autodiff.forward as adf 

def inv(M):
    """
    Calculate the inverse of a square matrix M.

    Parameters
    ----------
    M : adarray
        The matrix to be inverted.

    Returns
    -------
    adarray 
        The inverse matrix
        
    Notes
    -----
    The last two axes of the base shape are considered the 
    matrix axes.

    """
    
    #
    # The derivatives of the matrix inverse are determined
    # by substituting the Taylor series of M and inv(M)
    # into the defining relation
    #
    #   M @ inv(M) = 1
    #
    # and collecting like powers of the variables
    #
    # The zeroth-order relation is just the inverse of the 
    # matrix value. Higher-order terms allow the
    # n**th derivatives of inv(M) to be determined by only
    # (n-1)**th derivatives inv(M) and the derivatives
    # of M itself.
    #
    #
    #  iM(c) = - iM(0) * (Sum_{b < c} M(c-b) @ iM(b) )
    #
    # where b and c are multi-indices
    #


    M0 = M.d[0] # The value
    
    # Create the derivative array for the result.
    # This has the same shape and type as M
    iM = np.empty(M.d.shape, dtype = M.d.dtype)
    
    
    #
    # Compute the inverse matrix 
    # and store in the value of the result array
    #
    np.copyto(iM[0], np.linalg.inv(M0)) 
    
    
    
    nd = M.nd # The number of derivatives 
    
    # Loop through the derivatives of inv(M). The 
    # multi-index table is already sorted by increasing
    # order.
    
    # Create a temp array with the base shape
    temp = np.empty(M.d.shape[1:], dtype = M.d.dtype) 
    
    #
    # Start the loop beginning with first deriatives.
    # (The value has already been computed)
    #
    for iC in range(1,nd):
        # Compute the (C) derivative
        idxC = M.idx[iC,:]    # The multi-index for (C)
        kC = np.sum(idxC)     # The order of (C)
        
        # Loop through all derivatives B less than C
        #
        temp.fill(0) # Initialize temp <--- 0 
        #
        for iB in range(nd):
            idxB = M.idx[iB,:]  # The derivative index of B
            kB = np.sum(idxB)   # The derivative degree
            
            # Check that (B) < (C)
            if kB >= kC:
                break # Stop the B loop; all remaining are >= (C)
            if np.any(idxB > idxC):
                continue 
            
            # 
            #  Accumulate - M(C-B) @ iM(B)  (Note the minus sign)
            # 
            iCmB = adf.idxpos(idxC-idxB, M.nck)
            
            temp -= M.d[iCmB] @ iM[iB]
        
        
        #
        # The (C) derivative is now iM(0) @ temp
        #
        np.matmul(iM[0], temp, out = iM[iC])
    
    
    #####################
    # Derivatives have been calculated
    # 
    # Finish up the adarray stuff
    
    if M.zlevel == -1 :
        raise ValueError("Inverse of zero")
    elif M.zlevel == 0:
        # M is constant, so is its inverse
        zlevel = 1
    else:
        zlevel = M.k  # General case
    
    # Individual zlevels
    # If any are 0, then M does not depend on that
    # variable. Neither does its inverse.
    # Otherwise, full zlevels
    zlevels = [ 0 if zl == 0 else M.k for zl in M.zlevels ]
    
    # Construct and return the adarray object
    
    return adf.array(iM, M.k, M.ni, zlevel = zlevel, zlevels = zlevels,
                     nck = M.nck, idx = M.idx)