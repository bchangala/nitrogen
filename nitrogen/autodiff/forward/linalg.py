"""
nitrogen.autodiff.forward.linalg
--------------------------------

Linear algebra routines for forward-type automatic differentiaion

"""
import numpy as np 
import warnings 


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
        zlevel = 0
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

def sqrtm(M, assumeh = False):
    """
    Matrix square root.
    
    Parameters
    ----------
    M : adarray
        The matrix argument.
    assumeh : boolean, optional
        Assume Hermitian, the default is False.

    Returns
    -------
    adarray 
        The square-root matrix
        
    Notes
    -----
    The last two axes of the base shape are considered the 
    matrix axes.
    
    The differentiation algorithm requires that `M` is
    diagonalizable. 
    
    Numerical imprecision may cause instability if
    the eigenvalues of `M` lie close to the branch-cut
    of the underlying matrix square-root function.

    """
    
    #
    # The derivatives of the matrix square-root are determined
    # by the defining relation
    #
    #  A @ A = M 
    # 
    # where A = sqrt[M]
    #
    # Expanding the equation by Taylor series provides an expression
    # for the derivatives of A in the form of a Sylvester equation
    #
    #   A(C) @ A(0) + A(0) @ A(C) = M(C) - sum_{0<A<C} B(C-A) @ A(B)
    #
    # A(C) is determined by diagonalizing A(0). As long as v1 != -v2, 
    # where v1 and v2 are eigenvalues of A(0). This should never occur
    # is a consistent branch choice is used for the sqrt itself.
    #
    # Numerical noise near the branch-cut may cause instabilities 
    # here. For example, the sqrt of -1 +/- 1e-15 * 1j, will be
    # +/- 1j using the standard branch-cut.
    #
    
    
    #
    # Because we need to diagonalize sqrt[M], we may as well just
    # diagonalize M and calculate its sqrt that way.
    # 
    
    if assumeh:
        w,P = np.linalg.eigh(M.d[0]) 
        iP = P.T 
    else:
        # M = P @ diag(w) @ inv(P)
        #
        w,P = np.linalg.eig(M.d[0])  
        iP = np.linalg.inv(P) 
        # the columns of P are the right eigenvectors
        # of M. 
    
    
    # Allocate the result derivative array
    A = np.empty(M.d.shape, dtype = M.d.dtype)
    
    # Calculate the sqrt value array
    # First, check whether the eigenvalues of M
    # lie close to the standard sqrt branch-cut
    # (which lies on the negative real axis)
    if np.any(np.logical_and(np.real(w) < 1e-10, 
                             np.absolute(np.imag(w)) < 1e-10)):
        warnings.warn('One of more eigenvalues of M lies within 1e-10 of the sqrt branch cut.')
    
    # Calculate the sqrt matrix by its eigen decomposition
    lam = np.sqrt(w) # The sqrt eigenvalues
    np.copyto(A[0], P @ np.diag(lam) @ iP)
    
    nd = M.nd # The number of derivatives 
    
    # Create a temp array with the base shape
    temp = np.empty(M.d.shape[1:], dtype = M.d.dtype) 
    
    # Calculate 1 / (lam[i] + lam[j]) 
    ilamij = np.empty(M.d.shape[1:], dtype = lam.dtype) 
    n = lam.shape[-1] # the square matrix rank 
    for i in range(n):
        for j in range(n):
            ilamij[...,i,j] = 1.0 / (lam[i] + lam[j])
    
    
    #
    # Start the loop beginning with first deriatives.
    # (The value has already been computed)
    #
    for iC in range(1,nd):
        # Compute the (C) derivative
        idxC = M.idx[iC,:]    # The multi-index for (C)
        kC = np.sum(idxC)     # The order of (C)
        
        # Loop through all derivatives B less than C
        # and greater than 0
        
        temp.fill(0) # Initialize temp <--- 0 
        #
        for iB in range(1,nd):
            idxB = M.idx[iB,:]  # The derivative index of B
            kB = np.sum(idxB)   # The derivative degree
            
            # Check that (B) < (C)
            if kB >= kC:
                break # Stop the B loop; all remaining are >= (C)
            if np.any(idxB > idxC):
                continue 
            
            # 
            #  Accumulate  - A(C-B) @ A(B)  (Note the minus sign)
            # 
            iCmB = adf.idxpos(idxC-idxB, M.nck)
            
            temp -= A[iCmB] @ A[iB]
        
        
        #
        # We now solve the Sylvester equation
        # for this derivative 
        Gsim = iP @ (M.d[iC] + temp) @ P 
        #
        # A(C) @ A(0) + A(0) @ A(C) = G
        #
        # Similarity transform each side via iP @ (...) @ P
        #
        # Asim @ Lam + Lam @ Asim = Gsim
        #
        # Lam is diagonal. Its entries are the eigenvalues 
        # of the sqrt matrix.
        #
        # Asim[i,j] = Gsim[i,j] / (lam[i] + lam[j])
        #
        Asim = Gsim * ilamij # Element-wise multiplication 
        np.copyto(A[iC], P @ Asim @ iP)
        
    
    
    #####################
    # Derivatives have been calculated
    # 
    # Finish up the adarray stuff
    
    if M.zlevel == -1 :
        zlevel = -1  # The sqrt of constant zero is constant zero 
    elif M.zlevel == 0:
        # M is constant, so is its sqrt 
        zlevel = 0
    else:
        zlevel = M.k  # General case
    
    # Individual zlevels
    # If any are 0, then M does not depend on that
    # variable. Neither does its sqrt.
    # Otherwise, full zlevels
    zlevels = [ 0 if zl == 0 else M.k for zl in M.zlevels ]
    
    # Construct and return the adarray object
    
    return adf.array(A, M.k, M.ni, zlevel = zlevel, zlevels = zlevels,
                     nck = M.nck, idx = M.idx)
    

def eigh_block(M,indices):
    """
    Calculate an eigen-subspace of a real, symmetric matrix.

    Parameters
    ----------
    M : adarray
        The matrix to block diagonalize.
    indices: sequence of int
        The eigenpair indices to include in the target block. The indices
        are in ascending order with respect to the eigenvalues of the
        matrix value.
        
    Returns
    -------
    Lam : adarray
        The target block of the block-diagonalized matrix.
    
    V : adarray
        The basis vectors of the target block
        
    Notes
    -----
    The gauge choice of `V` is determined by imposing a symmetry
    condition on the derivatives of V within its own sub-space,
    i.e. V(0)^T@ V(C) = V(C)^T @ V(0) for all derivatives C.

    """
    
    #
    # Diagonalize the zeroth order matrix
    #
    M0 = M.d[0] 
    n = M0.shape[-1] # The matrix dimension
    w0,U0 = np.linalg.eigh(M0)
    
    # Extract the eigenvalues and eigenvectors
    # of the target subspace
    nV = len(indices) # The size of the target subspace
    
    if nV == 0:
        raise ValueError("The subspace cannot be empty")
    if nV == n:
        raise ValueError("The subspace cannot be the entire space")
        
    lam0 = w0[...,indices]
    V0 = U0[...,:,indices]
    
    # Now extract those of the orthogonal space 
    nVp = n - nV 
    indp = [i for i in range(n) if i not in indices] # Indices of the orthogonal space
    lamp0 = w0[...,indp]
    Vp0 = U0[...,:,indp]
    
    
    # Allocate the derivative arrays for the 
    # subspace basis vector and the subspace
    # block
    nd = M.nd # The number of derivatives
    pre_shape = M.d.shape[1:-2]
    V = np.empty((nd,) + pre_shape + (n,nV), dtype = M.d.dtype)
    Lam = np.zeros((nd,) + pre_shape + (nV,nV), dtype = M.d.dtype)
    
    # Calculate the eigenvalue difference arrays
    dlamVV = np.empty(pre_shape + (nV,nV), dtype = M.d.dtype)
    for i in range(nV):
        for j in range(nV):
            dlamVV[...,i,j] = lam0[...,i] - lam0[...,j] 
     
    #
    # Calculate -1 / (lam'[i] - lam[j])
    #
    midlamVpV = np.empty(pre_shape + (nVp,nV), dtype = M.d.dtype)
    for i in range(nVp):
        for j in range(nV):
            midlamVpV[...,i,j] = -1.0/(lamp0[...,i] - lam0[...,j])
    
    # Copy the target eigenpair values to the zeroth derivative
    # result
    for i in range(nV):
        np.copyto(Lam[0,...,i,i:(i+1)], lam0[...,i:(i+1)])
    np.copyto(V[0], V0)
    

    
    # Create a temp array with the size of the sub block
    tempVV = np.empty(pre_shape + (nV,nV), dtype = M.d.dtype)
    tempMV = np.empty(pre_shape + (n, nV), dtype = M.d.dtype)
    
    
    #
    # Start the loop beginning with first deriatives.
    # (The value has already been computed)
    #
    for iC in range(1,nd):
        # Compute the (C) derivative of Lam and V 
        
        idxC = M.idx[iC,:]    # The multi-index for (C)
        kC = np.sum(idxC)     # The order of (C)
        
        # Loop through all derivatives B less than C
        # and greater than 0
        
        tempVV.fill(0) # Initialize temp <--- 0 
        tempMV.fill(0) 
        #
        for iB in range(1,nd):
            idxB = M.idx[iB,:]  # The derivative index of B
            kB = np.sum(idxB)   # The derivative degree
            
            # Check that (B) < (C)
            if kB >= kC:
                break # Stop the B loop; all remaining are >= (C)
            if np.any(idxB > idxC):
                continue 
            
            iCmB = adf.idxpos(idxC-idxB, M.nck)
            # 
            # Accumulate  -0.5 * V(C-B)^T @ V(B)  
            # 
            tempVV += (-0.5) * np.swapaxes(V[iCmB],-1,-2) @ V[iB]
        
        
            # 
            # Accumulate  M(C-B) @ V(B) - V(B) @ Lam(C-B)
            #
            tempMV += (M.d[iCmB] @ V[iB] - V[iB] @ Lam[iCmB])
            
        #
        # Finally, add M(C) @ V0 to temp MV
        tempMV += M.d[iC] @ V0
        
        
        # Calculate the derivatives
        #
        # tempVV equals the projection of V(C) onto V(0)
        #
        # Eigenblock derivative
        #
        LamC = tempVV * dlamVV # Note element-wise multiplication 
        LamC += np.swapaxes(V0,-1,-2) @ tempMV
        
        np.copyto(Lam[iC], LamC)
        
        # Orthogonal projection derivative
        #
        VpV = midlamVpV * (np.swapaxes(Vp0,-1,-2) @ tempMV) 
        # (Note element-wise multiplication from the first factor)
        
        # Compute the total V(C) derivative and 
        # copy to result 
        np.copyto(V[iC], V0 @ tempVV + Vp0 @ VpV)
        
    
    #####################
    # Derivatives have been calculated
    # 
    # Finish up the adarray stuff
    

    if M.zlevel <= 0:
        # M is constant or zero, its spectrum is constant
        zlevel = 0
    else:
        zlevel = M.k  # General case
    
    # Individual zlevels
    # If any are 0, then M does not depend on that
    # variable. Neither does its eigenspace.
    # Otherwise, full zlevels
    zlevels = [ 0 if zl == 0 else M.k for zl in M.zlevels ]
    
    # Construct and return the adarray objects
    
    Lam_adf = adf.array(Lam, M.k, M.ni, zlevel = zlevel, zlevels = zlevels,
                     nck = M.nck, idx = M.idx)
    
    V_adf = adf.array(V, M.k, M.ni, zlevel = zlevel, zlevels = zlevels,
                     nck = M.nck, idx = M.idx)
    
    return Lam_adf, V_adf 
        