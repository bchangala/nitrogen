"""
nitrogen.linalg.packed
-------------------------

Packed-storage matrices and routines.
The lower triangle is stored in row-major order.
(For symmetric matrices, this is equivalent to upper triangle
column-major order. For Hermitian matrices, this is
the conjugate of the upper triangle in column-major order.)

"""

import numpy as np
import warnings 

def k2IJ(k):
    """
    Calculate the full 2D index (`I`,`J`) for a given
    packed 1D index `k`. The returned index is always
    in the lower triangle.

    Parameters
    ----------
    k : int
        The packed 1D index.

    Returns
    -------
    I,J : np.uint64
        The unpacked 2D index.

    """
    
    I = np.uint64((np.sqrt(8*k+1)-1)/2)
    J = np.uint64(k - (I*(I+1))//2)
    
    return I,J

def IJ2k(I,J):
    """
    Assuming a symmetric array, calculate the 1D
    packed storage index for the (I,J) = (J,I) element
    """
    
    # Let (i,j) be the equivalent position
    # in the lower triangle
    i = max(I,J)
    j = min(I,J)
    
    k = np.uint64( (i*(i+1))//2 + j)
    
    return k

def n2N(n):
    """
    Calculate the square matrix rank N
    for a packed storage size n

    Parameters
    ----------
    n : int
        The packed length.

    Returns
    -------
    N : np.uint64
        The matrix rank.

    """
    
    N = np.uint64((np.sqrt(8*n+1)-1)/2)
    
    return N

def symfull(P):
    """
    Return the full array for a symmetric array 
    in packed storage

    Parameters
    ----------
    P : ndarray
        The lower triangle of a symmetric array in packed storage.

    Returns
    -------
    ndarray
        The full symmetric matrix

    """
    
    n = P.shape[0]
    base = P.shape[1:]
    N = n2N(n)
    
    full = np.ndarray((N,N)+base, dtype = P.dtype)
    full.fill(0)
    
    k = 0
    for i in range(N):
        for j in range(i+1):
            full[i,j] += P[k]
            if i != j: 
                full[j,i] += P[k]
            k += 1
    
    return full 
    
def trilfull(L):
    """
    Return the full array for a lower triangle array.
    in packed storage

    Parameters
    ----------
    L : ndarray
        The lower triangle in packed storage

    Returns
    -------
    ndarray
        The full lower triangle array.

    """
    
    n = L.shape[0]
    base = L.shape[1:]
    N = n2N(n)
    
    full = np.ndarray((N,N)+base, dtype = L.dtype)
    full.fill(0)
    
    k = 0
    for i in range(N):
        for j in range(i+1):
            full[i,j] += L[k]  # lower triangle element
            
            k += 1
    
    return full

def inv_sp(A, out = None):
    """
    Inverse of a real symmetric, positive definite matrix in packed format.
    (Complex symmetric matrices may not be defined.)

    Parameters
    ----------
    A : (np, ...) ndarray
        The packed storage array.
    out : (np, ...) ndarray
        Output buffer. If None, this will be created. 
        If out = A, then in-place inversion is performed


    Returns
    -------
    out : ndarray
        The result. 

    """
    if A.ndim < 1:
        raise ValueError('A must be at least one-dimensional')
    
    if out is None:
        out = A.copy() # out is ready for in-place routines
    elif out is A:
        pass # out is already ok for in-place
    else: 
        np.copyto(out, A) 
        
    chol_sp(out, out = out)  # out <-- Cholesky decomposition
    inv_tp(out, out = out)   # out <-- inverse of Cholesky 
    ltl_tp(out, out = out)   # out <-- inverse of A

    return out     

def chol_sp(A, out = None):
    """
    Cholesky decomposition of a symmetric matrix in packed format.
    If real symmetric, then A should be positive definite.
    If complex symmetric (*not* Hermitian), then H should have
    non-zero pivots.

    Parameters
    ----------
    A : (np, ...) ndarray
        A is stored in 1D packed format (see :mod:`nitrogen.linalg.packed`)
    out : (np, ...) ndarray
        Output buffer. If None, this will be created. 
        If out = H, then in-place decomposition is performed

    Returns
    -------
    out : ndarray
        The lower triangle Cholesky decomposition L in packed storage.
        H = L @ L.T
    
    """
    
    if A.ndim < 1:
        raise ValueError('A must be at least one-dimensional')
    
    if out is None:
        out = A.copy() # out is ready for in-place routine
    elif out is A:
        pass # out is already ok for in-place
    else: 
        np.copyto(out, A) 
    
    # Perform in-place routine 
    _chol_sp_unblocked(out)
    
    return out
    
def _chol_sp_unblocked(H):
    """
    An unblocked, in-place implementation of Cholesky
    decomposition for symmetric matrices H.

    Parameters
    ----------
    H : (np, ...) ndarray
        H is a symmetric matrix in 1D packed storage.
        (Lower triangle row-packed)

    Returns
    -------
    ndarray
        The in-place result.
        
    Notes
    -----
    This routine uses a standard Cholesky algorithm
    for *real* symmetric matrices. It can be 
    analytically continued to complex symmetric matrices
    in some cases, but this behavior is not necessarily
    stable. Use caution!

    """

    n = H.shape[0] # the packed size
    N = n2N(n)     # The full matrix rank

    L = np.ndarray((N,N), dtype = np.ndarray)
    # Copy references to packed array elements to the lower 
    # triangle of a full "reference" matrix
    # References above the diagonal are undefined.
    k = 0
    for i in range(N):
        for j in range(i+1):
            L[i,j] = H[k:(k+1)] # Leave a singleton leading dimension 
            k += 1
    
    # Compute the Cholesky decomposition
    # with a simple unblocked algorithm
    tol = 1e-10
    pivmax = np.abs( np.sqrt(L[0,0]) ) # for pivot threshold checking
    
    for j in range(N):
        r = L[j,:j]         # The j^th row, left of the diagonal
    
        # L[j,j] <-- sqrt(d - r @ r.T)
        np.sqrt( L[j,j] - r @ r.T,  out = L[j,j]) # overwrite
        
        # Check that the pivot is sufficiently large
        if (np.abs(L[j,j]) / pivmax < tol).any() :
            warnings.warn("Small diagonal (less than rel. tol. = {:.4E} encountered in Cholesky decomposition".format(tol))
        
        # Store the new maximum pivot
        pivmax = np.maximum(pivmax, np.abs(L[j,j]))
        
        # Calculate the column below the diagonal element j
        #B = L[j+1:,:j]      # The block between r and c
        #c = L[j+1:,j]       # The j^th column, below the diagonal
        for i in range(j+1,N):
            Bi = L[i,:j]    # An 
            ci = L[i, j]    # An 
            #L[j+1:,j] = (c - B @ r.T) / L[j,j]
            np.divide( (ci - Bi @ r.T), L[j,j], out = L[i,j])
        

    return H

def inv_tp(L, out = None):
    """
    Invert a triangular matrix in lower row-packed (or upper column-packed)
    storage.

    Parameters
    ----------
    L : (np, ...) ndarray
        L is stored in 1D packed format (see :mod:`nitrogen.linalg.packed`)
    out : (np, ...) ndarray
        Output buffer. If None, this will be created. 
        If out = L, then in-place inversion is performed

    Returns
    -------
    out : ndarray
        The inverse of the triangular matrix in packed storage.
    
    """
    
    if L.ndim < 1:
        raise ValueError('L must be at least one-dimensional')
    
    if out is None:
        out = L.copy() # out is ready for in-place routine
    elif out is L:
        pass # out is already ok for in-place
    else: 
        np.copyto(out, L)  
    
    # Now perform in-place on `out`
    _inv_tp_unblocked(out)
    
    return out
    
def _inv_tp_unblocked(L):
    """
    Invert a lower triangular matrix in row-packed storage.

    Parameters
    ----------
    L : (np, ...) ndarray
        L is a triangular matrix in 1D packed storage.
        (Row-packed for lower, column-packed for upper)

    Returns
    -------
    ndarray
        The in-place result.

    """
    
    n = L.shape[0] # the packed size
    N = n2N(n)     # the full matrix rank
    one = np.uint64(1)
    
    X = np.ndarray((N,N), dtype = np.ndarray)
    # Copy references to packed element arrays to the lower 
    # triangle of a full "reference" matrix
    # Elements above the diagonal are not defined!
    k = 0
    for i in range(N):
        for j in range(i+1):
            X[i,j] = L[k:(k+1)] # Leave a singleton leading dimension
            k += 1
    
    # Compute the triangular inverse
    # with a simple in-place element by element algorithm
    abstol = 1e-15 
    # In-place lower triangle inversion
    for j in range(N - one, -1,-1):
        
        # Compute j^th diagonal element
        if (np.abs(X[j,j]) < abstol).any():
            warnings.warn(f"Small diagonal (less than abs. tol. = {abstol:.4E})" \
                          "encounted in triangle inversion")
        
        #X[j,j] = 1.0/X[j,j]
        np.copyto(X[j,j], 1.0/ X[j,j])
        
        # Compute j^th column, below diagonal
        for i in range(N - one, j, -1):
            np.multiply( -X[j,j], X[i, j+1:i+1] @ X[j+1:i+1, j], out = X[i,j])
      
    return L

def llt_tp(L, out = None):
    """
    L @ L.T of a lower triangular matrix.

    Parameters
    ----------
    L : (np, ...) ndarray 
        Lower triangular matrix in packed storage.
    out : (np, ...) ndarray
        Output buffer. If None, this will be created. 
        If out = L, then in-place multiplication is performed

    Returns
    -------
    out : ndarray
        The symmetric result in packed storage.
    
    """
    
    if L.ndim < 1:
        raise ValueError('L must be at least one-dimensional')
    
    if out is None:
        out = L.copy() # out is ready for in-place routine
    elif out is L:
        pass # out is already ok for in-place
    else: 
        np.copyto(out, L)  
        
    # Perform in-place L @ L.T
    _llt_tp_unblocked(out)
    
    return out

def _llt_tp_unblocked(L):
    """
    An unblocked, in-place routine for multiplying
    L @ L.T where L is a lower triangular matrix
    in packed row-order storage.
    
    This is equivalent to U.T @ U where U is in
    upper triangular packed column-order storage.
    
    The resulting symmetric matrix is returned in
    packed storage.

    Parameters
    ----------
    L : (np, ...) ndarray 
        A lower triangular matrix in 1D packed 
        row-order storage.

    Returns
    -------
    ndarray 
        The in-place result.
        

    """

    # Calculate matrix dimensions
    n = L.shape[0] # the packed size
    N = n2N(n)     # the full matrix rank
    one = np.uint64(1)

    A = np.ndarray((N,N), dtype = np.ndarray)
    # Copy references to adarrays to the lower 
    # triangle of a full "reference" matrix
    # References above the diagonal are undefined.
    k = 0
    for i in range(N):
        for j in range(i+1):
            A[i,j] = L[k:(k+1)] # leave a singleton leading dimension
            k += 1
    
    # This is similar to a "reverse Cholesky decomposition"
    # so we will work in the opposite direction as that
    
    for j in range(N-one, -1, -1):
        
        r = A[j,:j]         # The j^th row, left of the diagonal
        
        for i in range(N-one, j, -1):
            Bi = A[i,:j]    # An ndarray of ndarray
            ci = A[i, j]    # An            ndarray
            
            # ci <-- Ljj * ci + Bi @ r.T
            np.add(A[j,j] * ci, Bi @ r.T, out = A[i,j])
        
        np.add(A[j,j]**2, r @ r.T, out = A[j,j])

    return L

def ltl_tp(L, out = None):
    """
    L.T @ L with a lower triangular matrix.

    Parameters
    ----------
    L : (np, ...) ndarray
        Lower triangular matrix in packed storage.
    out : (np, ...) ndarray
        Output buffer. If None, this will be created. 
        If out = L, then in-place multiplication is performed

    Returns
    -------
    out : ndarray
        The symmetric result in packed storage.
    
    """
    
    if L.ndim < 1:
        raise ValueError('L must be at least one-dimensional')
    
    if out is None:
        out = L.copy() # out is ready for in-place routine
    elif out is L:
        pass # out is already ok for in-place
    else: 
        np.copyto(out, L) 
        
    # Perform in-place L @ L.T
    _ltl_tp_unblocked(out)
    
    return out

def _ltl_tp_unblocked(L):
    """
    An unblocked, in-place routine for multiplying
    L.T @ L where L is a lower triangular matrix
    in packed row-order storage.
    
    This is equivalent to U @ U.T where U is in
    upper triangular packed column-order storage.
    
    The resulting symmetric matrix is returned in
    packed storage.

    Parameters
    ----------
    L : ndarray
        A lower triangular matrix in 1D packed 
        row-order storage.

    Returns
    -------
    ndarray
        The in-place result.
        

    """

    # Calculate matrix dimensions
    n = L.shape[0] # the packed size
    N = n2N(n)     # the full matrix rank
    

    A = np.ndarray((N,N), dtype = np.ndarray)
    # Copy references to adarrays to the lower 
    # triangle of a full "reference" matrix
    # References above the diagonal are undefined.
    k = 0
    for i in range(N):
        for j in range(i+1):
            A[i,j] = L[k:(k+1)] # retain leading singleton dimension
            k += 1
    
    # This is the "converse" of the llt routine
    # for L @ L.T
    for i in range(N):
        for j in range(i+1):
            # Compute A[i,j]
            # This is the dot product of the
            # i^th row of L.T and the
            # j^th column of L
            # 
            # The i^th row of L.T is zero until
            # its i^th element
            # 
            # The j^th column of L is zero until
            # its j^th element
            #
            # So the dot product need only begin
            # at the max(i,j)^th element
            # 
            # By the loop ranges, j is always <= i
            # so max(i,j) = i, and we can begin
            # the dot product with the i^th element
            
            # The first factor is the
            # i^th row of L.T beginning at its i^th element
            # This is the transpose of the i^th column of 
            # L beginning at its i^th element, which is in
            # the lower triangle, so A's reference is OK
            F1 = (A[i:,i]).T
            # The second factor is the j^th column
            # of L beginning at its i^th element, which is
            # also in the lower triangle, so OK
            F2 = A[i:,j]
            
            np.copyto(A[i,j], F1 @ F2)

    return L

def trAB_sp(A, B, out = None):
    """
    The trace of A @ B, in symmetric packed stroage.

    Parameters
    ----------
    A,B : (np, ...) ndarray
        Symmetric matrix in packed storage.
    out : (...) ndarray, optional
        Output buffer. If None, this will be created.
        If scalar, this is ignored.

    Returns
    -------
    out : ndarray or scalar
        The result.

    """
    
    if A.shape != B.shape:
        raise ValueError("A and B must be the same shape")
    if A.ndim < 1:
        raise ValueError("A and B must have at least one dimension.")
    
    # Calculate matrix dimensions
    n = A.shape[0] # the packed size
    N = n2N(n)     # the full matrix rank
    
    if out is None:
        if A.ndim == 1: # Scalar result
            out = 0
        else: # A.ndim > 1 
            out = np.zeros(A.shape[1:], dtype = A.dtype) 
    else: # out is a buffer
        if A.ndim == 1: 
            raise ValueError("output cannot be buffered for scalar result.")
        else:
            out.fill(0.0)
    
    # Loop over lower triangle only 
    k = 0 
    for i in range(N):
        for j in range(i+1):
            if i == j: # Diagonal element
                out += A[k] * B[k] 
            else: # Off-diagonal, counts twice 
                out += 2.0 * (A[k] * B[k])
            k += 1 
            
    
    return out 
    