import numpy as np
from scipy.sparse.linalg import LinearOperator

__all__ = ['eigstrp']

def eigstrp(H, k = 5, pad = 10, tol = 1e-10, maxiter = None, v0 = None,
            rper = 20, P = None, pper = 1):
    """
    Thick-restart Lanczos iterative eigensolver with projection.

    Parameters
    ----------
    H : LinearOperator
        A real symmetric or complex Hermitian operator.
    k : int, optional
        The number of eigenvalues requested. The default is 5.
    pad : int, optional
        The total number of Lanczos vectors kept after a restart is
        `k` + `pad`. If `pad` is too small, the convergence will be
        less than optimal. The default is 10.
    tol : float, optional
        The absolute eigenvalue convergence tolerance. The default is 1e-10.
    maxiter : int, optional
        The maximum number of iterations. If None, no maximum is set.
        The default is None.
    v0 : ndarray, optional
        Initial vector. The default is random (None).
    rper : int, optional
        The restart period. The default is 20.
    P : LinearOperator, optional
        Projection operator applied to the Lanczos vectors. The default is None.
    pper : int, optional
        The projection period. The default is 1.

    Returns
    -------
    w : ndarray
        Array of `k` eigenvalues
    v : ndarray
        Array of `k` eigenvectors. ``v[:,i]`` is the eigenvector corresponding
        to eigenvalue ``w[i]``.
    
    Notes
    -----
    The restarted Lanczos method is based on that described by Wu 
    and Simon [1]_.
    
    References
    ----------
    .. [1] K. Wu and H. Simon, "Thick-Restart Lanczos Method for Large Symmetric Eigenvalue Problems."
       SIAM Journal on Matrix Analysis and Applications, 22(2), 602-616. (2000)
       https://doi.org/10.1137/S0895479898334605
    """
    
    if type(H) is not LinearOperator:
        raise TypeError("H must be a LinearOperator")
        
    n,m = H.shape
    if n != m:
        raise ValueError("H must be a square matrix")
    if k < 1:
        raise ValueError("k must be > 0")
    if pad < 0:
        raise ValueError("pad must be >= 0")
    if rper < 1:
        raise ValueError("rper must be > 0")
    if k + pad + rper > n:
        raise ValueError("k + pad + rper must be <= n")
    if tol < 0:
        raise ValueError("tol must be >= 0")
            
    # Determine dtype of Lanczos vectors
    if P is None:
        vtype = H.dtype
    else:
        raise NotImplementedError("Projection not yet supported.")
        vtype = np.result_type(H.dtype, P.dtype)
    
    # Initialize starting vector v0
    if v0 is None:
        v0 = np.random.rand(n).astype(vtype)
    else:
        if v0.shape != (n,) and v0.shape != (n,1):
            raise ValueError("v0 has the wrong shape")
        if v0.dtype != vtype:
            print("Supplied v0 is being recast")
            v0 = v0.astype(vtype)
    
    # Project and normalize v0
    if P is not None:
        v0 = P.matvec(v0)
    v0 /= np.linalg.norm(v0)
    
    # Allocate the block of Lanczos vectors
    nL = k + pad + rper
    L = np.ndarray((n, nL), dtype = vtype, order = 'F')
    Lnext = v0.copy()
    
    # Allocate the sub-space Hamiltonian
    T = np.zeros((nL,nL), dtype = np.float64) # T will always be Real.
    
    # Perform the initial Lanczos sweep

    for i in range(k+pad):
        
        # Copy the i^th Lanczos vector into the Lanczos block
        np.copyto(L[:,i], Lnext)
        w = H.matvec(L[:,i])
        alpha = np.real(np.vdot(w, L[:,i]))  # Force real (should be anyway)
        T[i,i] = alpha
        if i > 0:
            beta = np.vdot(w, L[:,i-1])
            T[i,i-1] = beta
            T[i-1,i] = beta
        
        # Calculate the next Lanczos vector
        # Explicitly normalize first to help with finite precision errors
        w /= np.linalg.norm(w)
        for j in range(i+1):
            c = np.vdot(w, L[:,j])
            w -= c * L[:,j] # Sequential Gram-Schmidt orthogonalization
        # Renormalize and store in Lnext
        np.copyto(Lnext, w/np.linalg.norm(w))
    
    # T is filled to the first (k+pad, k+pad) block
    # L is filled to the first k+pad columns
    # Lnext is the next Lanczos vector
    
    # Begin a restart
    itercnt = 0
    restart_cnt = 0
    evals = None
    
    while True:
        
        # Calculate `rper` more Lanczos vectors
        for i in range(k+pad, nL):
            np.copyto(L[:,i], Lnext) # Copy the i^th Lanczos vector into the block
            w = H.matvec(L[:,i])     # Calculate the next vector
            
            alpha = np.real(np.vdot(w, L[:,i]))  # Force real (should be anyway)
            T[i,i] = alpha
            
            if restart_cnt > 0 and i == (k+pad):
                # If this is the first loop in the restart block,
                # then we need to calculate the "arrowhead"
                for j in range(i):
                    beta = np.vdot(w, L[:,j])
                    T[i,j] = beta
                    T[j,i] = beta
            else:
                # This is a normal Lanczos step
                beta = np.vdot(w, L[:,i-1])
                T[i,i-1] = beta
                T[i-1,i] = beta
            
            # Calculate the next Lanczos vector
            w /= np.linalg.norm(w) # Normalize first
            for j in range(i+1):
                c = np.vdot(w, L[:,j])
                w -= c * L[:,j] # Sequential Gram-Schmidt orthogonalization
            w /= np.linalg.norm(w) # Normalize first
            for j in range(i+1):
                c = np.vdot(w, L[:,j])
                w -= c * L[:,j] # Sequential Gram-Schmidt orthogonalization
            # Renormalize and store in Lnext
            np.copyto(Lnext, w/np.linalg.norm(w))
            
            itercnt += 1 # Increment the iteration count (counting H matvec calls)
    
        # The restart is over and we have filled L, T, Lnext as before
        
        # Diagonalize the reduced Lanczos Hamiltonian T
        if restart_cnt > 0: 
            evals_old = evals.copy() # Save previous Ritz values
        evals,evecs = np.linalg.eigh(T)
        
        # Store the first k+pad Ritz vectors
        np.copyto(L[:, 0:(k+pad)], np.matmul(L,evecs[:,0:(k+pad)]))
    
        # Refill T
        T.fill(0) 
        for i in range(k+pad):
            T[i,i] = evals[i]
 
        # Check maxiter
        if maxiter is not None:
            if itercnt > maxiter:
                print("Maximum iterations reached!")
                break
 
        # Finally, perform a convergence check on the first `k` Ritz values
        if restart_cnt > 0:
            if ( np.abs(evals[0:k] - evals_old[0:k]) > tol ).any():
                # At least one eigenvalue is above convergence
                # tolerance, we will continue
                pass
            else:
                # All eigenvalues have converged, exit the restart loop
                print("Convergence reached in {:d} restarts ({:d} iterations).".format(restart_cnt, itercnt))
                break
        
        restart_cnt += 1
    
    # We have reached convergence. Return the first `k` Ritz values and vectors
    w = evals[0:k].copy()
    v = L[:, 0:k].copy()
    
    return w,v



