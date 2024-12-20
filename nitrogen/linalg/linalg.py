import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import aslinearoperator
import scipy.fft 
import scipy.linalg

__all__ = ['eigstrp','aslinearoperator','bounds',
           'chebauto', 'chebspec', 'chebwindow',
           'full', 'msqrth', 'simple_corr',
           'RealOrthogonalProjector']

def eigstrp(H, k = 5, pad = 10, tol = 1e-10, maxiter = None, v0 = None,
            rper = 20, P = None, pper = 1, printlevel = 0, eigval = 'smallest',
            projection_level = 2, refine_per = 20):
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
    printlevel : int, optional
        Print level. The default is 0.
    eigval : {'smallest', 'largest'}, optional
        Calculat the smallest or largest eigenvalues. The default is 'smallest'.
    projection_level : integer, optional
        The projection level determines how aggressively the projection operator
        is applied at each iteration.
        If 2 (default), it will be applied immediately
        after the matrix-vector routine and again after orthogonalization.
        If 1, it will be applied only after orthogonalization.
        If 0, it will not be applied.
    refine_per : integer, optional
        After `refine_per` restarts, the target Ritz vectors will 
        be explicitly rediagonalized and reorthogonalized. The default is 20. 
        

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
    and Simon [WS00]_.
    
    References
    ----------
    .. [WS00] K. Wu and H. Simon, "Thick-Restart Lanczos Method for Large Symmetric Eigenvalue Problems."
       SIAM Journal on Matrix Analysis and Applications, 22(2), 602-616. (2000)
       https://doi.org/10.1137/S0895479898334605
    """
    
    if not isinstance(H,LinearOperator):
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
        project = False
    else:
        vtype = np.result_type(H.dtype, P.dtype)
        project = True 
        
    plevel = projection_level 
    
    if printlevel >= 1:
        print("--------------------------------")
        print(" Starting thick-restart Lanczos ")
        print("--------------------------------")
    
    #################################################
    #
    # Initialize starting vector v0
    if v0 is None:
        v0 = np.random.rand(n).astype(vtype)
    else:
        if v0.shape != (n,) and v0.shape != (n,1):
            raise ValueError("v0 has the wrong shape")
        if v0.dtype != vtype:
            print("Supplied v0 is being recast")
            v0 = v0.astype(vtype)
    #
    # Project and normalize v0
    if project:
        v0 = P.matvec(v0)
    v0 /= np.linalg.norm(v0)
    #
    #
    #################################################
    
    #################################################
    # Allocate the block of Lanczos vectors
    #
    #     _______nL_______
    #    /                \
    #     [target]   [restart]
    #     --------   -------
    #      k    pad    rper
    #   +------+----+-------+
    #   |      |    |       |
    #   |      |    |       |
    # n |      |    |       |
    #   |      |    |       |
    #   |      |    |       |
    #   +------+----+-------+
    #
    nL = k + pad + rper
    L = np.ndarray((n, nL), dtype = vtype, order = 'F')
    Lnext = v0.copy()
    #
    # Allocate the sub-space Hamiltonian
    T = np.zeros((nL,nL), dtype = np.float64) # T will always be Real.
    #
    #################################################
    
    #################################################
    # Perform the initial Lanczos sweep
    # to populate the target space [k + pad]
    #
    for i in range(k+pad):
        
        # Copy the i^th Lanczos vector into the Lanczos block, L
        # 
        # For projection, Lnext is already projected
        
        np.copyto(L[:,i], Lnext)
        w = H.matvec(L[:,i])
        
        if project and i % pper == 0 and plevel >= 2:
            w = P.matvec(w)
            if printlevel >= 2:
                print("Projecting w")
        
        alpha = np.real(np.vdot(w, L[:,i]))  # Force real (should be anyway)
        T[i,i] = alpha
        if i > 0:
            beta = np.real( np.vdot(w, L[:,i-1]) ) # Force real (should be anyway)
            T[i,i-1] = beta
            T[i-1,i] = beta
        
        if printlevel >= 2:
            print(f"Hvec ... alpha = {alpha:.3e}")
        
        # Calculate the next Lanczos vector
        # ---------------------------------
        # Explicitly normalize first to help with finite precision errors
        w /= np.linalg.norm(w)
        for j in range(i+1):
            c = np.vdot(L[:,j], w)
            w -= c * L[:,j] # Sequential Gram-Schmidt orthogonalization
            
        # Project 
        if project and i % pper == 0 and plevel >= 1:
            w = P.matvec(w)
            if printlevel >= 2:
                print("Projecting w")
        
        # Renormalize and store in Lnext 
        np.copyto(Lnext, w/np.linalg.norm(w))
    
    # T is filled to the first (k+pad, k+pad) block
    # L is filled to the first k+pad columns
    # Lnext is the next Lanczos vector
    #
    ##################################################
    
    itercnt = 0       # The number of Lanczos iterations total
    restart_cnt = 0   # The number of restarts performed
    evals = None
    
    while True:
        #########################################
        # Begin a restart 
        # 
        # L contains the first k+pad Lanczos vectors 
        # T is filled to the  (k+pad, k+pad) block 
        # Lnext is the next Lanczos vector to add to the space.
        #
        ##########################################
        #
        # Check for a Ritz refinement
        if restart_cnt > 0 and (restart_cnt % refine_per == 0):
            #
            # Refine the target Ritz vectors 
            #
            if printlevel >= 1 :
                print("Refining target Ritz vectors")
                
            # 1) Reorthogonalize/project
            # 2) Rediagonalize 
            #
            # 1) Reorthogonalize 
            #
            maxc = 0.0 
            for i in range(k+pad):
                #
                for j in range(i):
                    c = np.vdot(L[:,i], L[:,j]) #
                    maxc = max(maxc, abs(c))
                    L[:,i] -= c * L[:,j]   # GS orthogonalize 
                    
                if project and plevel >= 1:
                    v = P.matvec(L[:,i]) # Reproject 
                    np.copyto(L[:,i], v) 
                    
                L[:,i] /= np.linalg.norm(L[:,i])  # Renormalize 
            #
            if printlevel >= 1: 
                print(f" (largest overlap = {maxc:.3E})")
            # 2) Rediagonalize 
            T.fill(0.0) 
            for i in range(k+pad):
                w = H.matvec(L[:,i])
                for j in range(i+1):
                    Hij = np.vdot(L[:,j], w)
                    T[i,j] = np.real(Hij)  # should always be real
                    T[j,i] = T[i,j]
                    
            #Lnext = np.random.rand(len(Lnext))
            
            wT,UT = np.linalg.eigh(T[:(k+pad), :(k+pad)])
            
            T.fill(0)
            for i in range(k+pad):
                T[i,i] = wT[i] # Fill diagonal 
            np.copyto(L[:, 0:(k+pad)], np.matmul(L[:,0:(k+pad)], UT))
            
            # # Reorthogonalize Lnext 
            # if project and plevel >= 1: 
            #     Lnext = P.matvec(Lnext) 
            # for i in range(k+pad):
            #     c = np.vdot(L[:,i], Lnext) 
            #     Lnext -= c * L[:,i] 
            # Lnext /= np.linalg.norm(Lnext) 
            
            #
            #
            # L now contains the refined k+pad Lanczos vectors 
            # T is filled to the [k+pad, k+pad] block
            # Lnext contains the next vector
            # 
            # END REFINEMENT
        ################################################
        #
        #
        #
        # We need to calculate `rper` more Lanczos vectors and 
        # fill the corresponding blocks in T.
        #
        for i in range(k+pad, nL):
            np.copyto(L[:,i], Lnext) # Copy the i^th Lanczos vector into the block
            w = H.matvec(L[:,i])     # Calculate the next vector
            
            if project and i % pper == 0 and plevel >= 2:
                w = P.matvec(w)
                if printlevel >= 2:
                    print("Projecting w")
                    
            alpha = np.real(np.vdot(w, L[:,i]))  # Force real (should be anyway)
            T[i,i] = alpha
            
            if printlevel >= 2:
                print(f"Hvec ... alpha = {alpha:.3e}")
                
            #####################################
            # For a standard restarted Lanczos,
            # the T matrix will have an "arrowhead"
            # appearance. The block between the 
            # target vectors and the restart vectors
            # will be otherwise zero.
            #
            # I have found that loss of orthogonality
            # amongst the Ritz vectors can creep in, 
            # and it may be necessary to explicitly
            # reorthogonalize the target space, which
            # will spoil the Lanczos recursion and lead to 
            # non-zero matrix elements between the 
            # restart and target blocks. 
            # 
            # Since we are explicitly reorthogonalizing
            # anyway, these matrix elements are already 
            # being calculated, so its really no extra
            # cost to compute them.
            #
            # First, calculate the matrix elements in the 
            # [restart | restart] block, which maintains
            # the Lanczos recursion
            #
            # 
            # if i > (k+pad):
            #    beta = np.real( np.vdot(w, L[:,i-1]) ) 
            #    T[i,i-1] = beta
            #    T[i-1,i] = beta
            #
            # Now calculate matrix elements between the current
            # restart vector and the target block.
            #
            # We will do this during the usual first-pass 
            # orthogonalization 
            #
            for j in range(k+pad):
                c = np.vdot(L[:,j], w)
                T[j,i] = np.real(c) 
                T[i,j] = T[j,i]
                
                # And orthogonalize
                w -= c * L[:,j] 
            # Finish orthogonalizing against current restart vectors
            #
            for j in range(k+pad,i+1):
                c = np.vdot(L[:,j], w) # (nominally zero due to Lanczos recursion)
                if j < i:
                    T[j,i] = np.real(c) 
                    T[i,j] = T[j,i]
                    
                w -= c * L[:,j]
            
            # Normalize w 
            w /= np.linalg.norm(w) 
            
            # Project after first-pass orthogonalization 
            if project and i % pper == 0 and plevel >= 1: 
                w = P.matvec(w)
                if printlevel >= 2:
                    print("Projecting w")
                    
            # And orthogonalize again (second pass)
            for j in range(i+1):
                c = np.vdot(L[:,j], w)
                w -= c * L[:,j] # Sequential Gram-Schmidt orthogonalization
            
            # Finally, renormalize and store in Lnext     
            np.copyto(Lnext, w/np.linalg.norm(w))
            
            itercnt += 1 # Increment the iteration count (counting H matvec calls)
    
        # The restart is over and we have filled L, T, Lnext as before
        
        # Diagonalize the reduced Lanczos Hamiltonian T
        if restart_cnt > 0: 
            evals_old = evals.copy() # Save previous Ritz values
        
        evals,evecs = np.linalg.eigh(T)
        # eigh returns the evals in ascending order
        if eigval == 'smallest':
            pass # do nothing
        elif eigval == 'largest':
            # flip the order of the Ritz values/vectors
            evals = np.flip(evals)
            evecs = np.flip(evecs,axis = 1)
        
        # Store the first k+pad Ritz vectors
        np.copyto(L[:, 0:(k+pad)], np.matmul(L,evecs[:,0:(k+pad)]))
    
        ##########################
        # Refill T 
        # The target [k+pad] block will
        # be diagonal
        # 
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
            nunconverged = np.count_nonzero(np.abs(evals[0:k] - evals_old[0:k]) > tol)
            nconverged = k - nunconverged
            if nunconverged > 0:
                # At least one eigenvalue is above convergence
                # tolerance, we will continue
                if printlevel >= 1:
                    print(f"End of restart {restart_cnt:d}. {nconverged:d}/{k:d} Ritz values have converged.")
                if printlevel >= 2 or (printlevel >= 1 and restart_cnt % 10 == 0):
                    print("Current Ritz values:")
                    for i in range(k):
                        diff = evals[i]-evals_old[i]
                        print(f"  {evals[i]:.10E}  ({+diff:.10E})")
            else:
                # All eigenvalues have converged, exit the restart loop
                print("Convergence reached in {:d} restarts ({:d} iterations).".format(restart_cnt, itercnt))
                if printlevel >= 1 :
                    print("Final Ritz values:")
                    for i in range(k):
                        diff = evals[i]-evals_old[i]
                        print(f"  {evals[i]:.10E}  ({+diff:.10E})")
                    
                break
        
        restart_cnt += 1
    
    # We have reached convergence. Return the first `k` Ritz values and vectors
    w = evals[0:k].copy()
    v = L[:, 0:k].copy()
    
    return w,v


def bounds(H, k = 10, m = 3):
    """
    Estimate lower and upper bounds to the
    spectral range of the Hermitian operator H.

    Parameters
    ----------
    H : LinearOperator
        A Hermitian operator.
    k : int, optional
        The number of Lanczos steps to take
        before estimating the bounds. The 
        default is 10.
    m : int, optional
        The safe-guard level for the bound estimate.
        This must be less than or equal to `k`. The 
        default is 3.

    Returns
    -------
    low : float
        An estimate of the lower bound of the
        smallest eigenvalue.
    high : float
        An estimate of the upper bound of the
        largest eigenvalue.
        
    Notes
    -----
    These spectral bound estimates are based the work of Y. Zhou
    and R.-C. Li [ZL11]_. Their bounds (2.6), (2.7), and (2.8) 
    correspond to `m` = 1, `k`, and 3, respectively.
    
    In general, a larger value of `m` will provide a safer, but
    looser, bound on the spectral range.
    
    References
    ----------
    .. [ZL11] Y. Zhou and R.-C. Li, "Bound the spectrum of large Hermitian matrices."
       Linear Algebra and its Applications, 435, 480-493 (2011).
       https://doi.org/10.1016/j.laa.2010.06.034
    

    """
    
    N = H.shape[0] # The size of H
    k = 5 # The number of Lanczos steps
    
    
    # Initialize a random number generator
    rng = np.random.default_rng() 
    
    # Generate a random, normalized vector
    v = rng.random((N,))
    v = v / np.linalg.norm(v)
    
    # Initialize Lanczos steps
    f = H.matvec(v)
    alpha = np.vdot(f,v)
    f -= alpha * v 
    
    T = np.zeros((k,k))
    T[0,0] = alpha 
    
    for j in range(1,k):
        beta = np.linalg.norm(f)
        v0 = v 
        v = f/beta 
        f = H.matvec(v) 
        f -= beta * v0
        alpha = np.vdot(f,v) 
        f -= alpha*v 
        T[j,j-1] = beta 
        T[j-1,j] = beta 
        T[j,j] = alpha 
    
    # Calulate the Ritz values
    lam,Z = np.linalg.eigh(T)
    
    lmin = lam[0]  # Smallest Ritz value
    lmax = lam[-1] # Largest Ritz value
    
    z = np.absolute(Z[-1,:]) # The magnitude of the last element of each
                             # Ritz vector
    
    m = min(m,k)
    
    fnorm = np.linalg.norm(f)
    # Strategy 1:
    #
    #   max < lmax + |fk|
    # ( min > lmin - |fk| )  
    #
    # Strategy 2:
    # max < lmax + zmax*|fk|
    
    low = np.min( lmin - z[:m] * fnorm )
    high = np.max( lmax + z[-m:] * fnorm )
    
    
    return low, high 
    
def chebauto(H, K, v0 = None):
    """
    Calculate the auto-correlation function of a 
    wavepacket via the Chebyshev propagator of
    of a Hermitian operator H.

    Parameters
    ----------
    H : LinearOperator
        A Hermitian operator.
    K : int
        The number of propagation steps.
    v0 : ndarray, optional
        The initial vector. If None, a uniform vector is used.
        The default is None.

    Returns
    -------
    C : ndarray
        A (2*K+1,) array containing the auto-correlation function
        of `v0` in the Chebyshev order domain.
    scale : (float, float)
        The mean energy and half-width, (`ebar`, `de`) used to scale the
        Hamiltonian. The unscaled energy is related to the 
        Chebyshev angle parameter as ``E = de * cos(theta) + ebar``.
    
    See Also
    --------
    bounds : Used to scale the Hamiltonian
    chebspec : Calculate a spectrum from the Chebyshev auto-correlation function.
    
    Notes
    -----
    See Ref. [CG99]_ for a description of Chebyshev propagators.
    
    References
    ----------
    .. [CG99] R. Chen, H. Guo. "The Chebyshev propagator for 
       quantum systems." Computer Physics Communications, 119,
       19-31 (1999). https://doi.org/10.1016/S0010-4655(98)00179-9

    """
    
    
    ##################
    # Before beginning the Chebyshev propagation
    # we must calculate the scaled Hamiltonian.
    #
    # Estimate the spectral range:
    elow,ehigh = bounds(H, k = 10, m = 4)
    ebar = (elow + ehigh) / 2.0  # The mean eigenvalue
    de = (ehigh - elow) / 2.0    # The spectral width
    print("Estimated outer bounds of H:")
    print(f"  Low ... {elow:10.3f}")
    print(f" High ... {ehigh:10.3f}")
    #
    # Define a scaled Hamitonian matrix-vector 
    # product function
    def Hs(x):
        return (H.matvec(x) - ebar*x) / de 
    #
    ##################

    ##################
    # Initialize the auto-correlation function
    #
    C = np.zeros((2*K + 1,))
    #
    # Initial wavefunction
    NH = H.shape[0] 
    if v0 is None:
        c0 = np.full((NH,), np.sqrt(1.0 / NH))
    else:
        if np.size(v0) != NH:
            raise ValueError("The initial vector has an unexpected size.)")
        c0 = v0.reshape((NH,))
    #
    # Seed the Chebyshev vectors
    c1 = Hs(c0)
    # Initialize auto-correlation
    C[0] = np.dot(c0,c0) 
    # Initialize recursion
    ckm1 = c0
    ck   = c1
    # Perform propagation
    for k in range(1,K):
        # 
        # At beginning of loop
        # ck is c_k and
        # ckm1 = c_{k-1}
        #
        # Calculate auto-correlation
        C[k] = np.dot(ck,c0)
        
        # Calculate the next Chebyshev vector
        ckp1 = 2 * Hs(ck) - ckm1
        
        # Use symmetry to calculate auto-correlation
        # function for k >= K
        if 2*k >= K:
            C[2*k] = 2*np.dot(ck,ck) - C[0]
        
        if 2*k + 1 >= K:
            C[2*k+1] = 2*np.dot(ckp1,ck) - C[1]
        
        # Update Chebyshev vectors for next iteration
        ckm1 = ck
        ck   = ckp1
    #
    # At loop exit,
    # ck is c_{k = k}
    # 
    # Calculate the final element in the auto-correlation
    C[2*K] = 2*np.dot(ck,ck) - C[0] 
    #
    # C now contains the auto-correlation function 
    # in the Chebyshev order domain
    #
    # Return the raw auto-correlation.
    #########################
    
    return C, (ebar,de)

def chebwindow(N, window, window_scale = 1.0):
    
    """ 
    Calculate the window function for a Chebyshev
    order-domain correlation function.
    
    Parameters
    ----------
    N : int
        Length of correlation function.
    window : {'gaussian', 'boxcar', None}
        The window function type.
    window_scale : float
        A scaling parameter for the window function. Must be
        >= 1.0 The default is 1.0. See Notes for details.
        
    Returns
    -------
    f : ndarray
        The window function.
    hw : float
        The approximate half-width-half-maximum value
        (in units of radians).
    
    Notes 
    -----
    The window functions are always scaled to a dimensionless
    order-time parameter, tau, ranging from 0 to 1 over the total 
    propagation time. The 'gaussian' window is equal to
    ``np.exp(-(3.0 * window_scale * tau)**2)``. The 'boxcar'
    window is 1 for tau < 1/window_scale and 0 for tau >-
    1/window_scale.
    
    """
    tau = np.linspace(0.,1.,N+1)[:-1]
    
    if window_scale < 1.0 :
        print("Warning: resetting window_scale to 1.0")
        window_scale = 1.0 
        
        
    if window is None:
        ############
        # Do nothing
        # (equivalent to boxcar with window_scale = 1.0)
        f = np.ones_like(tau)
        hw = 1.9 / N
    elif window == 'boxcar':
        f = np.ones_like(tau)
        f[tau >= 1.0/window_scale] = 0.0 
        hw = window_scale * 1.9 / N 
    elif window == 'gaussian':
        f = np.exp(-(3.0*tau*window_scale)**2)
        hw = window_scale * 5.0 / N
    else:
        raise ValueError("Invalid window type")
    
    return f, hw

def chebspec(H, K, v0 = None, window = 'gaussian', window_scale = 1.0,
             sample_factor = 1, norm = 'area'):
    """
    Calculate the auto-correlation function of a 
    wavepacket via the Chebyshev propagator of
    of a Hermitian operator H. 

    Parameters
    ----------
    H : LinearOperator
        A Hermitian operator.
    K : int
        The number of propagation steps.
    v0 : ndarray, optional
        The initial vector. If None, a uniform vector is used.
        The default is None.
    window : {'gaussian', None}
        The window function. If None, no window function is used.
        The default is 'gaussian'. See Notes for details.
    window_scale : float
        A scaling parameter for the window function. The default
        is 1.0. See Notes for details.
    sample_factor : int
        The over-sampling factor. Must be an integer >= 1.
        The default is 1.
    norm : {'area', 'peak'}
        The normalization convention for the energy spectrum.
        The default is 'area'. See Notes for details.

    Returns
    -------
    E : ndarray
        The energy axis.
    G : ndarray
        The energy spectrum, normalized according to `norm`.
    fwhm : ndarray
        The energy-dependent FWHM line width. 
    C : ndarray
        The windowed auto-correlation function.
        
    See Also
    --------
    bounds : Used to scale the Hamiltonian
    chebauto : Calculates the Chebyshev auto-correlation function.
    chebwindow : Standard windows functions.
    
    Notes
    -----
    This spectral method follows that described in Ref. [CG99]_.
        
    If the `norm` parameter is 'area' (default), then 
    the returned spectrum `G` is normalized such that
    its energy integral is equal to the norm-squared of
    the initial vector, `v0`. 
    
    The non-linear relationship between energy and the
    Chebyshev angle lead to an energy-dependent line width.
    Therefore, the peak amplitude of a resolved transition
    will not be proportional to the spectral intensity
    globally. The `norm` is 'peak', then the spectrum
    is rescaled such that the peak amplitude of 
    individual, resolved transitions does correspond
    to the spectral intensity. The line widths are not 
    affected, however, such that the integrated area
    now no longer has a meaningful norm.

    """
    
    ################################
    # Compute the Chebyshev propagator
    # auto-correlation function 
    # (in the order domain)
    #
    C, scale = chebauto(H, K, v0)
    #
    ################################
    
    ################################
    # Apply window function
    f, hw = chebwindow(len(C), window, window_scale)
    C *= f 
    # 
    ################################
    
    ################################
    # Zero-pad the auto-correlation function
    # to over-sample the spectrum
    sample_factor = max(1,round(sample_factor))
    N = sample_factor * len(C) 
    Cz = np.zeros((N,))
    np.copyto(Cz[0:len(C)], C)
    # Cz is the zero-padded auto-correlation
    ################################
    
    ################################
    # Compute the discrete cosine transform,
    #
    # G(theta) = 1/pi * sum_k (2-delta_k,0) cos(k*theta) * C_k
    #
    # We use SciPy's built-in DCT. See 
    # SciPy documention for a careful description
    # of their conventions and normalization. The definition
    # of G(theta) above corresponds to 
    # Type III, norm = "backward"
    # with an extra prefactor of 1/pi
    #
    G = (1/np.pi) * scipy.fft.dct(Cz, type = 3)
    # The Chebyshev angles are uniformly distributed
    theta = (np.arange(N) + 0.5) * np.pi/ N
    # Compute the actual energy from the Chebyshev angles
    E = scale[0] + scale[1] * np.cos(theta)
    #
    # As defined above, G(theta) is normalized
    # such that its integral w.r.t. theta over 
    # [0,pi] is equal to C[0]. i.e. if the 
    # initial vector is a normalized wavefunction,
    # then this is unity. Upon transforming to the
    # energy domain, G(E) = G(theta) / sin(theta)
    # will be normalized to `de`, because
    # E = de * cos(theta) + ebar.
    # Therefore, if we want the `area` normalization
    # to integrate to the total "absorption" we need
    # to divide out by `de`.
    #
    if norm == 'peak':
        pass 
    elif norm == 'area':
        # G(E) integrates w.r.t. E 
        # to C[0]
        G /= (np.sin(theta) * scale[1])
    # 
    #################################
        
    ##################################
    # Finally, determine the energy-dependent 
    # FWHM line-width
    # 
    # | \Delta E | = de * sin(theta) * | \Delta theta | 
    #
    fwhm = (2*hw) * np.sin(theta) * scale[1]
    #
    ##################################
        
    
    return np.flip(E), np.flip(G), np.flip(fwhm), C
    
def full(H):
    """
    Return the full matrix of a LinearOperator

    Parameters
    ----------
    H : LinearOperator
        A matrix operator in LinearOperator form.

    Returns
    -------
    Hfull : ndarray
        The full, dense matrix.

    """
    
    n = H.shape[0] # the matrix rank
    return H.matmat(np.eye(n))

def msqrth(A):
    """
    Calculate the square-root matrix of a hermitian
    matrix, A.
    
    Parameters
    ----------
    A : array_like
        The matrix
    
    Returns
    -------
    rtA : ndarray
        The square-root matrix.
        
    Notes
    -----
    The square root is calculated via eigendecomposition. The
    square root of each eigenvalue is determined via NumPy's sqrt
    function, which defines the branch convention.
    """
    
    w,U = np.linalg.eigh(A)
    # w are the eigenvalues
    # U is the orthogonal eigenvector matrix
    
    # A = U @ w @ U.T 
    
    rtA = U @ np.diag(np.sqrt(w)) @ U.T  
    rtA = 0.5 * (rtA + np.conj(rtA.T)) # Enforce exact hermiticity.
    
    return rtA 


def simple_corr(H, dt, n, v0):
    """
    Simple finite time-step propagation for 
    an autocorrelation function.

    Parameters
    ----------
    H : ndarray
        The Hamiltonian matrix.
    dt : float
        The time step (in units of :math:`t/\\hbar`)
    n : integer
        The number of time steps.
    v0 : ndarray
        The initial vector.
        
    Returns
    -------
    C : ndarray
        The autocorrelation function.

    """
     
    
    # Calculate the small step propagator
    #    exp[-iH * dt]
    eHt = scipy.linalg.expm(-1j * H * dt)
    
    # Initialize the auto-correlation function
    C = np.zeros((n,), dtype = np.complex128)
    
    vt = v0  # The initial wavefunction 
    for i in range(n):
        C[i] = np.vdot(v0, vt) # Calculate overlap with initial wavefunction
        vt = eHt @ vt          # Propagate one step
    
    return C 

class RealOrthogonalProjector(LinearOperator):
    """
    A projection operator to orthogonalize against a given 
    set of vectors
    
    ..  math::
        
        \hat{P} = 1 - U U^T
    
    
    """
    
    def __init__(self,U):
        """
        
        Parameters
        ----------
        U : (N,n) ndarray
            `n` column vectors to orthogonalize against. The columns
            of `U` are assumed to be real and orthonormal.

        """
        
        N = U.shape[0] 
        self.shape = (N,N)
        self.dtype = U.dtype # must be a real type!
        
        self.N = N 
        self.U = U 
        
    def _matvec(self, x):
        """
        The matrix-vector product 
        """
        
        x = x.reshape((self.N,))
        
        y = x.copy() 
        
        y -= self.U @ (self.U.T @ x) # Note tranpose b/c real 
        
        return y 
 