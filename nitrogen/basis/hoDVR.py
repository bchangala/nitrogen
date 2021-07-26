import numpy as np

def _hoDVR(start,stop,num):
    """ 
    Calculate harmonic oscillator DVR grid and operators
    
    Parameters
    ----------
    start : float
        Grid start value
    stop : float
        Grid stop value. Must be larger than start
    num : int
        Number of grid points. Must be >= 2.

    Returns
    -------
    grid : ndarray
        Array of DVR grid points
    D : ndarray
        First derivative operator in DVR basis, shape (num, num)
    D2 : ndarray (num,num)
        Second derivative operator in DVR basis, shape (num, num)

    """
    
    if stop <= start:
        raise ValueError("stop value must be larger than start value")
    if num < 2 :
        raise ValueError("num must be >= 2")
        
    # Construct coordinate and derivative operators 
    # in finite basis represenation (i.e. harmonic oscillator representation)
    # with m*omega = hbar
    # Q = 1/sqrt(2) * ( a + a^dagger)
    QFBR = np.zeros((num,num))
    for i in range(num):
        for j in range(num):
            if i == j + 1:
                QFBR[i,j] = np.sqrt(i/2.0)
            elif i == j - 1:
                QFBR[i,j] = np.sqrt(j/2.0)
                
    # d/dQ = 1/sqrt(2) * (a - a^dagger))
    DFBR = np.zeros((num,num))
    for i in range(num):
        for j in range(num):
            if i == j + 1:
                DFBR[i,j] = -np.sqrt(i/2.0)
            elif i == j - 1:
                DFBR[i,j] = np.sqrt(j/2.0)
    # d^2/dQ^2 = (d/dQ)^2
    #          = 1/2 * (a^2 + (a^dagger)^2 - (2N+1))
    D2FBR = np.zeros((num,num))
    for i in range(num):
        for j in range(num):
            if i == j + 2:
                D2FBR[i,j] = 0.5 * np.sqrt( i*(i-1.0))
            elif i == j - 2:
                D2FBR[i,j] = 0.5 * np.sqrt( j*(j-1.0))
            elif i == j:
                D2FBR[i,j] = -(i+0.5)
    
    # Diagonalize the coordinate operator
    w,v = np.linalg.eigh(QFBR)
    # Adjust phase of eigenfunctions
    for j in range(num):
        if v[0,j] < 0:
            v[:,j] *= -1.0 
    # Calculate derivative operators in unscaled DVR representation
    d = v.transpose() @ DFBR @ v
    d2 = v.transpose() @ D2FBR @ v
    
    # Calculate the shift and scale required to
    # generated the requested grid
    scale = (stop - start) / (w[-1] - w[0])
    shift = (stop + start) / 2.0 # (w is symmetric about 0)
    
    grid = scale*w + shift
    grid[0] = start # force strict equality for start and stop
    grid[-1] = stop
    
    D = d / scale 
    np.fill_diagonal(D, 0) # force zero on diagonal of D
    D2 = d2 / (scale*scale)
                
    return grid,D,D2

def _hoDVRwfs(q,start,stop,num):
    """
    Calculate harmonic oscillator DVR wavefunctions.

    Parameters
    ----------
    q : ndarray
        A 1D array of coordinate values.
    start : float
        DVR grid start value.
    stop : float
        DVR grid stop value.
    num : int
        The number of DVR grid points

    Returns
    -------
    wfs : ndarray
        A (`q`.size, `num`) shaped array with
        the DVR wavefunctions evaluated at grid points `q`.

    """
    
    if np.ndim(q) != 1:
        raise ValueError("q must be 1-dimensional")
    
    if stop <= start:
        raise ValueError("stop value must be larger than start value")
    if num < 2 :
        raise ValueError("num must be >= 2")
        
    nq = q.size
    
    
    # Construct coordinate operator in FBR
    # with m*omega = hbar
    # Q = 1/sqrt(2) * ( a + a^dagger)
    QFBR = np.zeros((num,num))
    for i in range(num):
        for j in range(num):
            if i == j + 1:
                QFBR[i,j] = np.sqrt(i/2.0)
            elif i == j - 1:
                QFBR[i,j] = np.sqrt(j/2.0)
    
    # Diagonalize the coordinate operator
    w,v = np.linalg.eigh(QFBR)
    # Adjust phase of eigenfunctions
    for j in range(num):
        if v[0,j] < 0:
            v[:,j] *= -1.0 
    scale = (stop - start) / (w[-1] - w[0])
    shift = (stop + start) / 2.0 # (w is symmetric about 0)
    #    q = scale*x + shift
    # -> x = (q-shift) / scale
    x = (q - shift) / scale
    
    # Calculate the unscaled Hermite polynomials
    # via the recurrence relation
    #  H_n = 2x*H_n-1 - 2(n-1)H_n-2
    #
    herm = np.ndarray((nq,num), dtype = q.dtype)
    herm[:,0] = 1
    herm[:,1] = 2*x
    for n in range(2,num):
        herm[:,n] = 2*x*herm[:,n-1] - 2*(n-1)*herm[:,n-2]
    # Calculate the unscaled harmonic oscillator
    # wavefunctions
    uho = np.empty_like(herm)
    nfact = 1.0
    for n in range(0,num):
        uho[:,n] = herm[:,n] * np.exp(-x*x / 2.0) / np.sqrt( 2**n * nfact * np.sqrt(np.pi))
        nfact = nfact * (n+1.0)
    
    # Calculate the scaled wavefunctions, which
    # are properly normalized w.r.t coordinate q
    sho = uho / np.sqrt(scale)
    
    wfs = sho @ v
    
    return wfs
