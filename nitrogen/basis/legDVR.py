import numpy as np

def _legDVR(start,stop,num):
    """ 
    Calculate Legendre DVR grid and operators
    
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
        Symmetrized derivative operator (d^dagger * d) in DVR basis, shape (num, num)
        
    Notes
    -----
    The Legendre DVR is defined over a finite integration domain with
    non-periodic boundary conditions. The derivative operator is thus not
    stricty anti-Hermitian because of non-zero boundary terms. The `D2`
    matrix returned is the representation of the explicitly symmetrized 
    :math:`-\partial^\dagger \cdot \partial`
    operator, which is *not* equivalent to :math:`\partial \cdot \partial`

    """
    
    # Construct coordinate operator in FBR
    XFBR = np.zeros((num,num))
    for p in range(num-1):
        
        val = np.sqrt(2*p+1) * np.sqrt(2*p+3) * (stop-start)/4.0 \
            * 2*(p+1) / ( (2*p+1)*(2*p+3))
        XFBR[p,p+1] = val 
        XFBR[p+1,p] = val
        
    XFBR += np.eye(num)*(stop+start)/2.0
    
    # Calculate derivative operator matrix elements
    DFBR = np.zeros((num,num))
    D2FBR = np.zeros((num,num)) # -d^dagger * d
    for n in range(num):
        for m in range(num):
            
            if (m-n) % 2 == 1 and n <= m:
                DFBR[n,m] = np.sqrt( (2*n+1) * (2*m+1) ) * 2.0/(stop-start)
            
            if (m-n) % 2 == 0 :
                p = min(m,n)
                D2FBR[m,n] = -np.sqrt(2*m+1) * np.sqrt(2*n+1) \
                    * 2.0 / (stop-start)**2 * p*(p+1)
    
    # Diagonalize coordinate matrix in FBR
    w,v = np.linalg.eigh(XFBR)
    # Adjust phase of eigenfunctions
    for j in range(num):
        if v[0,j] < 0:
            v[:,j] *= -1.0 
    # Calculate derivative operators in DVR 
    D = v.transpose() @ DFBR @ v
    D2 = v.transpose() @ D2FBR @ v
    
    grid = w
    
    return grid, D, D2

def _legDVRwfs(q,start,stop,num):
    """
    Calculate Legendre DVR wavefunctions.

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
    XFBR = np.zeros((num,num))
    for p in range(num-1):
        
        val = np.sqrt(2*p+1) * np.sqrt(2*p+3) * (stop-start)/4.0 \
            * 2*(p+1) / ( (2*p+1)*(2*p+3))
        XFBR[p,p+1] = val 
        XFBR[p+1,p] = val
        
    XFBR += np.eye(num)*(stop+start)/2.0
      
    # Diagonalize coordinate matrix in FBR
    w,v = np.linalg.eigh(XFBR)
    # Adjust phase of eigenfunctions
    for j in range(num):
        if v[0,j] < 0:
            v[:,j] *= -1.0 
    
    z = (q - (start+stop)/2.0) * 2.0/(stop-start)
    
    # Calculate the standard Legendre polynomials
    # via the recurrence relation
    #  n*P_n = (2n-1) z P_{n-1}  - (n-1) P_{n-2}
    #
    leg = np.ndarray((nq,num), dtype = q.dtype)
    leg[:,0] = 1
    leg[:,1] = z
    for n in range(2,num):
        leg[:,n] = (2*n-1)/n * z*leg[:,n-1]  -  (n-1)/n * leg[:,n-2]
    # Now, normalize the functions for (start, stop)
    for n in range(0,num):
        leg[:,n] *= np.sqrt((2*n+1)/(stop-start))
    
    wfs = leg @ v
    
    return wfs