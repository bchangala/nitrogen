import numpy as np

def _sincDVR(start,stop,num):
    """ 
    Calculate sinc DVR grid and operators
    
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
        

    grid = np.linspace(start,stop,num)
    
    D = np.zeros((num,num))
    D2 = np.zeros((num,num))

    delta = grid[1]-grid[0] # Grid spacing
    
    for i in range(num):
        for j in range(num):
            
            # Calculate first derivative operator D
            if i==j:
                pass #zero
            else:
                D[i,j] = (-1.0)**(i-j) / (delta * (i-j))
            
            # Calculate second derivative operator D2
            if i==j:
                D2[i,j] = -np.pi**2 / (3.0 * delta**2)
            else:
                D2[i,j] = -2.0 * (-1.0)**(i-j) / (delta**2 * (i-j)**2)
    
    
    return grid,D,D2

def _sincDVRwfs(q,start,stop,num):
    """
    Calculate sinc DVR wavefunctions.

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
        
    nq = q.size
    
    wfs = np.ndarray((nq,num), dtype = q.dtype)
    
    grid = np.linspace(start,stop,num)
    delta = grid[1] - grid[0]
    
    for i in range(num):
        # Compute sinc DVR basis function at
        # grid position i
        # Note: np.sinc function is defined as sin(pi*x)/pi*x 
        #
        fi = np.sqrt(1.0/delta) * np.sinc( (q - grid[i])/delta )
        wfs[:,i] = fi
        
    return wfs
