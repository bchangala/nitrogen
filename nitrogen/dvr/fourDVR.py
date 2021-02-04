import numpy as np

def _fourDVR(start,stop,num):
    """ 
    Calculate periodic Fourier (exponential) DVR grid and operators.
    
    Parameters
    ----------
    start : float
        Grid start value
    stop : float
        Grid stop value. Must be larger than start
    num : int
        Number of grid points. Must be >= 3 and odd.

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
    if num < 3 :
        raise ValueError("num must be >= 3")
    if num % 2 == 0:
        raise ValueError("num must be odd")
        
    
    grid = np.linspace(start, stop, num+1)[0:num]
    period = stop - start
    
    D = np.ndarray((num,num))
    D2 = np.ndarray((num,num))
    
    for i in range(num):
        for j in range(i,num):
            delta = i - j 
            if delta == 0:
                D[i,j] = 0
                D2[i,j] = -(np.pi / period)**2 * (1/3.0) * (num**2 - 1)
            else:
                D[i,j] = (np.pi / period) * (-1)**delta / np.sin(np.pi*delta/num)
                D[j,i] = -D[i,j]
                
                D2[i,j] = -2*(np.pi / period)**2 * (-1)**delta * np.cos(np.pi * delta / num) / (np.sin(np.pi*delta/num)**2)
                D2[j,i] = +D2[i,j]
    
    return grid, D, D2

def _fourDVRwfs(q, start, stop, num):
    """
    Calculate Fourier (exponential) DVR wavefunctions.

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
    if num < 3 :
        raise ValueError("num must be >= 3")
    if num % 2 == 0:
        raise ValueError("num must be odd")
        
    nq = q.size
    
    period = stop - start
    
    wfs = np.ndarray((nq,num), dtype = q.dtype)
    
    m = (num-1)//2 # max frequency
    
    for i in range(num): # DVR function at i^th grid point
        t = 1
        for k in range(1,m+1): # Calculate exponential sum as cosines explicitly
            t += 2*np.cos(k*(q - start - i*period/num) * 2*np.pi/period)
            
        wfs[:,i] = t/np.sqrt(num*period)
    
    return wfs