import numpy as np
import scipy.special

def _besDVR(start,stop,num,nu):
    """
    Construct a Bessel DVR with an 
    angular momentum parameter `nu`

    Parameters
    ----------
    start : float
        Minimum value of first DVR grid point.
    stop : float
        DVR grid stop value
    num : int
        Number of DVR functions
    nu : int
        Integer order parameter

    Returns
    -------
    None.

    """
    
    if nu < 0:
        raise ValueError("nu must be non-negative")
    if start >= stop:
        raise ValueError("start must be < stop")
    if start < 0 or stop < 0:
        raise ValueError("start and stop must be >= 0")
        
    # Look for the correct set of zeros
    nz = num
    while True: 
        z = scipy.special.jn_zeros(nu, nz)
        K = z[-1] / stop
        r = z / K
        
        if r[-num] >= start:
            # we are done
            break 
        else:
            # continue, looking for one more zero
            nz += 1
    
    # Construct the full KEO operator
    #
    #  T = -d^2/dr^2 + (nu^2 - 1/4) / r^2
    T = np.zeros((nz,nz))
    for i in range(nz):
        for j in range(nz):
            
            if i == j:
                T[i,j] = K**2/3.0 * (1+(2*(nu**2-1))/z[i]**2)
            else:
                T[i,j] = (-1)**(i-j) * 8.0 * K**2 \
                    * z[i]*z[j]/(z[i]**2 - z[j]**2)**2
    # Construct the full D2 operator
    d2 = -T + np.diag((nu**2 - 1/4.0) / r**2)
    #
    
    # Construct an approximate d operator
    d = np.zeros((nz,nz))
    for i in range(nz):
        for j in range(nz):
            if i == j:
                d[i,j] = 0.0
                continue
            
            # J'(nu, K*r[i])
            # Jpi = 0.5*(scipy.special.jn(nu-1, z[i]) - scipy.special.jn(nu+1,z[i] ))
            # Ji = scipy.special.jn(nu,K*r[i]) # this should always be zero
            
            den = (K*r[i])**2 - z[j]**2
            
            dFj = (-1)**(i+1)*K*z[j]*(np.sqrt(2*r[i])*K/den) # * Jpi
            # The next term should always be zero
            # + (0.5*np.sqrt(2/r[i])/den - np.sqrt(8*r[i])*K*r[i]/den**2) * Ji)
            
            
            Fi = (-1)**(i+1) * np.sqrt(K*z[i]/2.0) # * Jpi
            d[i,j] = dFj/Fi
    #
    # Force skew-symmetric
    d = (d - d.transpose()).copy() * 0.5
    
    # Calculate the truncated arrays
    grid = r[-num:].copy()
    D2 = d2[-num: , -num:].copy()
    D = d[-num:, -num:].copy()
    # This D is only an "ok" approximation.
    # For now, we will return None
    
    return grid, None, D2