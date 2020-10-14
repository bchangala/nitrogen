import numpy as np
import scipy.special
import warnings 

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
    nu : float
        The Bessel order `nu`.

    Returns
    -------
    grid, D, D2 : ndarrays

    """
    
    if nu < 0:
        raise ValueError("nu must be non-negative")
    if start >= stop:
        raise ValueError("start must be < stop")
    if start < 0 or stop < 0:
        raise ValueError("start and stop must be >= 0")
        
        
    # Get the grid points for the (start, stop) range
    z, K, r, nz = _besDVRzeros(start, stop, num, nu)
    
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
    
    # Construct the quadrature
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
    # D = d[-num:, -num:].copy()
    # This D is only an "ok" approximation.
    # For now, we will return None
    
    return grid, None, D2

def _besDVRwfs(q, start, stop, num, nu):
    
    z, K, r, nz = _besDVRzeros(start, stop, num, nu)
    
    nq = q.size
    
    wfs = np.ndarray((nq,num), dtype = q.dtype)
    
    for i in range(num):
        wfs[:,i] = _besFnun( i + (nz-num), nu, K, z, q)
    
    return wfs 

def _besDVRquad(start, stop, num, nu):
    
    z, K, r, nz = _besDVRzeros(start, stop, num, nu)
    
    Fn = np.zeros(nz)
    
    # Calculate each DVR basis function at its grid point
    for n in range(nz):
        Fn[n] = (-1)**(n+1) * np.sqrt(K * z[n] / 2.0) * scipy.special.jvp(nu, z[n])
        
    wgts = 1.0 / Fn**2
    
    return r[-num:], wgts[-num:]
    
    
def _besFnun(n, nu, K, z, r):
    """ z is the list of zeros of a given order
        Bessel function J_nu,
        with z[0] being the first 
        
    """
    
    ZERO_TOL = 1e-10  # Small denominator threshold value
    
    num = (-1) ** (n+1) * K * z[n] * np.sqrt(2*r) / (K*r + z[n])
    den = (K*r - z[n])
    
     
    I = np.abs(den) > ZERO_TOL
    
    F = np.empty_like(r)
    
    # For values away from the grid point, just evaluate normally
    J = scipy.special.jv(nu, K*r)   
    F[I] = num[I] * J[I] / den[I]
    
    # For values near the gridpoint, expand about the zero
    Jp = scipy.special.jvp(nu, z[n], n=1)   # First derivative
    Jpp = -Jp/z[n] # via the defining differential equation at a zero of J_nu
    F[~I] = num[~I] * (Jp + 0.5*Jpp * den[~I])
    
    return F
    

def _besDVRzeros(start, stop, num, nu):
    
    # Look for the correct set of zeros
    nz = num
    while True: 
        #z = scipy.special.jn_zeros(nu, nz)
        z = _besselzero(nu, nz)
        K = z[-1] / stop
        r = z / K
        
        if r[-num] >= start:
            # we are done
            break 
        else:
            # continue, looking for one more zero
            nz += 1
            
    return z, K, r, nz

def _besselzero(nu, nz = 5):
    """
    The zeros of Bessel functions of the first kind.
    
    This algorithm is adapated from the MATLAB besselzero function
    (% Originally written by 
     % Written by: Greg von Winckel - 01/25/05
     % Contact: gregvw(at)chtm(dot)unm(dot)edu
     %
     % Modified, Improved, and Documented by 
     % Jason Nicholson 2014-Nov-06
     % Contact: jashale@yahoo.com

    Parameters
    ----------
    nu : float
        The order of the Bessel function. 
    nz : int
        The number of zeros requested. The default is 5.

    Returns
    -------
    ndarray
        First nz zeros of the Bessel function.

    """
    
    # Check arguments
    if nz < 1:
        raise ValueError("nz must be >= 1")
        
    ORDER_MAX = 146222.16
    if nu > ORDER_MAX:
        raise ValueError(f"nu must be less than {ORDER_MAX:.10f}")
        
    x = np.zeros(nz)
    
    coeffs1j = [0.411557013144507, 0.999986723293410, 0.698028985524484, 1.06977507291468]
    exponent1j = [0.335300369843979, 0.339671493811664]
    
    # guess for nz = 1

    x[0] = coeffs1j[0] + coeffs1j[1] * nu \
     + coeffs1j[2] * (nu+1)**(exponent1j[0]) + coeffs1j[3] * (nu+1)**(exponent1j[1])
    # find first root
    x[0] = _findzero(nu, 1, x[0])
    
    if nz >= 2:
        # guess for second root
        coeffs2j = [1.93395115137444, 1.00007656297072, -0.805720018377132, 3.38764629174694]
        exponent2j = [0.456215294517928, 0.388380341189200]
        x[1] = coeffs2j[0] + coeffs2j[1] * nu \
         + coeffs2j[2] * (nu+1)**(exponent2j[0]) + coeffs2j[3] * (nu+1)**(exponent2j[1])
        # find second root
        x[1] = _findzero(nu, 2, x[1]) 
    
    if nz >=3:
        # guess for third root
        coeffs3j = [5.40770803992613, 1.00093850589418, 2.66926179799040, -0.174925559314932]
        exponent3j = [0.429702214054531,0.633480051735955]
        x[2] = coeffs3j[0] + coeffs3j[1] * nu \
         + coeffs3j[2] * (nu+1)**(exponent3j[0]) + coeffs3j[3] * (nu+1)**(exponent3j[1])
        # find third root
        x[2] = _findzero(nu, 3, x[2]) 
    
    if nz >= 4:
        for i in range(3,nz):
            # Guesses for remaining roots
            # x[k] = spacing + x[k-1]
            spacing = x[i-1] - x[i-2]
            x0 = spacing + x[i-1] # guess for x[i]
            x[i] = _findzero(nu, i+1, x0)
    
    return x
    
def _findzero(nu, k, x0):
    """
    Find the k^th zero of Bessel_nu.

    Parameters
    ----------
    nu : float
        Bessel order
    k : int
        The zero's index (starting at 1).
    x0 : float
        Initial guess

    Returns
    -------
    x : float
        The zero.

    """
    
    MAX_ITER = 100
    REL_TOL = 1e4 
    
    error = 1.0
    loopCount = 0 
    x = 1 
    
    while np.abs(error) > np.spacing(x)*REL_TOL and loopCount < MAX_ITER:
        a = scipy.special.jv(nu, x0)
        b = scipy.special.jv(nu+1, x0)
        
        xSquared = x0 * x0 
        
        num = 2*a*x0*(nu*a - b*x0)
        den = (2*b*b*xSquared-a*b*x0*(4*nu+1)+(nu*(nu+1)+xSquared)*a*a)
        error = num/den
        
        # Prepare for next loop
        x = x0 - error 
        x0 = x 
        loopCount += 1 
    
    if loopCount > MAX_ITER - 1:
        warnings.warn("Failed to converge to within rel. tol. of {:e} for nu={:f} and k = {:d} in {:d} iterations".format(
            np.spacing(x)*REL_TOL, nu, k, MAX_ITER))
        
    return x 