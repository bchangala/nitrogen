"""
nitrogen.dvr
------------

This module provides support for discrete-variable 
representation (DVR) basis functions. The main object
is the :class:`DVR` class.

"""

import numpy as np


class DVR:
    """
    A 1D DVR basis class
    
    Attributes
    ----------
    start : float
        The DVR grid starting value.
    stop : float
        The DVR grid stopping value.
    num : int
        The number of DVR grid points.
    basis : {'sinc','ho','fourier', 'legendre'}
        The DVR basis type.
    grid : ndarray
        The DVR grid points.
    D : ndarray
        First derivative operator in DVR representation.
    D2 : ndarray
        Second derivative operator in DVR representation.
    
    """
    
    def __init__(self, start = 0, stop = 1, num = 10, basis = 'sinc'):
        """
        Create a DVR object.

        Parameters
        ----------
        start : float, optional
            The DVR grid starting value. The default is 0.
        stop : float, optional
            The DVR grid stopping value. The default is 1.
        num : int, optional
            The number of DVR grid points. The default is 10.
        basis : {'sinc','ho','fourier','legendre'}, optional
            The DVR basis type. The default is 'sinc'.

        """
        
        if stop <= start:
            raise ValueError("stop value must be larger than start value")
        if num < 2 :
            raise ValueError("num must be >= 2")
        
        self.start = start 
        self.stop = stop 
        self.num = num 
        self.basis = basis.lower()
        
        if self.basis == 'sinc' : 
            self.grid, self.D, self.D2 = _sincDVR(start, stop, num)
            self.wfs = lambda q : _sincDVRwfs(q, self.start, self.stop, self.num)
        elif self.basis == 'ho' : 
            self.grid, self.D, self.D2 = _hoDVR(start, stop, num)
            self.wfs = lambda q : _hoDVRwfs(q, self.start, self.stop, self.num)
        elif self.basis == 'fourier' : 
            self.grid, self.D, self.D2 = _fourDVR(start, stop, num)
            self.wfs = lambda q : _fourDVRwfs(q, self.start, self.stop, self.num)
        elif self.basis == 'legendre' : 
            self.grid, self.D, self.D2 = _legDVR(start, stop, num)
            self.wfs = lambda q : _legDVRwfs(q, self.start, self.stop, self.num)
        else:
            raise ValueError("basis type '{}' not recognized".format(basis))
    
    
    def wfs(self, q):
        """
        Evaluate DVR basis wavefunctions.

        Parameters
        ----------
        q : ndarray
            A 1-D array of coordinate values.

        Returns
        -------
        wfs : ndarray
            An array of shape (`q`.size, :attr:`num`) containing
            the values of the DVR wavefunctions evaluated at 
            coordinate values `q`.

        """
        pass # This will be replaced by an instance method for a given DVR basis type
            
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
            t += 2*np.cos(k*(q - i*period/num) * 2*np.pi/period)
            
        wfs[:,i] = t/np.sqrt(num*period)
    
    return wfs

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