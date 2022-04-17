"""
nitrogen.math
-----------------

General mathematical tools and functions

"""

import numpy as np 

import nitrogen.autodiff.forward as adf 

def cumsimp(y, x, axis = 0):
    """
    Calculate the cumulative integral by Simpson's Rule

    Parameters
    ----------
    y : array_like
        The uniformly spaced samples
    x : array_like or scalar
        The sample points along the integration axis or, if scalar, the sample spacing.
    axis : int, optional
        The integration axis. The default is 0.

    Returns
    -------
    res : ndarray
        The result of cumulative integration.
        
    Notes
    -----
    
    The first element of the cumulative integral is 
    fixed to zero. The second element is handled in one of two ways. 
    If the length of `y` is only 2, then the trapezoid rule is used. Otherwise,
    the next element in `y` is used for quadratic interpolation.
    The remaining elements are evaluated via Simpson's 1/3 rule.
    
    """
    
    y = np.array(y)
        
    if y.shape[axis] < 1:
        raise ValueError('y must be have non-zero length along integration axis')
        
    if not np.isscalar(x):
        if len(x) != y.shape[axis]:
            raise ValueError('x must have the same integration points as y') 
            # This is not strictly necessary because we assume the 
            # integration points are uniform, but I will enforce this 
            # check because it will probably catch some logical errors
            # or unintentional parameters
    
    # Move integration axis to the front 
    y = np.moveaxis(y, axis, 0) 
    
    res = np.zeros_like(y)
    
    # First element is always zero
    # res[0] = 0.0 

    ylen = y.shape[0] # The integration length 
    
    if ylen > 1:    
        
        # Gather the sample spacing
        if np.isscalar(x):
            dx = x 
        else:
            dx = x[1] - x[0] 
        
        
        # Second element has two cases
        if ylen == 2:
            # 1) The total length is only 2
            # Use trapezoid rule 
            res[1] = dx * 0.5 * (y[0] + y[1]) 
        else: 
            # 2) The length is greater than 2
            #    Use quadratic interpolation for the
            #    first semgment ("half-way Simpson's Rule")
            res[1] = (dx/12) * (5*y[0] + 8*y[1] - y[2])
    
    if ylen > 2:
        for i in range(2,ylen):
            # For the third and later elements,
            # use the standard 1/3 Simpson's rule 
            res[i] = res[i-2] + (dx/3) * (y[i-2] + 4*y[i-1] + y[i])
            
    # Move integration axis back to original position 
    res = np.moveaxis(res, 0, axis) 
    return res 


def spech_fft(C,dt,sample_factor = 1, damping = 0.0):
    """
    Calculate the intensity spectrum from a hermitian
    autocorrelation function C.

    Parameters
    ----------
    C : ndarray
        The autocorrelation function.
    dt : float
        The time step.
    sample_factor : int
        The over-sampling factor. The default is 1.
    damping : float
        The Gaussian damping factor. The default is 0.
    

    Returns
    -------
    g : ndarray
        The (real) intensity spectrum
    freq : ndarray
        The angular frequency axis 
    
    Notes
    -----
    The autocorrelation function is provided for :math:`t \geq 0`, i.e.
    C[0] is the :math:`t = 0` value. It is assumed that :math:`C(-t) = C(t)^*`.
    
    The returned spectrum approximates the Fourier transformation
    
    ..  math::
        
        g(\\omega) = \\frac{1}{2\\pi}\\, \\int_{-\\infty}^{\\infty} dt e^{i \\omega t} C(t)
    
    
    This normalization means that the angular frequency integral of 
    :math:`g(\\omega)` equals :math:`C(0)`, which is typically unity.
    
    
    The calculated spectrum may be over-sampled by zero-padding the 
    autocorrelation function. This sampling factor is controlled by
    `sample_factor`. 
    
    A Gaussian window function is also applied to the autocorrelation 
    function, of the form :math:`C(t) \\exp(-a (t/T)^2)`, where 
    :math:`T` is length of the correlation function (before any padding).
    The damping factor :math:`a` is controlled by the `damping` keyword.
    At sufficiently large :math:`a`, the line shape becomes Gaussian
    with an angular frequency full-width at half-maximum equal to
    :math:`\\omega_\\text{FWHM} = 4 T^{-1} \\sqrt{a \\ln 2}`.
    
    """
    
    
    # First window and zero-pad as necessary 
    tau = np.linspace(0.0, 1.0, len(C))
    C = C * np.exp(-damping * tau**2) 
    # zero-pad
    sample_factor = int(sample_factor)
    if sample_factor < 1: 
        raise ValueError("sample_factor must be a postive integer")
    Cz = np.zeros(len(C) * sample_factor, dtype = C.dtype) 
    Cz[:len(C)] = C[:] 
    
    #
    # Because C is hermitian, we can reconstruct
    # its negative-time values explicitly
    # (We could also just use numpy's hfft, but I will do it this way instead)
    fpos = Cz # t >= 0
    fneg = Cz[:0:-1].conj()  # t < 0 
    
    f = np.concatenate((fneg, fpos)) # [-T,T]
    N = len(f)  # N is odd 
    n = (N-1) // 2 # integer
    
    F = np.fft.fft(f, norm = None) # Use standard normalization fft
    k = np.concatenate((np.arange(0,-(n+1), -1), np.arange(n,0,-1)))
    
    t0 = -(N-1)/2 * dt 
    
    # Correct for DFT phase 
    G = (dt/(2*np.pi)) * np.exp(+1j * 2*np.pi * t0 / (N*dt)  * k) * F
    
    # Calculate angular frequency values
    dw = (2*np.pi) / (N*dt)
    freq = k * dw 
    
    # Re-sort result in ascending frequency 
    IDX = np.argsort(freq) 
    freq = freq[IDX]
    g = np.real(G[IDX]) 
    
    return g, freq 


def gaussianFWHM(x, fwhm, norm = 'area'):
    """
    Calculate a Gaussian function.
    
    Parameters
    ----------
    x : array_like
        Input array
    fwhm : scalar
        The full-width at half-maximum value.
    norm : {'area', 'amplitude'}
        The normalization convention. The default is 'area'. See Notes for
        definitions.
    
    Returns
    -------
    y : ndarray
        The result.
        
    Notes
    -----
    For `norm` = ``\`area\```, the integrated area is equal to 
    unity. For `norm` = ``\`amplitude\```, the
    peak amplitude is equal to unity.
    
    """
    
    y = np.exp(-4*np.log(2) * (x/fwhm)**2)
    
    if norm == 'area':
        y *= np.sqrt(np.log(16)/np.pi) / fwhm 
    elif norm == 'amplitude':
        pass
    else:
        raise ValueError('unexpected norm keyword')
    
    return y 


def mpolyfit(x, y, deg):
    """
    Multivariable polynomial least-squares fitting.

    Parameters
    ----------
    x : (N,nx) or (N,) array_like
        The input coordinates.
    y : (N,) array_like
        The output coordinates
    deg : int
        The degree of the polynomial

    Returns
    -------
    p : (nt,) ndarray
        The polynomial coefficients
    res : (N,) ndarray
        The residuals.
    
    Notes
    -----
    The polynomial is ordered using the standard 
    lexical ordering defined by the ``autodiff``
    and ``DFun`` modules.

    """
    
    y = np.array(y)
    N = y.shape[0] # The number of data points
    x = np.array(x)
    x = x.reshape((N,-1)) # Force to 2-dim
    nx = x.shape[1]  # The number of variables 
    
    # Calculate the powers
    # in standard lexical ordering
    pows = adf.idxtab(deg, nx) 
    nt = pows.shape[0] # the number of terms
    
    # Create the least-squares array
    # Initialize to all ones
    C = np.ones((N,nt), dtype = x.dtype)
    
    # Calculate the powers of the inputs
    xpow = [ [ x[:,i] ** k for k in range(deg+1)] for i in range(nx)]
    
    for r in range(nt):
        pr = pows[r,:] # the powers of this term 
        for i in range(nx):        
            C[:,r] *= xpow[i][pr[i]]
    
    # Now solve the linear least-squares problem
    p,_,_,_ = np.linalg.lstsq(C, y, rcond=None)
    
    res = y - C @ p
    
    return p, res
        
def mpolyval(p, x):
    """
    Evaluate a polynomial in standard lexical order.
    
    Parameters
    ----------
    p : array_like
        The polynomial coefficients.
    x : (N,nx) or (N,) array_like
        The input values.

    Returns
    -------
    y : (N,) ndarray
        The polynomial value.

    """
    
    p = np.array(p)
    x = np.array(x)
    
    if x.ndim != 2:
        # Assume (N,) shaped
        x = x.reshape((-1,1))
        
    N = x.shape[0] # The number of data points
    nx = x.shape[1] # The number of variables 
    
    nt = len(p) # The number of terms
    
    # Figure out the degree of the polynomial
    deg = 0 
    while True:
        if nt == adf.nderiv(deg, nx):
            break # found 
        elif nt < adf.nderiv(deg, nx):
            raise ValueError("The length of p appears inconsistent with x.")
        deg += 1 
    
    # Calculate the powers
    # in standard lexical ordering
    pows = adf.idxtab(deg, nx) 

    # Calculate the powers of the inputs
    xpow = [ [ x[:,i] ** k for k in range(deg+1)] for i in range(nx)]
    
    
    y = np.zeros((N,), dtype = np.result_type(p,x))
    
    for r in range(len(p)):
        
        term = p[r] # The coefficient
        for i in range(nx):        
            term *= xpow[i][pows[r,i]]
            
        y += term 
        
    return y 

            