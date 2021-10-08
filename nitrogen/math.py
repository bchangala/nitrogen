"""
nitrogen.math
-----------------

General mathematical tools and functions

"""

import numpy as np 

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
    
    The returned spectrum is calculated by a Fourier transforms as
    
    ..  math::
        
        g(\\omega) = \\frac{1}{2\\pi} \\int_{-\\infty}^{\\infty} dt e^{i \\omega t} C(t)
    
    
    This normalization means that the angular frequency integral of 
    :math:`g(\\omega)` equals :math:`C(0)`, which is typically unity.
    
    
    The calculated spectrum may be over-sampled by zero-padding the 
    autocorrelation function. This sampling factor is controlled by
    `sample_factor`. 
    
    A Gaussian window function is also applied to the autocorrelation 
    function, of the form :math:`C(t) \\exp(-a (t/T)^2)`, where 
    :math:`T` is length of the correlation function (before any padding).
    The damping factor :math:`a` is controlled by the `damping` keyword
    
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