"""
nitrogen.vpt
-----------------

Vibrational perturbation theory and harmonic oscillator methods

"""


import numpy as np 


def autocorr_linear(w, f, t):
    """
    Calculate the vacuum state autocorrelation function
    for propagation on a linear potential energy surface.

    Parameters
    ----------
    w : array_like
        The harmonic frequency (in energy) of each mode.
    f : array_like
        The derivative array, including at least first derivatives.
    t : array_like
        The time array.

    Returns
    -------
    C : ndarray
        The autocorrelation function, :math:`C(t)`. 
        
    Notes
    -----
    
    A linear potential is separable into 1-D components, so the
    total autocorrelation function is a product with a factor
    for each mode equal to
    
    ..  math::
        C_i(t) = \\frac{1}{(1 + i \\omega t )^{1/2}} \\exp[-(f^2 t^2/24)(6 + i \\omega t)]
    
    
    """
    
    n = len(w) # the number of modes
    
    if len(f) < n + 1:
        raise ValueError("The derivative array must contain at least first derivatives")
    
    # Initialize C(t) with the contribution from
    # the energy offset
    C = np.exp(-1j * f[0] * t) 
    
    # For each mode, calculate its contribution and multiply 
    for i in range(n):
        
        wi = w[i] 
        fi = f[i+1] 
        
        Ci = np.exp(-fi**2 * (t**2)/24  * (6 + 1j*wi *t)) / np.sqrt(1 + 1j*wi*t)
    
        C *= Ci 
    
    return C 

    
    