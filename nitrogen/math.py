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
        The output value
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

def mpolyfit_grad(x, y, yg, deg, scale = None):
    """
    Multivariable polynomial least-squares fitting including
    gradient constraints.

    Parameters
    ----------
    x : (N,nx) or (N,) array_like
        The input coordinates.
    y : (N,) array_like
        The output value
    yg : (N,nx) array_like
        The output gradient
    deg : int
        The degree of the polynomial
    scale : (nx,) array_like, optional
        The coordinate scale. The default is 1 for each
        coordinate.

    Returns
    -------
    p : (nt,) ndarray
        The polynomial coefficients
    res : (N,nx+1) ndarray
        The residuals. The first column is the `y` residual.
        The second is the gradient residual including
        `scale`.
    
    Notes
    -----
    The polynomial is ordered using the standard 
    lexical ordering defined by the ``autodiff``
    and ``DFun`` modules.
    
    The gradient data is premultiplied by `scale`
    for each coordinate before least-squares
    optimization.

    """
    
    y = np.array(y)
    N = y.shape[0] # The number of data points
    x = np.array(x)
    x = x.reshape((N,-1)) # Force to 2-dim
    nx = x.shape[1]  # The number of variables 
    
    yg = np.array(yg) # (N,nx) 
    if yg.ndim != 2:
        raise ValueError('yg must be 2-d')
    if yg.shape[0] != N or yg.shape[1] != nx:
        raise ValueError('unexpected shape for yg')
    
    if scale is None:
        scale = [1.0 for i in range(nx)]
    scale = np.array(scale)
    if scale.ndim != 1 or len(scale) != nx:
        raise ValueError("scale must be 1-d array of length nx")
    
    # Calculate the powers
    # in standard lexical ordering
    pows = adf.idxtab(deg, nx) 
    nt = pows.shape[0] # the number of terms
    
    # Create the least-squares array
    # Initialize to all ones
    C = np.ones((nx+1,N,nt), dtype = x.dtype)
    
    # Calculate the powers of the inputs
    xpow = [ [ x[:,i] ** k for k in range(deg+1)] for i in range(nx)]
    
    for r in range(nt):
        # For each term
        pr = pows[r,:] # the powers of this term 
        #
        # Compute the zeroth derivative (value)
        for i in range(nx):        
            C[0,:,r] *= xpow[i][pr[i]]
        
        # Compute the first derivative for each coordinate k
        for k in range(nx): 
            for i in range(nx):
                if i == k:
                    #
                    # derivative of (xi)**p equals
                    # p * (xi)**(p-1)
                    #
                    # if p == 0, then the array index is -1,
                    # but that's okay because p is zero anyway
                    #
                    C[k+1,:,r] *= (pr[i] * xpow[i][pr[i]-1])
                else:
                    C[k+1,:,r] *= xpow[i][pr[i]]
    
    
    Y = np.concatenate((y.reshape((1,N)), yg.T), axis = 0).copy() # (nx+1, N)
    
    # Scale the gradient bits
    for k in range(nx):
        C[k+1] *= scale[k] 
        Y[k+1] *= scale[k] 
        
    # Reshape for matrix least-squares
    Y = Y.reshape( (-1,) )
    C = C.reshape( (-1, nt) )
    
    # Now solve the linear least-squares problem
    p,_,_,_ = np.linalg.lstsq(C, Y, rcond=None)
    
    res = Y - C @ p # residual
    
    res = np.reshape(res, (nx+1, N))
    res = res.T  # (N, nx+1)
    
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

def levi3():
    """
    Return the 3-index Levi-Civita :math:`\\epsilon_{ijk}` tensor.
    
    Returns
    -------
    eps : (3,3,3) ndarray
        The Levi-Civita tensor 
        
    """
    
    return np.array([[[ 0,  0,  0],
                      [ 0,  0,  1],
                      [ 0, -1,  0]],

                     [[ 0,  0, -1],
                      [ 0,  0,  0],
                      [ 1,  0,  0]],
        
                     [[ 0,  1,  0],
                      [-1,  0,  0],
                      [ 0,  0,  0]]])
    

def constrainedPolynomial(x, df, x0 = None):
    """
    Calculate a single-variable
    polynomial that exactly satisfies
    the value and derivatives at one or more points
    
    Parameters
    ----------
    x : (n,) array_like
        The matching positions
    df : (deriv+1, n) array_like
        The scaled derivative arrays of :math:`f(x)` at the matching positions
    x0 : scalar, optional
        The expansion point of the polynomial. If None, 
        `x0` is zero.
    
    Returns
    -------
    c : (nc,) ndarray
        The power series coefficients for the matching polynomial, 
        ``c[0] + c[1]*(x-x0) + c[2]*(x-x0)**2 + ...``, where 
        `nc` = `n`\ *(\ `deriv` +1).
        
    Notes
    -----
    The `df` derivative array contains the **scaled** derivatives,
    ``df[n]`` = :math:`f^{(n)} = \partial_x^n f / n!`.
    
    """
    
    x = np.array(x)
    df = np.array(df)
    
    if x0 is None:
        x0 = 0.0 
        
    n = len(x) # The number of points 
    deriv = df.shape[0] - 1 # The derivative order 
    
    # The number of parameters is equal to the
    # total number of boundary conditions, 
    # which equals n * (deriv + 1)
    #
    nc = n * (deriv + 1) 
    
    # We now set up the linear equation relating the 
    # polynomial expansion coefficients to the 
    # derivatives at each matching position
    
    C = np.zeros((nc,nc)) 
    b = np.zeros((nc,))
    
    nck = adf.ncktab(nc)
    
    
    for i in range(n): 
        # At matching point x[i]
        for k in range(deriv+1): 
            # The k**th scaled derivative at x[i]
            
            idx = i*(deriv+1) + k # The constraint index
            
            b[i*(deriv+1) + k] = df[k,i]  # The constraint value 
            
            for m in range(nc):
                # The (x-x0)**m term 
                #
                # The k**th scaled deriative of (x-x0)**m 
                # is (m choose k) * (x-x0)**(m-k) for k <= m 
                
                if k > m:
                    # The derivative is zero
                    C[idx, m] = 0.0 
                else:
                    C[idx, m] = nck[m,k] * (x[i] - x0)**(m-k)
                    
                    
    # C is now complete.
    # Solve the linear equation C @ c = b 
    c = np.linalg.solve(C, b) 
    
    return c 

def constrainedFourier(x, df, period = None):
    """
    Calculate a single-variable
    Fourier series that exactly satisfies
    the value and derivatives at one or more points
    
    Parameters
    ----------
    x : (n,) array_like
        The matching positions
    df : (deriv+1, n) array_like
        The scaled derivative arrays of :math:`f(x)` at the matching positions
    period : float, optional
        The period of the coordinate `x`. If None, :math:`2\\pi` is assumed.
    
    Returns
    -------
    c : (nc,) ndarray
        The Fourier series coefficients. 
        
    Notes
    -----
    The `df` derivative array contains the **scaled** derivatives,
    ``df[n]`` = :math:`f^{(n)} = \partial_x^n f / n!`.
    
    The expansions coefficients are defined as 
    
    ..  math::
        
        f(x) = c_0 + c_1 \\sin \\sigma x + c_2 \\cos \\sigma  x +
        c_3 \\sin 2 \\sigma  x + c_4 \\cos 2 \\sigma  x + \\cdots
        
    where :math:`\\sigma = 2\\pi/`\ `period`.
    
    """
    
    x = np.array(x)
    df = np.array(df)
    
    if period is None:
        scale = 1.0 
    else:
        scale = (2*np.pi) / period 
        
        
    n = len(x) # The number of points 
    deriv = df.shape[0] - 1 # The derivative order 
    
    # We assume a fully linearly independent
    # set of constraints
    #
    # The number of parameters is equal to the
    # total number of boundary conditions, 
    # which equals n * (deriv + 1)
    #
    nc = n * (deriv + 1) 
    
    # We now set up the linear equation relating the 
    # polynomial expansion coefficients to the 
    # derivatives at each matching position
    
    C = np.zeros((nc,nc)) 
    b = np.zeros((nc,))
    
    
    for i in range(n): 
        # At matching point x[i]
        
        for k in range(deriv+1): 
            # The k**th scaled derivative at x[i]
            
            idx = i*(deriv+1) + k # The constraint index
            
            b[i*(deriv+1) + k] = df[k,i]  # The constraint value 
            
            for m in range(nc):
                # The c_m expansion term:
                # 
                # if m is even, then cos(w * scale * x) with w = m // 2
                #    m is odd,  then sin(w * scale * x) with w = (m+1) // 2
                #
                # Calculate the k**th scaled derivative of the term
                # at x[i]
                #
                if m == 0: 
                    # constant term
                    C[idx, m] = (1.0 if k == 0 else 0.0) 
                    
                elif (m % 2) == 0:
                    # cosine
                    ws = (m // 2) * scale
                    if k % 4 == 0:
                        C[idx, m] = (ws)**k * np.cos(ws * x[i]) / np.math.factorial(k)
                    elif k % 4 == 1:
                        C[idx, m] = (ws)**k * (-np.sin(ws * x[i])) / np.math.factorial(k)
                    elif k % 4 == 2:
                        C[idx, m] = (ws)**k * (-np.cos(ws * x[i])) / np.math.factorial(k)
                    else: #k % 4 == 3
                        C[idx, m] = (ws)**k * np.sin(ws * x[i]) / np.math.factorial(k)
                    
                else:
                    # sine 
                    ws = ( (m+1) // 2 ) * scale
                    if k % 4 == 0:
                        C[idx, m] = (ws)**k * np.sin(ws * x[i]) / np.math.factorial(k)
                    elif k % 4 == 1:
                        C[idx, m] = (ws)**k * np.cos(ws * x[i]) / np.math.factorial(k)
                    elif k % 4 == 2:
                        C[idx, m] = (ws)**k * (-np.sin(ws * x[i])) / np.math.factorial(k)
                    else: #k % 4 == 3
                        C[idx, m] = (ws)**k * (-np.cos(ws * x[i])) / np.math.factorial(k)
            
    # C is now complete.
    # Solve the linear equation C @ c = b 
    c = np.linalg.solve(C, b) 
    
    return c 
