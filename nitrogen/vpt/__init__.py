"""
nitrogen.vpt
-----------------

Vibrational perturbation theory and harmonic oscillator methods

"""

import nitrogen.math 
from nitrogen.autodiff.forward import idxtab as adf_idxtab
from nitrogen.autodiff.forward import idxpos as adf_idxpos 
from nitrogen.autodiff.forward import ncktab as adf_ncktab 
import numpy as np 


def autocorr_linear(w, f, t):
    """
    Calculate the vacuum state autocorrelation function
    for propagation on a linear potential energy surface.

    Parameters
    ----------
    w : array_like
        The harmonic frequency (in energy units) of each mode.
    f : array_like
        The derivative array, including at least first derivatives.
    t : array_like
        The time array, in units where :math:`\\hbar = 1`. (Alternatively,
        the `t` array can be identified with :math:`t/\\hbar`.)

    Returns
    -------
    C : ndarray
        The autocorrelation function, :math:`C(t)`. 
        
    See also
    --------
    ~nitrogen.math.spech_fft : Calculate the spectrum of an autocorrelation function
        
    Notes
    -----
    
    A linear potential is separable into 1-D components, so the
    total autocorrelation function is a product with a factor
    for each mode equal to
    
    ..  math::
        C_i(t) = \\frac{1}{(1 + i \\omega t )^{1/2}} \\exp[-(f^2 t^2/24)(6 + i \\omega t)]
    
    times a final phase factor of :math:`\\exp[-i f_0 t]`, where :math:`f_0` is the
    energy offset equal to ``f[0]``.
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

def autocorr_quad(W, f, t, method = 'direct'):
    
    """
    Calculate the vacuum state autocorrelation function 
    for propagation on a quadratic potential energy surface.
    
    
    Parameters
    ----------
    W : (n,) or (n,n) array_like
        The harmonic frequency (in energy units) of each mode, or the 
        inverse mass tensor. 
    f : array_like
        The derivative array, including up to at least second derivatives.
    t : array_like
        The time array, in units where :math:`\\hbar = 1`. (Alternatively,
        the `t` array can be identified with :math:`t/\\hbar`.)
    method : {'direct','integral','integral_log'}
        The calculation method. See Notes

    Returns
    -------
    C : ndarray
        The autocorrelation function, :math:`C(t)`. 
        
    See also
    --------
    corr_quad_recursion_elements : Calculate quadratic correlator recursion coefficients
    ~nitrogen.math.spech_fft : Calculate the spectrum of an autocorrelation function
        
    Notes
    -----
    For `method` = 'direct', a direct expression based on 
    a discontinuity-free BCH disentangling formula is used.
    
    For `method` = 'integral', an alternative method is
    used to first calculate the logarithmic derivative
    of :math:`C(t)`. This is numerically integrated
    by a cumulative version of Simpson's rule and then 
    exponentiated.
    
    For `method` = 'integral_log', the integrated logarithm
    is returned directly, without exponentiation. That is,
    the branch-cut discontinuity-free logarithm of :math:`C(t)` 
    is returned.
    
    For the integral methods, a sufficiently small time-step
    in the `t` array is required for accurate results. The direct
    method does not rely in numerical integration.

    """
    
    W = np.array(W)
    if W.ndim == 1:
        W = np.diag(W) 
    
    n = W.shape[0] 
    
    # Extract the gradient and hessian 
    F,K = _partition_darray(f, n)
    h0 = f[0] # The energy offset

    t = np.array(t)
    if t.ndim != 1:
        raise ValueError('t must be 1-dimensional')
    
    
    if method == 'integral' or method == 'integral_log':
        #
        # Calculate the correlation function by
        # integration of its logarithmic derivative
        #
        # Check for a valid time vector
        if t[0] != 0:
            raise ValueError('t[0] must be zero for integral methods')
        if np.any( np.abs(np.diff(t) - (t[1]-t[0])) > 1e-8):
            raise ValueError('Time vector must be uniformly spaced.')
            
        #
        # Calculate the correlator recursion coefficients
        r,S,T = corr_quad_recursion_elements(W, f, t)
        
        # Calculate the ODE coefficient sum
        sumIH = 0
        for i in range(n):
            sumIH += 0.25 * ( (W[i,i] + K[i,i]) - (W[i,i] - K[i,i])*(r[:,i]**2 - T[:,i,i]))
            sumIH += (-np.sqrt(0.5)) * F[i] * r[:,i] 
            
            for j in range(i): # j < i 
                sumIH += 0.5 * (K[i,j] - W[i,j]) * (r[:,i] * r[:,j] - T[:,i,j]) 
                
        g = (-1j) * sumIH # the derivative of the logarithm
        #
        # C'(t) = g * C(t)
        #
        # --> C(t) = exp[ integral of g(t) ]
        #
        # Integrate the logarithm via
        # Simpson's 1/3 rule, cumulatively 
        #
        logC = nitrogen.math.cumsimp(g, t)
        
        # Add the energy offset phase correction 
        logC += -1j * h0 * t 
        
        if method == 'integral_log':
            # Return the continuous logarithm of C
            return logC 
        else:
            # Return C
            C = np.exp(logC)
            return C
        
    elif method == 'direct':
        #
        # Calculate the correlation function by
        # the direct method
        #
        
        # First, calculate the propagation normal
        # modes
        
        w,wU = np.linalg.eigh(W)
        #
        # W = wU @ w @ wU.T 
        #
        rtW = wU @ np.diag(np.sqrt(w)) @ wU.T
        irW = wU @ np.diag(1/np.sqrt(w)) @ wU.T

        z2,L = np.linalg.eigh(rtW @ K @ rtW)
        # Force L to have positive determinant! 
        if np.linalg.det(L) < 0:
            L[:,0] *= -1 
        
        omega = np.sqrt(np.abs(z2))
        sigma = np.array([1 if z2[i] > 0 else -1j for i in range(n)])
        
       
        rtSO = np.diag(np.sqrt(sigma * omega))
        irSO = np.diag(1/np.sqrt(sigma*omega))
        LamP = irW @ L @ rtSO + rtW @ L @ irSO 
        LamM = irW @ L @ rtSO - rtW @ L @ irSO 
        iLamP = np.linalg.inv(LamP)
        
        C = np.zeros_like(t, dtype = np.complex128)
        
        def eta(x):
            #
            # eta(x) = (e^x - 1) / x
            #
            
            result_small = 1.0 + x/2 + x**2/6 + x**3/24 + x**4/120 + x**5/720 + x**6/5040
            result_big = np.expm1(x) / (x + 1e-20) 
            
            result = np.choose(abs(x) > 1e-2,
                               [result_small, result_big])
            
            return result 
        
        def zeta(x):
            #
            # zeta(x) = (e^x - x - 1) / x**2
            #
            
            result_small = 1/2 + x/6 + x**2/24 + x**3/120 + x**4/720 + x**5/5040 + x**6/40320
            result_big = (np.expm1(x) - x) / (x + 1e-20)**2 
            
            result = np.choose(abs(x) > 1e-2,
                               [result_small, result_big])
            
            return result 
        
        # Force all time values to be non-negative.
        # Afterward, negative time can be evaluated
        # via the hermiticity of C(t)
        for i in range(len(t)):
            
            tp = abs(t[i]) # The current time value
            
            # The exp^- diagonal 
            em = np.diag(np.exp(-1j * tp * sigma*omega))
            
            #
            # Calculate det(exp[A'])**1/2:
            #
            # A factoring and eigendecomposition
            # procedure ensures there are no
            # branch-cut discontinuities
            #
            quad_term = np.exp(-1j * tp * sum(sigma*omega) / 2)
            quad_term *= np.linalg.det(LamP / 2) ** -1 
            
            
            M = iLamP.T @ em @ iLamP @ LamM @ em @ LamM.T 
            evs = np.linalg.eigvals(M)
            
            for a in evs:
                quad_term *= np.sqrt(1 - a)**-1
        
            #
            # Calculate the gradient contributions
            #
            hp = -1j*tp*h0 # Trivial phase contribution
            
            # The eta^- and zeta^- diagonal matrices
            etam = np.diag(eta(-1j*tp*sigma*omega))
            zetam = np.diag(zeta(-1j*tp*sigma*omega))
        
            # First term
            temp1 = iLamP @ LamM @ em @ LamM.T @ iLamP.T 
            G1 = -etam @ temp1 @ np.linalg.inv(np.eye(n) - em@temp1) @ etam 
            
            # Second term
            temp2 = iLamP.T @ em @ iLamP @ LamM @ em @ LamM.T
            temp3 = etam @ LamM.T @ np.linalg.inv(np.eye(n) - temp2) @ iLamP.T @ etam 
            G2 = -(temp3 + temp3.T)
            
            # Third term
            G3 = -2*zetam - etam @ LamM.T @ \
                np.linalg.inv(np.eye(n) - temp2) @ \
                iLamP.T @ em @ iLamP @ LamM @ etam 
            
            Gamma = G1 + G2 + G3
            
            Fbar = (LamP - LamM).T @ F 
            hp += (tp/4)**2 * np.dot(Fbar, Gamma @ Fbar) 
            
            C[i] = quad_term * np.exp(hp)
            
            # For negative time values, correct
            # for the complex conjugate 
            if t[i] < 0:
                C[i] = np.conjugate(C[i])
                
        return C 
        
    else:
        raise ValueError('Invalid method option')

def autocorr_quad_finiteT(w, f, t, beta, method = 'direct'):
    
    """
    Calculate the thermal correlation function 
    for propagation on a quadratic potential energy surface.
    
    
    Parameters
    ----------
    w : array_like
        The harmonic frequency (in energy units) of each mode.
    f : array_like
        The derivative array, including up to at least second derivatives.
    t : array_like
        The time array, in units where :math:`\\hbar = 1`. (Alternatively,
        the `t` array can be identified with :math:`t/\\hbar`.)
    beta : float 
        The value of :math:`\\beta = 1/kT` in inverse energy units.
    method : {'direct'}
        The calculation method. See Notes.

    Returns
    -------
    C : ndarray
        The thermal autocorrelation function, :math:`C(t)`. 
        
    See also
    --------
    autocorr_quad : Calculate the zero-temperature correlation function
    ~nitrogen.math.spech_fft : Calculate the spectrum of an autocorrelation function
        
    Notes
    -----
    For `method` = 'direct' (currently the only method),
    a direct evaluation of the thermal trace
    
    .. math::
       
       C(t, \\beta) = \\frac{1}{Z_0(\\beta)} \\text{Tr}\\left[ e^{(+it-\\beta)H_0} e^{-it H_1} \\right]
        
    is performed using a stable, discontinuity-free method.
    
    """    
    
    n = len(w)
    
    # Extract the gradient and hessian 
    F,K = _partition_darray(f, n)
    h0 = f[0] # The energy offset

    t = np.array(t)
    if t.ndim != 1:
        raise ValueError('t must be 1-dimensional')
    
    
    if method == 'direct':
        #
        # Calculate the thermal correlation function by
        # the direct trace approach.
        
        # First, calculate the propagation normal
        # modes
        rtW = np.diag(np.sqrt(w))
        irW = np.diag(1/np.sqrt(w))

        z2,L = np.linalg.eigh(rtW @ K @ rtW)
        # Force L to have positive determinant! 
        if np.linalg.det(L) < 0:
            L[:,0] *= -1 
        
        omega = np.sqrt(np.abs(z2))
        sigma = np.array([1 if z2[i] > 0 else -1j for i in range(n)])
        
       
        rtSO = np.diag(np.sqrt(sigma * omega))
        irSO = np.diag(1/np.sqrt(sigma*omega))
        LamP = irW @ L @ rtSO + rtW @ L @ irSO 
        LamM = irW @ L @ rtSO - rtW @ L @ irSO 
        iLamP = np.linalg.inv(LamP)
        
        Fbar = (LamP.T - LamM.T) @ F
        
        # Calculate the ground state partition function
        # referenced to the ground state origin
        Z0p = 1 
        for i in range(n):
            Z0p *= 1/(-np.expm1(-w[i]*beta))
        
        C = np.zeros_like(t, dtype = np.complex128)
        
        def eta(x):
            #
            # eta(x) = (e^x - 1) / x
            #
            
            result_small = 1.0 + x/2 + x**2/6 + x**3/24 + x**4/120 + x**5/720 + x**6/5040
            result_big = np.expm1(x) / (x + 1e-20) 
            
            result = np.choose(abs(x) > 1e-2,
                               [result_small, result_big])
            
            return result 
        
        def zeta(x):
            #
            # zeta(x) = (e^x - x - 1) / x**2
            #
            
            result_small = 1/2 + x/6 + x**2/24 + x**3/120 + x**4/720 + x**5/5040 + x**6/40320
            result_big = (np.expm1(x) - x) / (x + 1e-20)**2 
            
            result = np.choose(abs(x) > 1e-2,
                               [result_small, result_big])
            
            return result 
        
        # Force all time values to be non-negative.
        # Afterward, negative time can be evaluated
        # via the hermiticity of C(t)
        for i in range(len(t)):
            
            tp = abs(t[i]) # The current time value
            tau = -1j * tp + beta
            
            # The exp^- diagonal 
            em = np.diag(np.exp(-1j * tp * sigma*omega))
            etam = np.diag(eta(-1j * tp * sigma * omega)) # eta^-
            zetam = np.diag(zeta(-1j * tp * sigma * omega)) # zeta^-
            xim = np.diag(np.exp(-tau * w)) # xi^-1
            xim2 = np.diag(np.exp(-tau * w/2)) # xi^-1/2
            one = np.eye(n) 
            
            ########################################
            # Calculate the quadratic contribution
            #
            # Factor 1
            irt_detMA = np.linalg.det(LamP/2)**-1 * np.exp(-1j * tp * sum(sigma*omega)/2)
            K1 = one - iLamP.T @ em @ iLamP @ (LamM @ em @ LamM.T + 4 * xim)
            iK1 = np.linalg.inv(K1)
            for a in np.linalg.eigvals(K1):
                irt_detMA *= 1/np.sqrt(a) 
            
            # Factor 2
            T1 = LamM @ iLamP @ (LamM @ em @ LamM.T + 4 * xim) @ iK1 @ iLamP.T @ LamM.T 
            T2 = LamP @ em @ (LamM.T @ iK1 @ iLamP.T @ em @ iLamP @ LamM @ em + one) @ LamP.T
            t3 = -LamP @ em @ LamM.T @ iK1 @ iLamP.T @ LamM.T 
            T3 = t3 + t3.T 
            MD = np.eye(n) - 0.25 * xim2 @ (T1 + T2 + T3) @ xim2
            
            irt_detMD = 1.0 
            for b in np.linalg.eigvals(MD):
                irt_detMD *= 1/np.sqrt(b)
           
            Tr = irt_detMD * irt_detMA 
            C[i] = Tr / (np.exp(-1j*tp*sum(w)/2) * Z0p) 
            #
            #########################################
            
            ###########################################3
            # Calculate the gradient contribution
            #
            K = LamM @ em @ LamM.T + 4 * xim # The commonly used expression
            KT = iLamP @ K @ iLamP.T # The transformed expression, also commonly used
            iDbar = np.linalg.inv(one - iLamP.T @ em @ iLamP @ K) @ iLamP.T @ em @ iLamP
            
            iDL = np.linalg.inv(one - em @ KT) # em on left side
            iDR = np.linalg.inv(one - KT @ em) # em on right side
            
            A1 = -etam @ KT @ iDL @ etam 
            A2 = -2*zetam - etam @ LamM.T @ iDbar @ LamM @ etam 
            a3 = -etam @ iDR @ iLamP @ LamM @ etam 
            A3 = a3 + a3.T
            A = A1 + A2 + A3 
            
            B1 = -LamM @ iDR @ KT @ etam 
            B2 = LamP @ (one + em @ LamM.T @ iDbar @ LamM) @ etam 
            B3 = -LamM @ iDR @ iLamP @ LamM @ etam 
            B4 = LamP @ em @ LamM.T @ iLamP.T @ iDL @ etam 
            B = B1 + B2 + B3 + B4 
            
            D1 = xim2 @ LamM @ KT @ iDL @ LamM.T @ xim2 
            D2 = xim2 @ LamP @ em @ (one + LamM.T @ iDbar @ LamM @ em) @ LamP.T @ xim2 - 4*one
            d3 = -xim2 @ LamM @ iDR @ iLamP @ LamM @ em @ LamP.T @ xim2 
            D3 = d3 + d3.T 
            D = D1 + D2 + D3 
            
            Gamma = A + B.T @ xim2 @ np.linalg.inv(D) @ xim2 @ B 
            
            hp = -1j * tp * h0 # Trivial offset phase
            hp += (tp**2 / 16) * Fbar @ ( Gamma @ Fbar ) # Gradient contribution
            
            C[i] *= np.exp(hp)
            
            # For negative time values, correct
            # for the complex conjugate 
            if t[i] < 0:
                C[i] = np.conjugate(C[i])
                
        return C 
        
    else:
        raise ValueError('Invalid method option')
   
def corr_quad_recursion_elements(W, f, t):
    """
    Calculate the correlation function recursion coefficients 
    for a quadratic Hamiltonian.

    Parameters
    ----------
    W : (n,) or (n,n) array_like
        If a 1-d vector, the harmonic frequencies (in energy units) defining 
        the dimensionless normal coordinates. If a 2-d array, then 
        the effective inverse mass tensor.
    f : array_like
        The derivative array of the propagating surface. This must
        have at least second derivatives. (Note the factorial scaling
        rules of derivative arrays.)
    t : array_like
        The scaled time array, :math:`t/\\hbar`. 

    Returns
    -------
    r, S, T : ndarray
        The coefficients arrays
        
    Notes
    -----
    Following Ref. [FT1989]_ , the correlation functions obey the 
    recursion relations given by 
    
    ..  math::
        
        \\langle m_i + 1 \\vert \\vert \\cdots \\rangle &= \\frac{1}{\\sqrt{m_i+1}} 
        \\left[ -r_i \\langle \\cdots \\vert \\vert \\cdots \\rangle + \\sum_j S_{ij} 
               \\sqrt{n_j} \\langle \\cdots \\vert \\vert n_j - 1 \\rangle - T_{ij} 
               \\sqrt{m_j} \\langle m_j - 1 \\vert \\vert \\cdots \\rangle \\right]
        
        \\langle \\cdots \\vert \\vert n_i + 1 \\rangle &= \\frac{1}{\\sqrt{n_i+1}} 
        \\left[ -r_i \\langle \\cdots \\vert\\vert\\cdots\\rangle + \\sum_j -T_{ij} 
               \\sqrt{n_j} \\langle \\cdots \\vert \\vert n_j -1 \\rangle + S_{ij} 
               \\sqrt{m_j} \\langle m_j - 1 \\vert \\vert \\cdots \\rangle \\right]
        
    where some simplications from the most general expressions have been applied
    given that the propagation is unitary. The matrices :math:`S` and :math:`T`
    are symmetric.
    
    
    References
    ----------
    .. [FT1989] F. M. Fernandez and R. H. Tipping, "Multidimensional harmonic oscillator matrix elements".
       J. Chem. Phys., 91, 5505 (1989).
       https://doi.org/10.1063/1.457553

    """
    
    # Process effective inverse mass
    #
    W = np.array(W)
    if W.ndim == 1:
        w = W.copy() 
        n = len(w)
        wU = np.eye(n)
    elif W.ndim == 2:
        n = W.shape[0] 
        w,wU = np.linalg.eigh(W)
    else:
        raise ValueError("W must be a 1- or 2-d array")
    
    #
    # W = wU @ w @ wU.T 
    #
    rtW = wU @ np.diag(w**0.5) @ wU.T # Sqrt[W]
    
    
    t = np.array(t) # The time array 
   
    F,K = _partition_darray(f, n)
    
    # Calculate displacement vector
    d = -np.linalg.inv(K) @ F 

    # Diagonalize mass-weighted Hessian
    Ktil = rtW @ K @ rtW 
    z2,L = np.linalg.eigh(Ktil)
    
    # Calculate diagonal frequencies
    omega = np.sqrt(abs(z2))
    # Calculate sigma for each mode
    #  1 for bounded modes
    # -i for unbounded modes
    sig = np.array([1 if z2[i] > 0 else -1j for i in range(n)])

    r, S, T = _corr_quad_recursion_elements_inner(w, wU, d, omega, sig, L, t)
    
    return r, S, T 

def _corr_quad_recursion_elements_inner(w, wU, d, omega, sig, L, t):
    """
    w : eigenvalues of W (usually the initial state frequencies)
    wU: eigenvectors of W (usually identity)
    d : quadratic displacement vector (-hessian @ gradient)
    omega : propagation frequencies
    sig : propagation sigma (1 or -1j)
    L : propagation L matrix 
    t : time vector 
    """
    
    #
    # W = wU @ diag(w) @ wU.T
    
    n = len(w) 
    
    rtW = wU @ np.diag(w**0.5) @ wU.T # Sqrt[W]
    irW = wU @ np.diag(w**-0.5) @ wU.T # 1/Sqrt[W]
    rtO = np.diag(np.sqrt(omega))
    irO = np.diag(1/np.sqrt(omega))
    
    # Weighted transformation matrix
    R = irW @ L @ rtO
    iR = irO @ L.T @ rtW 
    Rp = iR.T 
    Rt = R.T 
    
    # 
    # Compute the diagonal matrices
    # Sigma**1/2 and Sigma**-1/2
    #
    # in each case, we take the principal sqrt
    # (i.e. sqrt(-i) has positive real and negative imaginary parts)
    #
    rtSig = np.diag(np.sqrt(sig))
    irSig = np.diag(1/np.sqrt(sig))
    
    #
    # Calculate the Lambda matrices
    # 
    # Lambda_+/- =  R @ Sigma^1/2  +/-  R' @ Sigma^-1/2
    LamP = R @ rtSig + Rp @ irSig
    LamM = R @ rtSig - Rp @ irSig
    #
    # Calculate the inverse of Lambda_+
    # which always exists
    iLamP = np.linalg.inv(LamP)
    
    #
    # Calculate the exponential matrices
    #
    #ExpP = np.zeros((len(t), n, n), dtype = np.complex128)
    ExpM = np.zeros((len(t), n, n), dtype = np.complex128)
    for i in range(n): 
    #   ExpP[:,i,i] = np.exp(+1j * sig[i] * omega[i] * t)
        ExpM[:,i,i] = np.exp(-1j * sig[i] * omega[i] * t)
    
    #
    # Let A = Lambda_+ @ Exp[+] @ Lambda_+^T
    # 
    # Calculate its inverse  
    # A^-1 = Lambda_+' @ Exp[-] @ Lambda_+^-1
    #
    iA = iLamP.T @ ExpM @ iLamP
    
    #
    # Calculate the quantity
    # Exp[-] * (1 - Lambda_-^T @ A^-1 @ Lambda_- @ Exp[-])^-1
    #
    iC = ExpM @ np.linalg.inv(np.eye(n) - LamM.T @ iA @ LamM @ ExpM)
    
    # Now calculate
    # S = (Na')^-1
    # via the matrix inversion lemma
    #
    S = 4 * (iA + iA @ (LamM @ iC @ LamM.T) @ iA )
    
    # To calculate T and r
    # we need the rearranged Exp[+] terms
    #
    # (Na')^-1 @ Lambda_+ @ Exp[+]
    #
    iNapLamPeP = 4 * np.linalg.inv(np.eye(n) - iA @ (LamM @ ExpM @ LamM.T)) @ iLamP.T 
    
    # r = (Na')^-1 @ N0
    r1 = S @ (0.5 * LamM @ ExpM @ (rtSig @ Rt) + np.eye(n) )
    r2 = -0.5 * iNapLamPeP @ (rtSig @ Rt)
    r = np.sqrt(0.5) * (r1 + r2) @ d 
    r = r.reshape((-1,n))
    
    # T = (Na')^-1 @ Na 
    T1 = -0.25 * S @ LamM @ ExpM @ LamP.T 
    T2 = 0.25 * iNapLamPeP @ LamM.T
    T = T1 + T2 
    
    return r, S, T 
    

def corr_quad_ratio_table(r, S, T, nmax):
    """
    Calculate correlation function ratios using
    the quadratic recursion elements.

    Parameters
    ----------
    r, S, T : ndarray
        Quadratic recursion coefficients.
    nmax : int
        The maximum total quantum number.

    Returns
    -------
    Imn : (...,ns,ns) ndarray
        The recursion coefficients. `ns` is the number of
        states.
    qns : (ns, n) ndarray
        The quantum number table for `n` modes.

    See Also
    --------
    corr_quad_recursion_elements : Calculate the recursion coefficients.

    """
    if nmax < 0:
        raise ValueError('nmax must be a non-negative integer')
    nmodes = r.shape[-1] # The number of modes 
    
    # Calculate the list of quantum numbers
    # for states to include in the table
    qns = adf_idxtab(nmax, nmodes) 
    ns = qns.shape[0] # The number of states in the table
    
    nck = adf_ncktab(nmodes+nmax, min(nmodes,nmax)) # A binomial coefficient table
    
    base_shape = r.shape[:-1]
    Imn = np.zeros(base_shape + (ns,ns), dtype = np.complex128)
    
    ###########################
    # First, calculate the m,0 and 0,n elements
    # on the edges of the table
    #
    # The 0,0 element is always unity.
    if nmax >= 0:
        Imn[...,0,0] = 1.0 
    # Continue with higher states
    for M in range(1,ns):
        mp1 = qns[M] # < ... m+1 ... |
        #
        # Identify the first non-zero quantum number's position
        i = np.nonzero(mp1)[0][0]
        m = mp1.copy()
        m[i] -= 1 
        # m is quantum number vector of < m | 0 >
        
        ratio = 0 
        #
        # -r_i < m | 0 >  term
        ratio += -r[...,i] * Imn[...,adf_idxpos(m,nck), 0]
        
        for j in range(nmodes):
            if m[j] > 0:
                mm1 = m.copy()
                mm1[j] -= 1 
                # add -T_ij sqrt(m_j) < ... m_j-1 ... | 0 > term
                ratio += -T[...,i,j] * np.sqrt(m[j]) * Imn[...,adf_idxpos(mm1,nck),0]
        
        #
        #
        Imn[...,M,0] = ratio / np.sqrt(m[i]+1)
        #
        # Table is symmetric
        Imn[...,0,M] = Imn[...,M,0] 
    
    #
    # Now compute interior entries in
    # the table
    #   
    #   0XXXXXXXX
    #   Xxxx|
    #   Xxxx|
    #   XxxxV
    #   X-->.
    #
    # The block of the table with column and rows
    # less than the current position will always
    # be calculated already. Once the M = N
    # diagonal position is reached, this is still
    # true because the current column will be
    # complete at that point. 
    #
    for M in range(1,ns):
        mp1 = qns[M]
        i = np.nonzero(mp1)[0][0]
        m = mp1.copy()
        m[i] -= 1
        
        m_pos = adf_idxpos(m,nck)
        
        for N in range(1,M+1):
            # Calculate
            # < m | n >
            # Both m and n are non-zero 
            n = qns[N]
            n_pos = N
            
            #
            ratio = 0 
            #
            # -r term
            ratio += -r[...,i] * Imn[...,m_pos,n_pos]
            #
            # 
            for j in range(nmodes):
                # 
                # S and T terms
                if n[j] > 0:
                    nm1 = n.copy()
                    nm1[j] -= 1 
                    nm1_pos = adf_idxpos(nm1,nck) # position of |...n_j - 1 ...>
                    ratio += S[...,i,j] * np.sqrt(n[j]) * Imn[...,m_pos,nm1_pos]
                
                if m[j] > 0:
                    mm1 = m.copy()
                    mm1[j] -= 1 
                    mm1_pos = adf_idxpos(mm1,nck) # position of < ... m_j - 1 |
                    ratio += -T[...,i,j] * np.sqrt(m[j]) * Imn[...,mm1_pos, n_pos]
            
            Imn[...,M,N] = ratio / np.sqrt(m[i] + 1)
            # Apply symmetry for upper triangle
            Imn[...,N,M] = Imn[...,M,N]
     
    #
    # The table is now completely filled
    #
    
    return Imn, qns
    
                            
    
def _partition_darray(f,n):
    """
    Extract the gradient and hessian from
    an adarray-style derivative array
    
    Parameters
    ----------
    f : derivative array
    n : the number of coordinates
    
    Returns
    -------
    F : (n,) array
        The gradient
    K : (n,n) array
        The symmetric hessian
    """
    
    F = np.zeros((n,), dtype = f.dtype)  # The gradient vector
    K = np.zeros((n,n), dtype = f.dtype) # The hessian matrix
    
    if len(f) < ((n+2)*(n+1)) // 2:
        raise ValueError("derivative array must contain at least second derivatives")
    
    idx = n + 1
    for i in range(n):
        F[i] = f[i+1]
        for j in range(i,n):
            K[i,j] = f[idx]
            K[j,i] = K[i,j]
            
            if i == j:
                K[i,j] *= 2.0  # Account for permutation factorial
                
            idx += 1 
    
    return F, K 
    
def calc_rectilinear_modes(hes, mass, hbar = None, norm = 'dimensionless'):
    """
    Calculate the rectilinear normal modes
    and energies
    
    Parameters
    ----------
    hes : array_like
        The (3*N,3*N) Cartesian Hessian matrix.
    mass : array_like
        The N masses
    hbar : float, optional
        The value of :math:`\\hbar`. If None (default),
        NITROGEN units will be assumed.
    norm : {'dimensionless', 'mass-weighted'}
        The normalization convention of the displacement
        vectors. 
        
    Returns
    -------
    w : (N,) ndarray
        The harmonic frequencies, in energy units.
    R : (3*N,3*N) ndarray
        Each column of `R` is the displacement
        vector for the corresponding normal mode. 
        
    Notes
    -----
    For norm = 'dimensionless', the displacement vectors
    are those of the non-mass-weighted Cartesian coordinates
    with respect to dimensionless, normalized coordinates.
    In these coordinates, the potential energy surface is
    
    ..  math::
        V = \\sum_i \\frac{1}{2} \\omega_i q_i^2
        
    For norm = 'mass-weighted', the displacement vectors
    are those for the mass-weighted Cartesian coordinates
    and equal the eigenvectors of the mass-weighted Hessian, i.e.
    the traditional :math:`\\mathbf{L}` array.
    """
    
    hes = np.array(hes)
    if hes.ndim != 2 or hes.shape[0] != hes.shape[1]:
        raise ValueError("hes has an unexpected shape")
    if hes.shape[0] % 3 != 0:
        raise ValueError("the shape of hes must be a multiple of 3")
        
    N = hes.shape[0] // 3
    if N < 1:
        raise ValueError("there must be at least 1 atom")
    
    m = np.repeat(mass, [3]*N)
    iMrt = np.diag(m**-0.5)
    
    # Mass-scale the Cartesian hessian
    H = iMrt @ hes @ iMrt 
    
    # Diagonalize the mass-weighted hessian
    lam,L = np.linalg.eigh(H) 
    #
    # The eigenvalues equal the square of the angular frequency
    #
    
    if hbar is None:
        hbar = nitrogen.constants.hbar 

    # Calculate the harmonic energies    
    w = hbar * np.sqrt(abs(lam))

    # Calculate the Cartesian displacements
    # for the dimensionless normal coordinates
    #
    T = iMrt @ L
    for i in range(9):
        v = T[:,i] 
        a = v.T @ hes @ v 
        T[:,i] *= np.sqrt(abs(w[i] / a))
    
    if norm == 'dimensionless':
        return w, T 
    elif norm == 'mass-weighted':
        return w, L 
    else:
        raise ValueError('unexpected norm option')
        
     
    
    
    