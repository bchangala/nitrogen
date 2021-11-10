"""
nitrogen.vpt
-----------------

Vibrational perturbation theory and harmonic oscillator methods

"""

import nitrogen.math 
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

def autocorr_quad(w, f, t, calc_log = False):
    
    """
    Calculate the vacuum state autocorrelation function 
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
    calc_log : bool, optional
        Calculate the logarithm of :math:`C(t)` instead. The default is 
        False.

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
    Instead of computing the quadratic correlator via a closed-form
    expression, this function first calculates the derivative of its
    logarithm, which is determined by the recursion coefficients
    of quadratic exponential operators. This derivative is then
    numerically integrated by cumulative Simpson's rule and
    then exponentiated. 
    
    """
    
    n = len(w)
    
    # Calculate the correlator recursion coefficients
    r,S,T = corr_quad_recursion_elements(w, f, t)
    
    # Extract the gradient and hessian 
    F,K = _partition_darray(f, n)
    f0 = f[0] # The energy offset
    
    # Calculate the ODE coefficient sum
    sumIH = 0
    for i in range(n):
        sumIH += 0.25 * ( (w[i] + K[i,i]) - (w[i] - K[i,i])*(r[:,i]**2 - T[:,i,i]))
        sumIH += (-np.sqrt(0.5)) * F[i] * r[:,i] 
        
        for j in range(i): # j < i 
            sumIH += 0.5 * K[i,j] * (r[:,i] * r[:,j] - T[:,i,j]) 
            
    g = (-1j) * sumIH # the derivative of the logarithm
    
    #
    # Integrate the logarithm via
    # Simpson's 1/3 rule, cumulatively 
    #
    logC = nitrogen.math.cumsimp(g, t)
    
    # Add the energy offset phase correction 
    logC += -1j * f0 * t 
    
    if calc_log:
        return logC 
    else:
        C = np.exp(logC)
        return C
    

def corr_quad_recursion_elements(w, f, t):
    """
    Calculate the correlation function recursion coefficients 
    for a quadratic Hamiltonian.

    Parameters
    ----------
    w : array_like
        The harmonic frequencies (in energy units) defining 
        the dimensionless normal coordinates.
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
    n = len(w) # The number of dimensions 
    t = np.array(t) # The time array 
    
    # Process effective inverse mass
    w = np.array(w)
    # W = np.diag(w) 
    rtW = np.diag(w**0.5) # Sqrt[W]
    irW = np.diag(w**-0.5) # 1/Sqrt[W]
    

    F,K = _partition_darray(f, n)
    
    # Calculate displacement vector
    d = -np.linalg.inv(K) @ F 

    # Diagonalize mass-weighted Hessian
    Ktil = rtW @ K @ rtW 
    z2,L = np.linalg.eigh(Ktil)
    
    # Calculate diagonal frequencies
    omega = np.sqrt(abs(z2))
    rtO = np.diag(np.sqrt(omega))
    irO = np.diag(1/np.sqrt(omega))
    
    # Calculate sigma for each mode
    #  1 for bounded modes
    # -i for unbounded modes
    sig = np.array([1 if z2[i] > 0 else -1j for i in range(n)])

    # Weighted transformation matrix
    R = irW @ L @ rtO
    iR = irO @ L.T @ rtW 
    Rp = iR.T 
    Rt = R.T 
    
    #########################################
    #
    # COMMENTED OUT 10/21/2021 -- PBC
    #
    # This naive inversion is numerically unstable
    # at large t for unbounded modes.
    #
    # # Calculate time-dependent factors
    # # These are all purely real
    # cos = np.zeros((n,nt))     # cos(Sigma * Omega * t)
    # sigsin = np.zeros((n,nt))  # Sigma * sin(Sigma * Omega * t)
    # isigsin = np.zeros((n,nt)) # Sigma^-1 * sin(Sigma * Omega * t)

    # for i in range(n):
    #     if z2[i] > 0: # sigma = +1
    #         cos[i] = np.cos(O[i,i] * t)
    #         sigsin[i] = np.sin(O[i,i] * t)
    #         isigsin[i] = sigsin[i] 
    #     elif z2[i] < 0: # sigma = -i 
    #         cos[i] = np.cosh(O[i,i] * t)
    #         sigsin[i] = -np.sinh(O[i,i] * t)
    #         isigsin[i] = +np.sinh(O[i,i] * t) 
    #     else:
    #         raise ValueError("z2 == 0! Bad!")
    
    # # Shape for broadcasting as diag(cos) @ M
    # cos = (cos.T).reshape((nt,n,1))
    # sigsin = (sigsin.T).reshape((nt,n,1))
    # isigsin = (isigsin.T).reshape((nt,n,1))
    
    # # Calculate the X, Y coefficient matrices 
    # # that propagate q(t) and p(t)
    
    # d = d.reshape((n,1)) 
    
    # Xq = Rp @ (cos * Rt) # cos * Rt generates (nt, n, n), which is then stack matmul'ed
    # Xp = Rp @ (isigsin * iR) 
    # X0 = d - Rp @ (cos * (Rt@d))
    
    # Yq = -R @ (sigsin * Rt) 
    # Yp = R @ (cos * iR) 
    # Y0 = R @ (sigsin * (Rt@d))
    
    # Na = 0.5 * (Xq - Yp - 1j*Xp - 1j*Yq)
    # Nap = 0.5 * (Xq + Yp + 1j*Xp - 1j*Yq)
    # N0 = np.sqrt(0.5) * (X0 - 1j*Y0)

    # # Finally, calculate the recursion matrices 
    # S = np.linalg.inv(Nap) 
    # r = S @ N0 
    # T = S @ Na 
    #
    # END COMMENT BLOCK - PBC
    ####################################

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
    ExpP = np.zeros((len(t), n, n), dtype = np.complex128)
    ExpM = np.zeros((len(t), n, n), dtype = np.complex128)
    for i in range(n): 
        ExpP[:,i,i] = np.exp(+1j * sig[i] * omega[i] * t)
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
        
     
    
    
    