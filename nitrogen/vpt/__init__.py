"""
nitrogen.vpt
------------

Vibrational perturbation theory and harmonic oscillator methods

"""



import numpy as np 
import nitrogen.constants

from . import ho_core
from .ho_core import *  # Import the core namespace

from . import td_core
from .td_core import *  # Import the core namespace

from . import cfourvib # CFOUR interface routines and parsers

__all__ = ['calc_rectilinear_modes']
__all__ += ho_core.__all__
__all__ += td_core.__all__


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
    norm : {'dimensionless', 'mass-weighted','both'}
        The normalization convention of the displacement
        vectors. 
        
    Returns
    -------
    w : (N,) ndarray
        The harmonic frequencies, in energy units.
    R : (3*N,3*N) ndarray
        Each column of `R` is the displacement
        vector for the corresponding normal mode. 
        If `norm` is ''both'', then T and L are returned in that order.
        
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
    for i in range(3*N):
        v = T[:,i] 
        a = v.T @ hes @ v 
        T[:,i] *= np.sqrt(abs(w[i] / a))
    
    if norm == 'dimensionless':
        return w, T 
    elif norm == 'mass-weighted':
        return w, L 
    elif norm == 'both':
        return w, T, L 
    else:
        raise ValueError('unexpected norm option')

def calc_coriolis_zetas(L):
    """
    Calculate Coriolis zeta constants of rectilinear
    normal modes.

    Parameters
    ----------
    L : (3*n,N) ndarray
        The mass-weighted normal modes vectors for `N`
        modes. The columns of `L` are orthogonal.

    Returns
    -------
    zeta : (N,N,3) ndarray
        The Coriolis :math:`\\zeta` constants.

    """
    
    # Reshape L to (n,3,N)
    n = L.shape[0] // 3 # The number of atoms 
    N = L.shape[1]      # The number of modes
    
    L = L.reshape((n,3,N))
    
    zeta = np.zeros((N,N,3))
    
    
    for i in range(N):
        for j in range(N):
           for a in range(3):
                
               b = (a+1)%3  
               c = (a+2)%3 
               # a,b,c are cyclic in [0,1,2]
               
               # sum over atoms
               for K in range(n):
               
                   zeta[i,j,a] +=  L[K,b,i] * L[K,c,j] # eps_abc
                   zeta[i,j,a] += -L[K,c,i] * L[K,b,j] # eps_acb
    
    return zeta 

def calc_inertia_Qderiv(mass, Xe, L):
    """
    Calculate the inertia tensor derivatives with respect
    to the :math:`Q` normal coordinates.
    
    Parameters
    ----------
    mass : (n,) array_like
        The atomic masses 
    Xe : (3*n,) array_like
        The Cartesian reference geometry
    L : (3*n,N) array_like
        The mass-weighted normal modes.
    
    Returns
    -------
    aQ : (N,3,3) ndarray
        The inertia tensor derivatives
        
    Notes
    -----
    
    The inertia tensor deviatives are calculated 
    according to Eq. (22) of [Watson1968].
    
    References
    ----------
    .. [Watson1968] J. K. G. Watson, "Simplification of the molecular vibration-rotation Hamiltonian",
       Mol. Phys., 15, 479 (1968).
       https://doi.org/10.1080/00268976800101381
    
    """
    
    n = len(mass)
    L = np.array(L)
    N = L.shape[1] 
    
    aQ = np.zeros((N,3,3))
    
    XCOM = nitrogen.angmom.X2COM(Xe,mass) # Get COM coordinates 
    
    for k in range(N): # Normal mode k
        for a in range(3):
            for b in range(3):
                
                for i in range(n):
                    
                    # Case 1
                    # alpha = beta 
                    
                    if a == b:
                        c = (a + 1) % 3 # Let c be the cyclically next axis
                        d = (a + 2) % 3 # and d the next after 
                        
                        # Two terms, both with positive sign
                        aQ[k,a,b] += 2 * np.sqrt(mass[i]) * XCOM[3*i+c] * L[3*i+c,k]
                        aQ[k,a,b] += 2 * np.sqrt(mass[i]) * XCOM[3*i+d] * L[3*i+d,k]
                    
                    # Case 2
                    # alpha != beta 
                    
                    else:
                        # Only one term, the eps tensors are -1 total.
                        aQ[k,a,b] += -2 * np.sqrt(mass[i]) * XCOM[3*i+b] * L[3*i+a,k] 
        
        # Enforce numerical symmetry
        aQ[k] = (aQ[k] + aQ[k].T)/2 
    
    return aQ 
                    
def calcAlpha_harm(Be, omega, aQ, Ie):
    """
    Calculate the harmonic contribution to 
    the vibration-rotation :math:`\\alpha` parameters.

    Parameters
    ----------
    Be : (3,) array_like
        The equilibrium rotational constants in :math:`hc \\times \\mathrm{cm}^{-1}`` units.
    omega : (N,) array_like
        The harmonic frequencies in :math:`hc \\times \\mathrm{cm}^{-1}`` units.
    aQ : (N,3,3) array_like
        The inertia tensor derivatives w.r.t. :math:`Q`.
    Ie : (3,) array_like
        The equilibrium principal moments of inertia

    Returns
    -------
    alpha : (N,3) ndarray
        The :math:`\\alpha` constant for each mode and principal axis.

    """
    
    N = len(omega) 
    alpha = np.zeros((N,3))
    
    for k in range(N): # For each mode 
        for b in range(3): # For each axis 
        
            for c in range(3):
                alpha[k,b] += (-2*Be[b]**2 / omega[k]) * 3 * aQ[k,b,c]**2 / (4*Ie[c]) 
                
    return alpha 

def calcAlpha_cor(Be, omega, zeta):
    """
    Calculate the Coriolis contribution to 
    the vibration-rotation :math:`\\alpha` parameters.

    Parameters
    ----------
    Be : (3,) array_like
        The equilibrium rotational constants in :math:`hc \\times \\mathrm{cm}^{-1}`` units.
    omega : (N,) array_like
        The harmonic frequencies in :math:`hc \\times \\mathrm{cm}^{-1}`` units.
    zeta : (N,N,3) array_like
            The Coriolis :math:`zeta` constants.

    Returns
    -------
    alpha : (N,3) ndarray
        The :math:`\\alpha` constant for each mode and principal axis.

    """
    
    N = len(omega) 
    alpha = np.zeros((N,3))
    
    for k in range(N): # For each mode 
        for b in range(3): # For each axis 
        
            for ell in range(N):
                
                if ell == k:
                    continue # No k = ell contribution 
                
                vibfactor = (3*omega[k]**2 + omega[ell]**2) / (omega[k]**2 - omega[ell]**2)
                alpha[k,b] += (-2*Be[b]**2 / omega[k]) * zeta[k,ell,b]**2 * vibfactor 
   
    return alpha 
    
def calcAlpha_anharm(Be, omega, aQ, f3):
    """
    Calculate the cubic anharmonic contribution to 
    the vibration-rotation :math:`\\alpha` parameters.

    Parameters
    ----------
    Be : (3,) array_like
        The equilibrium rotational constants in :math:`hc \\times \\mathrm{cm}^{-1}`` units.
    omega : (N,) array_like
        The harmonic frequencies in :math:`hc \\times \\mathrm{cm}^{-1}`` units.
    aQ : (N,3,3) array_like
        The inertia tensor derivatives w.r.t. :math:`Q`.
    f3 : array_like
        The PES scaled derivative array including up to at least cubic derivatives.
        Only :math:`f_{kkk}` and :math:`f_{kkl}` type derivatives will be used.
        These derivatives are w.r.t. the :math:`q` dimensionless normal coordinates.

    Returns
    -------
    alpha : (N,3) ndarray
        The :math:`\\alpha` constant for each mode and principal axis.

    """
    
    N = len(omega) 
    alpha = np.zeros((N,3))
    
    hbar = nitrogen.constants.hbar 
    
    nck = nitrogen.autodiff.forward.ncktab(N+3) 
    
    
    for k in range(N): # For each mode 
        for b in range(3): # For each axis 
        
            for s in range(N):
                
                vibfactor = omega[k] / (omega[s] ** 1.5) 
                
                # need phi_kks 
                pos = np.array([0 for p in range(N)])
                pos[k] += 2 
                pos[s] += 1 
                idx = nitrogen.autodiff.forward.idxpos(pos, nck)
                if k == s :
                    scale = 6.0  # phi_kkk
                else:
                    scale = 2.0  # phi_kks, s != k 
                    
                phi_kks = f3[idx] * scale
                
                alpha[k,b] += (-2*Be[b]**2 / omega[k]) * aQ[s,b,b]/(2*hbar) * phi_kks * vibfactor
   
    return alpha 

def calctau(Ie,aQ,omega):
    """
    Calculate the quartic centrifugal distortion :math:`\\tau` parameters.
    
    Parameters
    ----------
    Ie : (3,) array_like
        The equilibrium principal moments of inertia
    aQ : (N,3,3) array_like
        The inertia tensor derivatives w.r.t. :math:`Q`.
    omega : (N,) array_like
        The harmonic frequencies in :math:`hc \\times \\mathrm{cm}^{-1}`` units.
        
    Returns
    -------
    tau : (3,3,3,3) ndarray
        The :math:`\\tau` tensor.
        
    """
    
    tau = np.zeros((3,3,3,3))
    N = len(omega) # the number of modes 
    
    hb = nitrogen.constants.hbar #
    hb6 = hb**6 
    
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    
                    for k in range(N):
                        
                        tau[a,b,c,d] += -hb6 /(2.0*Ie[a]*Ie[b]*Ie[c]*Ie[d]) * (aQ[k,a,b] * aQ[k,c,d]) / omega[k]**2
    
    return tau 

def calc_harmCD(Be,tau):
    """
    Calculate reduced quartic distortion parameters
    and effective rotational constants
    
    Parameters
    ----------
    Be : (3,) array_like
        The equilibrium rotational constants in :math:`hc \\times \\mathrm{cm}^{-1}`` units.
    tau : (3,3,3,3) array_like
        The unreduced quartic :math:`\\tau` parameters
        
    Notes
    -----
    The `Be` and `tau` arrays must be in axis order :math:`a,b,c`.
    
    Returns
    -------
    B0 : dict 
        Various corrected rotational constants.
    CD : dict
        Various reduction cases of the centrifugal distortion parameters
        
    See Also
    --------
    analyzeCD : Higher level function for calculating and printing CD parameters
    
    """
    
    # We follow Gordy and Cook, Section 8.3 to 
    # calculate the effective and reduced quartic CD equilibrium 
    # parameters
    #
    # The Be and tau arguments are the unreduced
    # equilibrium rotational constants and quartic CD parameters
    # (G&C Eq. 8.45 - 8.47)
    #
    # The first-order quartic CD Hamiltonian can always
    # be reduced first to an orthorhombic form with
    # 3 new effective rotational constants and 
    # 6 quartic CD tau's
    # 
    # These relations are given in G&C Table 8.5
    # 
    # t[i,j] will equal tau'[i,i,j,j], the effective non-zero
    # taus.
    #
    t = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            
            if i == j : 
                t[i,i] = tau[i,i,i,i]
            else:
                t[i,j] = tau[i,i,j,j] + 2*tau[i,j,i,j] 
    #
    # Note, t[i,j] = t[j,i] is symmetric
    #
    # G&C's big T equals the above litte t times 1/4
    
    # Bp will equal the effective rotational constants
    Bp = np.zeros((3,))
    for a in range(3):
        b = (a+1) % 3 
        c = (a+2) % 3
        Bp[a] = Be[a] + 0.25 * (3*tau[b,c,b,c] - 2*tau[c,a,c,a] - 2*tau[a,b,a,b])
    
    
    # 
    # Calculate the Kivelson-Wilson parameters first.
    # Note axis 0,1,2 = a,b,c = z,x,y (i.e. I^r)
    # 
    # See Gordy and Cook, Table 8.7
    #
    DJ = (-1/32) * (3*t[1,1] + 3*t[2,2] + 2*t[1,2])
    DK = DJ - 0.25 * (t[0,0] - t[0,1] - t[0,2])
    DJK = -DJ - DK - 0.25 * t[0,0]
    R5 = (-1/32) * (t[1,1] - t[2,2] - 2*t[0,1] + 2*t[0,2])
    R6 = (1/64) * (t[1,1] + t[2,2] - 2*t[1,2])
    deltaJ = (-1/16) * (t[1,1] - t[2,2])
    
    # Calculate the asymmetry parameter
    # in terms of the effective *orthorhombic* rotational
    # constants, i.e., Bp
    #
    sigma = (2 * Bp[0] - Bp[1] - Bp[2]) / (Bp[1] - Bp[2]) 
    
    #
    # We now calculate the standard reduced Hamiltonians
    #
    # First, A-reduced (I^r)
    #
    # Quartic CD parameters
    #
    DeltaJ = DJ - 2*R6 
    DeltaJK = DJK + 12*R6 
    DeltaK = DK - 10*R6 
    # deltaJ = deltaJ
    deltaK = -2*R5 - 4*sigma*R6 
    
    # Effective rotational constants
    # (0,1,2 = a,b,c = z,x,y)
    # 
    # G&C Eqs. 8.101-8.103. 
    # Note, the "Bx" constants in G&C are the effective
    # orthorhombic rotational constants, i.e. Bp
    #
    BxA = Bp[1] - 8*R6 * (sigma + 1)
    ByA = Bp[2] + 8*R6 * (sigma - 1)
    BzA = Bp[0] + 16*R6 
    BA = np.array([BzA, BxA, ByA])
    
    
    # Now calculate S-reduced, I^r parameters
    # (use conversions from Table 4 of Watson 1977
    #  or G&C Eq 8.107)
    #
    DJ_S = DJ + R5/sigma 
    DK_S = DK + 5*R5/sigma 
    DJK_S = DJK - 6*R5/sigma 
    d1 = -deltaJ 
    d2 = R6 + R5/(2*sigma)
    
    # The effective S-reduced constants
    # (G&C Eq. 8.114 - 8.116)
    #
    BxS = Bp[1] - 4*R6 + 2*R5*(1/sigma + 2)
    ByS = Bp[2] - 4*R6 + 2*R5*(1/sigma - 2)
    BzS = Bp[0] + 6*R6 - 5*R5/sigma 
    BS = np.array([BzS, BxS, ByS])
    
    CD_KW = {"DJ":DJ,
             "DK":DK,
             "DJK":DJK,
             "R5":R5,
             "R6":R6}
    
    CD_AIr = {"DeltaJ":DeltaJ,
              "DeltaK":DeltaK,
              "DeltaJK":DeltaJK,
              "deltaJ":deltaJ,
              "deltaK":deltaK}
    
    CD_SIr = {"DJ":DJ_S,
              "DK":DK_S,
              "DJK":DJK_S,
              "d1":d1,
              "d2":d2}
    
    
    B0 = {"Be":Be, "Bp":Bp, "BA":BA, "BS":BS}
    CD = {"KW":CD_KW, "AIr":CD_AIr, "SIr":CD_SIr}
    
    return B0, CD 
    
    
def analyzeCD(Xe,omega,Lvib,mass, printing = True):
    """
    Analyze and report quartic
    centrifugal distortion
    
    Parameters
    ----------
    Xe : (3*n,) array_like
        The Cartesian reference geometry (in the
        principal axis system)
    omega : (N,) array_like
        The harmonic frequencies 
    Lvib : (3*n,N) array_like
        The mass-weighted normal coordinates
    mass : (n,) array_like
        The atomic masses 
    printing : bool, optional 
        Print report. 
        
    Returns
    -------
    B0 : dict 
        Various corrected rotational constants.
    CD : dict
        Various reduction cases of the centrifugal distortion parameters
    
        
    """
    
    # Calculate the principal inertia tensors 
    Ie = np.diag(nitrogen.angmom.X2I(Xe, mass))
    hbar = nitrogen.constants.hbar
    Be = hbar**2 / (2*Ie) # The rotational constants, cm^-1
    # Be is ordered A B C
    
    # Calculate the inertia tensor derivatives 
    aQ = calc_inertia_Qderiv(mass, Xe, Lvib)
    
    # Calculate the unreduced
    # tau centrifugal distortion parameters 
    tau = calctau(Ie, aQ, omega)
    
    # Calculate reduced parameters
    B0,CD = calc_harmCD(Be, tau)

    
    # Print results 
    if printing:
        print("==========================================")
        print(" Harmonic centrifugal distortion analysis ")
        print("==========================================")
        print("")
        printCD(Be,B0,CD)

    return B0,CD

def printCD(Be,B0,CD):
    
        def printval(val, end = "\n"):
            # Print in cm^-1 and MHz 
            print(f"{val:15.5E}   {val*29979.2458:15.5f}   ", end = end)
            
        abc = ['A','B','C']
        Bp = B0["Bp"] # (the unreduced, effective orthorhombic parameters)
        sigma = (2*Bp[0] - Bp[1] - Bp[2])/(Bp[1]-Bp[2])  # asymmetry parameter 
        
        print("")
        print("               cm^-1             MHz      ")
        print("          --------------    --------------")
        for i in range(3):
            print("   " + abc[i] + "e   ",end = ""); printval(Be[i])
        print("")
        
        
        for i in range(3):
            print("   " + abc[i] + "'   ",end = ""); printval(B0["Bp"][i])
        print("")
        for i in range(3):
            print(" " + abc[i] + "' - " + abc[i] + "e" ,end = ""); printval(B0["Bp"][i]-Be[i])
        print("")
        
        for i in range(3):
            print("   " + abc[i] + "(A) ",end = ""); printval(B0["BA"][i])
        print("")
        for i in range(3):
            print(" " + abc[i] + "(A)-" + abc[i] + "e" ,end = ""); printval(B0["BA"][i]-Be[i])
        print("")
        
        for i in range(3):
            print("   " + abc[i] + "(S) ",end = ""); printval(B0["BS"][i])
        print("")
        for i in range(3):
            print(" " + abc[i] + "(S)-" + abc[i] + "e" ,end = ""); printval(B0["BS"][i]-Be[i])
        print("")
        print(f"   sigma ............ {sigma:.6f}         ")
        print("")
        print("  ---------------------------------------  ")
        print("         Kivelson-Wilson parameters        ")
        print("  ---------------------------------------  ")
        for p in CD["KW"]:
            print(f" {p:>7s}", end = ""); printval(CD["KW"][p])
        print("")
        print("  ---------------------------------------  ")
        print("          A-reduced (Ir) parameters        ")
        print("  ---------------------------------------  ")
        for p in CD["AIr"]:
            print(f" {p:>7s}", end = ""); printval(CD["AIr"][p])
        print("")
        print("  ---------------------------------------  ")
        print("          S-reduced (Ir) parameters        ")
        print("  ---------------------------------------  ")
        for p in CD["SIr"]:
            print(f" {p:>7s}", end = ""); printval(CD["SIr"][p])
        print("")
        print("==========================================")

def analyzeAlpha(Xe,omega,Lvib,mass,f3, printing = True):
    """
    Analyze and report VPT alpha constants.
    
    Parameters
    ----------
    Xe : (3*n,) array_like
        The Cartesian reference geometry (in the
        principal axis system)
    omega : (N,) array_like
        The harmonic frequencies 
    Lvib : (3*n,N) array_like
        The mass-weighted normal coordinates
    mass : (n,) array_like
        The atomic masses 
    f3 : ndarray
        The derivative array with up to at least cubic terms.
        Only :math:`f_{kkk}` and :math:`f_{kks}` types terms
        are used.
        These derivatives are w.r.t. the :math:`q` dimensionless normal coordinates.
    printing : bool, optional
        Print report
        
    Returns
    -------
    alpha : (N,3) ndarray
        The :math:`\\alpha` constant for each mode and principal axis.
        
    """
    
    # Calculate Ia, Ib, Ic
    Ie = np.diag(nitrogen.angmom.X2I(Xe, mass))
    # Calculate Ae, Be, Ce
    hbar = nitrogen.constants.hbar
    Be = hbar**2 / (2*Ie) 
    # Calculate inertia tensor derivatives
    aQ = calc_inertia_Qderiv(mass, Xe, Lvib)
    # Calculate Coriolis constants
    zeta = calc_coriolis_zetas(Lvib)
    
    # Calculate harmonic, anharmonic 
    # and Coriolis contributions to VPT2 
    # alpha's
    a_harm = calcAlpha_harm(Be, omega, aQ, Ie)
    a_anh  = calcAlpha_anharm(Be, omega, aQ, f3)
    a_cor  = calcAlpha_cor(Be, omega, zeta)
    # Total alpha
    alpha = a_harm + a_anh + a_cor 
    
    N = len(omega)
    c = 29979.2458
    #############
    # Print summary 
    def printabc(abc, end = "\n"):
        print(f"{abc[0]:13.5f}  {abc[1]:13.5f}  {abc[2]:13.5f}", end = end)
        
    if printing:
        print("=========================================================")
        print("              Vibration-rotation analysis                ")
        print("=========================================================")
        print("")
        print("   Contributions to alpha (MHz):                         ")
        print("")
        print("                  a              b              c        ")
        print("            -------------  -------------  -------------  ")
        for i in range(N):
            print("")
            print( "     Har.  ", end = ""); printabc(a_harm[i]*c)
            print(f" {i:2d}  Cor.  ", end = ""); printabc(a_cor[i]*c)
            print( "     Anh.  ", end = ""); printabc(a_anh[i]*c)
        
        print("")
        print("   Total alpha (MHz):                            ")
        print("")
        for i in range(N):
            print(f"   {i:2d}      ", end = ""); printabc(alpha[i]*c)
        print("")
        print("")
        
        B0_minus_Be = np.sum(-0.5 * alpha, axis = 0)
        
        print("  B0 - Be  ", end = ""); printabc(B0_minus_Be*c)
        print("")
        print("=========================================================")
    
    return alpha 

def deltaijk(omega):
    """
    Calculate :math:`\\Delta_{ijk} = (\\omega_i + \\omega_j + \\omega_k)
    (\\omega_i - \\omega_j - \\omega_k)
    (-\\omega_i + \\omega_j - \\omega_k)
    (-\\omega_i - \\omega_j + \\omega_k)`

    Parameters
    ----------
    omega : (N,) array_like
        The harmonic frequencies

    Returns
    -------
    delta : (N,N,N) ndarray
        The value of :math:`\\Delta_{ijk}`.

    """
    N = len(omega)
    delta = np.zeros((N,N,N))
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                delta[i,j,k] = (omega[i] + omega[j] + omega[k])*\
                               (omega[i] - omega[j] - omega[k])*\
                               (-omega[i] + omega[j] - omega[k])*\
                               (-omega[i] - omega[j] + omega[k])
    
    return delta 

def calc_g0(Be, omega, zeta, f4, printlevel = 0):
    """
    Evalulate the VPT2 value of the :math:`g_0` vibrational energy parameter.

    Parameters
    ----------
    Be : (3,) array_like
        The equilibrium rotational constants in :math:`hc \\times \\mathrm{cm}^{-1}`` units.
    omega : (N,) array_like
        The harmonic frequencies in :math:`hc \\times \\mathrm{cm}^{-1}`` units.
    zeta : (N,N,3) array_like
            The Coriolis :math:`zeta` constants.
    f4 : ndarray
        The derivative array with up to at least semi-diagonal quartic terms.
        These derivatives are w.r.t. the :math:`q` dimensionless normal coordinates.
    printlevel : integer, optional
        Print level. The default is 0.
        
    Returns
    -------
    g0 : float
        The value of :math:`g_0`.
    """
    
    N = len(omega) 
    delta = deltaijk(omega)
    nck = nitrogen.autodiff.forward.ncktab(N + 4) 
    
    g0 = 0.0 
    #####################
    # The expression for g_0 is taken from 
    # Gong et al 2018, Eq. 26
    #
    # Note that the phi's in the paper are 
    # *unscaled* derivatives w.r.t. q
    #
    # 1) Quartic contribution
    gquartic = 0.0 
    for k in range(N):
        
        # Get derivative index
        pos = np.array([0 for p in range(N)])
        pos[k] += 4 
        idx = nitrogen.autodiff.forward.idxpos(pos, nck)
        
        phi_kkkk = f4[idx] * 24.0  # Scale derivative
    
        gquartic += phi_kkkk / 64.0 
    g0 += gquartic 
    
    # 2a) One-mode cubic contribution
    gcubic = 0.0 
    for k in range(N):
        pos = np.array([0 for p in range(N)])
        pos[k] += 3 
        idx = nitrogen.autodiff.forward.idxpos(pos, nck)
        phi_kkk = f4[idx] * 6.0 
        
        gcubic += (-7.0 / 576.0) * phi_kkk**2 / omega[k] 
    
    # 2b) Two-mode cubic contribution 
    for k in range(N):
        for l in range(N):
            if k == l:
                continue 
            
            pos = np.array([0 for p in range(N)])
            pos[k] += 2 
            pos[l] += 1 
            idx = nitrogen.autodiff.forward.idxpos(pos, nck)
            phi_kkl = f4[idx] * 2.0  # 3/6 = 1/2 
            
            gcubic += (3.0 / 64.0) * (omega[l] * phi_kkl**2) / (4*omega[k]**2 - omega[l]**2)
    #
    # 2c) Three-mode cubic contribution 
    for k in range(N):
        for l in range(k):
            for m in range(l):
                # m < l < k 
                #
                pos = np.array([0 for p in range(N)])
                pos[k] += 1
                pos[l] += 1 
                pos[m] += 1 
                idx = nitrogen.autodiff.forward.idxpos(pos, nck)
                phi_klm = f4[idx] # 6/6 = 1 
                
                gcubic += (-1.0 / 4.0) * omega[k] * omega[l] * omega[m] * phi_klm**2 / delta[k,l,m]
    
    #
    g0 += gcubic 
    
    # 3) Coriolis contribution 
    gcor = 0.0 
    for a in range(3):
        for k in range(N):
            for l in range(k):
                # l < k 
                
                gcor += (-1/2) * Be[a] * zeta[k,l,a]**2
    g0 += gcor 
    
    
    # 
    # 4) Pseudo-potential contribution 
    gVT = 0.0 
    
    for a in range(3):
        gVT += (-1/4) * Be[a] 
    g0 += gVT 
    
    if printlevel > 0 :
        print("------------------------")
        print("  Contributions to g0")
        print("------------------------")
        print(f" Quartic ..... {gquartic:>+8.4f}")
        print(f" Cubic ....... {gcubic:>+8.4f}")
        print(f" Coriolis .... {gcor:>+8.4f}")
        print(f" Pseud-pot ... {gVT:>+8.4f}")
        print("")
        print(f" Total ....... {g0:>+8.4f}")
        print("")
    
    return g0 

def calc_xij(Be, omega, zeta, f4):
    """
    Evalulate the VPT2 anharmonicity parameters :math:`x_{ij}`.

    Parameters
    ----------
    Be : (3,) array_like
        The equilibrium rotational constants in :math:`hc \\times \\mathrm{cm}^{-1}`` units.
    omega : (N,) array_like
        The harmonic frequencies in :math:`hc \\times \\mathrm{cm}^{-1}`` units.
    zeta : (N,N,3) array_like
            The Coriolis :math:`zeta` constants.
    f4 : ndarray
        The derivative array with up to at least semi-diagonal quartic terms.
        These derivatives are w.r.t. the :math:`q` dimensionless normal coordinates.

    Returns
    -------
    g0 : float
        The value of :math:`g_0`.
    """
    
    N = len(omega) 
    delta = deltaijk(omega)
    nck = nitrogen.autodiff.forward.ncktab(N + 4) 
    
    xij = np.zeros((N,N)) 
    
    #####################
    # The expressions for x_ii and x_i!=j
    # are taken from Gong et al 2018, Eqs. 27-28
    #
    # Note that the phi's in the paper are 
    # *unscaled* derivatives w.r.t. q
    #
    # First, calculate diagonal anharmonicity 
    # constants, x_ii
    #
    for i in range(N):
        # x_ii
        
        # 1) phi_iiii contribution 
        pos = np.array([0 for p in range(N)])
        pos[i] += 4 
        idx = nitrogen.autodiff.forward.idxpos(pos, nck)
        
        phi_iiii = f4[idx] * 24.0 
        
        xij[i,i] += phi_iiii / 16.0 
        
        # 2) phi_iii contribution 
        pos = np.array([0 for p in range(N)])
        pos[i] += 3
        idx = nitrogen.autodiff.forward.idxpos(pos, nck)
        
        phi_iii = f4[idx] * 6.0 
        
        xij[i,i] += (-5.0 / 48.0) * phi_iii**2 / omega[i] 
        
        # 3) phi_iik contribution 
        for k in range(N):
            if k == i:
                continue 
            # i != k 
            pos = np.array([0 for p in range(N)])
            pos[i] += 2 
            pos[k] += 1 
            idx = nitrogen.autodiff.forward.idxpos(pos, nck)
            
            phi_iik = f4[idx] * 2.0 
            
            xij[i,i] += (-1.0 / 16.0) * phi_iik**2 * (8*omega[i]**2 - 3*omega[k]**2) /\
                ( omega[k] * (4*omega[i]**2 - omega[k]**2))
    #
    # Second, calculate the off-diagonal anharmonicity
    # constants
    #
    for i in range(N):
        for j in range(i):
            # j < i 
            
            pos = np.array([0 for p in range(N)])
            pos[i] += 2 
            pos[j] += 2 
            idx = nitrogen.autodiff.forward.idxpos(pos, nck)
            phi_iijj = f4[idx] * 4.0  # 6/24 = 1/4
            
            pos = np.array([0 for p in range(N)])
            pos[i] += 2 
            pos[j] += 1
            idx = nitrogen.autodiff.forward.idxpos(pos, nck)
            phi_iij = f4[idx] * 2.0  # 3/6 = 1/2
            
            pos = np.array([0 for p in range(N)])
            pos[i] += 1
            pos[j] += 2
            idx = nitrogen.autodiff.forward.idxpos(pos, nck)
            phi_ijj = f4[idx] * 2.0  # 3/6 = 1/2
            
            pos = np.array([0 for p in range(N)])
            pos[i] += 3
            idx = nitrogen.autodiff.forward.idxpos(pos, nck)
            phi_iii= f4[idx] * 6.0
        
            pos = np.array([0 for p in range(N)])
            pos[j] += 3
            idx = nitrogen.autodiff.forward.idxpos(pos, nck)
            phi_jjj= f4[idx] * 6.0
            
            
            # Two-mode terms
            xij[i,j] += phi_iijj / 4.0 
            
            xij[i,j] += (-1/2) * phi_iij**2 * omega[i] / (4*omega[i]**2 - omega[j]**2)
            xij[i,j] += (-1/2) * phi_ijj**2 * omega[j] / (4*omega[j]**2 - omega[i]**2)
            
            xij[i,j] += (-1/4) * phi_iii * phi_ijj / omega[i] 
            xij[i,j] += (-1/4) * phi_jjj * phi_iij / omega[j] # typo in Gong 2018 (??)
            
            # three-mode term 
            for k in range(N):
                if k == i or k == j:
                    continue 
                # k != i != j
                
                pos = np.array([0 for p in range(N)])
                pos[i] += 1
                pos[j] += 1 
                pos[k] += 1 
                idx = nitrogen.autodiff.forward.idxpos(pos, nck)
                phi_ijk = f4[idx] # 6/6 = 1
                
                pos = np.array([0 for p in range(N)])
                pos[i] += 2
                pos[k] += 1 
                idx = nitrogen.autodiff.forward.idxpos(pos, nck)
                phi_iik = f4[idx] * 2.0 # 3/6 = 1/2 
                
                pos = np.array([0 for p in range(N)])
                pos[j] += 2
                pos[k] += 1 
                idx = nitrogen.autodiff.forward.idxpos(pos, nck)
                phi_jjk = f4[idx] * 2.0 # 3/6 = 1/2 
                
                xij[i,j] += phi_ijk**2 * omega[k] * (omega[i]**2 + omega[j]**2 - omega[k]**2) / (2*delta[i,j,k])
                xij[i,j] += (-1/4) * phi_iik * phi_jjk / omega[k] 
            
            # Coriolis term 
            for a in range(3):
                xij[i,j] += Be[a] * zeta[i,j,a]**2 * (omega[i]/omega[j] + omega[j]/omega[i])
                
        
            
            xij[j,i] = xij[i,j] 
        
    return xij