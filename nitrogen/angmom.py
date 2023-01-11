"""
nitrogen.angmom
---------------

Angular momentum and spherical tensor routines.

"""

import numpy as np
import nitrogen.constants as constants 
import py3nj # For Wigner-nj symbols


def wigner3j(jj1, jj2, jj3, mm1, mm2, mm3):
    """
    Calculate the Wigner 3-j symbol,
    
    .. math::
    
       \\left(\\begin{array}{ccc} j_1 & j_2 & j_3 \\\\ m_1 & m_2 & m_3 \\end{array} \\right)

    Parameters
    ----------
    jj1 : integer
        Twice the value of :math:`j_1`.
    jj2 : integer
        Twice the value of :math:`j_2`.
    jj3 : integer
        Twice the value of :math:`j_3`.
    mm1 : integer
        Twice the value of :math:`m_1`.
    mm2 : integer
        Twice the value of :math:`m_2`.
    mm3 : integer
        Twice the value of :math:`m_3`.

    Returns
    -------
    float
        The result.
        
    Notes
    -----
    This currently wraps the `py3nj <https://github.com/fujiisoup/py3nj>`_ 
    implementation. The back-end may change in the future. 
    
    
    Examples
    --------
    >>> n2.angmom.wigner3j(2 * 20, 2 * 21, 2 * 22, 2 * 5, 2 * -15, 2 * 10)
    0.032597617477982975

    """
    
    result = py3nj.wigner3j(jj1, jj2, jj3, mm1, mm2, mm3)
    
    return result 

def wigner6j(jj1, jj2, jj3, jj4, jj5, jj6):
    """
    Calculate the Wigner 6-j symbol,
    
    .. math::
    
       \\left\\{\\begin{array}{ccc} j_1 & j_2 & j_3 \\\\ j_4 & j_5 & j_6 \\end{array} \\right\\}

    Parameters
    ----------
    jj1 : integer
        Twice the value of :math:`j_1`.
    jj2 : integer
        Twice the value of :math:`j_2`.
    jj3 : integer
        Twice the value of :math:`j_3`.
    jj4 : integer
        Twice the value of :math:`j_4`.
    jj5 : integer
        Twice the value of :math:`j_5`.
    jj6 : integer
        Twice the value of :math:`j_6`.

    Returns
    -------
    float
        The result.
        
    Notes
    -----
    This currently wraps the `py3nj <https://github.com/fujiisoup/py3nj>`_ 
    implementation. The back-end may change in the future. 
    
    
    Examples
    --------
    >>> n2.angmom.wigner6j(2 * 3, 2 * 6, 2 * 5, 2 * 4, 2 * 6, 2 * 9)
    -0.020558557070186504

    """
    
    result = py3nj.wigner6j(jj1, jj2, jj3, jj4, jj5, jj6)
    
    return result

def clebsch_gordan(jj1, jj2, jj3, mm1, mm2, mm3):
    """
    Calculate the Clebsch-Gordan coefficient,
    
    .. math::
        
       \\langle j_1\\,m_1, j_2 \\, m_2 \\vert j_3 \\, m_3 \\rangle
      
    Parameters
    ----------
    jj1 : integer
        Twice the value of :math:`j_1`.
    jj2 : integer
        Twice the value of :math:`j_2`.
    jj3 : integer
        Twice the value of :math:`j_3`.
    mm1 : integer
        Twice the value of :math:`m_1`.
    mm2 : integer
        Twice the value of :math:`m_2`.
    mm3 : integer
        Twice the value of :math:`m_3`.
    
    Returns
    -------
    float
        The result.
    
    Notes
    -----
    This currently wraps the `py3nj <https://github.com/fujiisoup/py3nj>`_ 
    implementation. The back-end may change in the future. 
    
    Examples
    --------
    >>> n2.angmom.clebsch_gordan(2 * 6, 2 * 9, 2 * 13, 2 * -3, 2 * 4, 2 * 1)
    0.4277601867185667
    
    
    """
    
    result = py3nj.clebsch_gordan(jj1, jj2, jj3, mm1, mm2, mm3)
    
    return result 

def dircos_tensor(N1,k1,m1,N2,k2,m2):
    """
    Calculate a matrix element of the 
    direction cosine spherical tensor,
    
    ..  math::
        
        \\langle N_1, k_1, m_1 \\vert \\lambda_{Qq} \\vert N_2, k_2, m_2 \\rangle
        

    Parameters
    ----------
    N1, k1, m2, N2, k2, m2 : integer
        Angular momentum quantum numbers

    Returns
    -------
    (3,3) ndarray
        The direction cosine tensor matrix element in terms
        of spherical tensor components. The components are ordered
        ``[0, +1, -1]`` so that normal array indexing is unchanged.
        
    Notes
    -----
    
    The basis functions are standard symmetric top rotational basis functions
    with the usual phase conventions. :math:`k` is the body-frame :math:`z` 
    component with respect to "anomalous" body-frame operators, :math:`J_x,J_y,J_z`.
    
    The direction cosine tensor :math:`\\lambda_{Q,q}^{(1,1)}` is a double
    tensor with respect to the lab-frame angular momentum (:math:`J_{X},J_Y,J_Z`)
    and the body-frame angular momentum (:math:`-J_x,-J_y,-J_z`). 
    Its components are
    
    ..  math::
        \\lambda_{Q,q} = (-1)^q \\left[D_{Q,-q}^{(1)}(\\phi,\\theta,\\chi)\\right]^*
    
    
    The matrix elements are
    
    ..  math::
        
        \\langle N_1, k_1, m_1 \\vert \\lambda_{Qq} \\vert N_2, k_2, m_2 \\rangle = 
             (-1)^{k_1 + k_2 + N_1 + N_2 - 1} \\sqrt{\\frac{2N_2+1}{2N_1+1}}
             
             \\times 
             \\langle N_2 m_2, 1 Q |  N_1 m_1 \\rangle
             \\langle N_2, -k_2, 1 q | N_1, -k_1 \\rangle
        

    """
    
    red = (-1) ** (k1 + k2 + N1 + N2 - 1) * np.sqrt( (2*N2 + 1) / (2*N1 + 1))
    
    lamQq = np.zeros((3,3))
    
    for Q in [-1,0,1]:
    
        cg1 = clebsch_gordan(2*N2, 2*1, 2*N1,
                             2*m2, 2*Q, 2*m1)
        
        for q in [-1, 0, 1]:
            
            cg2 = clebsch_gordan(2*N2, 2*1, 2*N1,
                                -2*k2, 2*q,-2*k1)
            
            lamQq[Q,q] = red * cg1 * cg2 
    
    return lamQq 
            
def dircos_tensor_cart(N1,k1,m1,N2,k2,m2):
    """
    Calculate a matrix element of the 
    direction cosine Cartesian tensor
    
    ..  math::
        
        \\langle N_1, k_1, m_1 \\vert \\lambda_{Ij} \\vert N_2, k_2, m_2 \\rangle
        

    Parameters
    ----------
    N1, k1, m2, N2, k2, m2 : integer
        Angular momentum quantum numbers

    Returns
    -------
    (3,3) ndarray
        The direction cosine tensor matrix element in terms
        of Cartesian components. 
    
    See Also
    --------
    dircos_tensor : Direction cosine spherical tensor matrix elements 
    
    Notes
    -----
    
    The indices are with respect to lab frame (:math:`I = X,Y,Z`) and 
    body-fixed frame (:math:`j = x,y,z`) axes.
    
    """
    
    # Calculate the spherical tensor representation
    lamQq = dircos_tensor(N1, k1, m1, N2, k2, m2)
    
    
    ir2 = 1/np.sqrt(2.0)
    #
    # Usc transforms Cartesian vector components (x,y,z)
    # to spherical vector components (q = 0, +1, -1)
    #
    Usc = np.array([
        [0, 0, 1],
        [-ir2, -1j*ir2, 0.0],
        [+ir2, -1j*ir2, 0.0]
        ])
    
    # Ucs transforms spherical to Cartesian
    # It is the conjugate transpose of Usc
    Ucs = np.conjugate(Usc.T)
    
    # Use Ucs to transform both spherical indices of 
    # lamQq to Cartesian indices
    # (Note: we use the Ucs.T for the right index, *not* 
    #  Ucs^dagger. This is not a similarity transformation)
    #
    
    lamIj = Ucs @ lamQq @ (Ucs.T) 
    
    return lamIj
    
def Jbf_cs(J):
    """
    Calculate Condon-Shortley body-fixed J operators

    Parameters
    ----------
    J : int
        Angular momentum: 0, 1, 2, ...

    Returns
    -------
    Jx,Jy,Jz : ndarray
        Body-fixed angular momentum components
        
    Notes
    -----
    The basis function order is :math:`k = -J, \ldots, +J`.

    """
    
    NJ = 2*J+1
    
    kI, kJ = np.meshgrid(np.arange(NJ)-J, np.arange(NJ)-J, indexing = 'ij')
    
    Jz = np.zeros((NJ,NJ))
    
    Jz[kI==kJ] = kJ[kI==kJ]
    
    # Calculate Jp = Jx + i*Jy
    # (this is the *lowering* operating in body-fixed frame
    #  because of the anamalous commutation sign of Jx, Jy, Jz)
    Jp = np.zeros((NJ,NJ))
    Jm = np.zeros((NJ,NJ))
    
    idx = (kI == kJ - 1)
    Jp[idx] = np.sqrt( J*(J+1) - kJ*(kJ-1) )[idx] # lowering operator
    
    idx = (kI == kJ + 1)
    Jm[idx] = np.sqrt( J*(J+1) - kJ*(kJ+1) )[idx] # raising operator
    
    Jx = (Jp + Jm) / 2.0
    Jy = (Jp - Jm) / (2.0 * 1j)
    
    return Jx, Jy, Jz

def U_wr2cs(J):
    """
    Wang transformation matrix, with additional
    phase factors for real functions.
    
    cs = U @ wr

    Parameters
    ----------
    J : int
        Angular momentum quantum number, 0, 1, 2, ...

    Returns
    -------
    W : ndarray
        The unitary transformation matrix

    """
    
    NJ = 2*J+1
    
    U = np.zeros((NJ,NJ), dtype = np.complex128)
    
    def NJKplus(J,K):
        if J % 2 == 1 and K % 2 == 1:
            return -1j * (1j)**(J+1)
        elif J % 2 == 0 and K % 2 == 0:
            return +1j * (1j)**(J+1)
        else:
            return 1 * (1j)**(J+1)
    def NJKminus(J,K):
        if J % 2 == 1 and K % 2 == 0:
            return +1j * (1j)**(J+1)
        elif J % 2 == 0 and K % 2 == 1:
            return -1j * (1j)**(J+1)
        else:
            return 1 * (1j)**(J+1)
        
    
    for i in range(J):
        
        U[i,i] = -1.0 / np.sqrt(2.0) * NJKminus(J,J-i)
        U[-(i+1), i] = 1.0 / np.sqrt(2.0) * NJKminus(J,J-i)
        
        U[i, -(i+1)] = 1.0 / np.sqrt(2.0) * NJKplus(J,J-i)
        U[-(i+1), -(i+1)] = 1.0 / np.sqrt(2.0) * NJKplus(J,J-i)
    
    U[J,J] = 1.0

    return U

def iJbf_wr(J):
    """
    Calculate body-fixed J operators in real, symmetrized JK basis
    ("Wang-Real")

    Parameters
    ----------
    J : int
        Angular momentum quantum number, 0, 1, 2, ...

    Returns
    -------
    iJx, iJy, iJz : ndarray
        The body-frame angular momentum operators multiplied
        by i. These are purely real, anti-symmetric matrices

    """
    
    Jxyz = Jbf_cs(J) # Condon-Shortley representation
    
    U = U_wr2cs(J)
    Uh = U.conj().T
    
    iJx = np.real(1j * (Uh @ Jxyz[0] @ U))
    iJy = np.real(1j * (Uh @ Jxyz[1] @ U))
    iJz = np.real(1j * (Uh @ Jxyz[2] @ U))
    
    # Enforce some strict selection rules
    NJ = 2*J+1
    i,j = np.meshgrid(range(NJ),range(NJ),indexing = 'ij')
    
    iJx[abs(i-j) != 1] = 0  # Only the diagonal +/- 1 is non-zero
    iJy[abs(i-(NJ-j-1)) != 1] = 0 # Only the anti-diagonal +/- 1 is non-zero
    
    iJz[i != (NJ-j-1)] = 0 # Only the anti-diagonal is non-zero
    iJz[J,J] = 0
    
    # Let's enforce strict anti-symmetry
    for O in (iJx, iJy, iJz):
        for i in range(NJ):
            for j in range(i+1):
                if i == j:
                    O[i,j] = 0
                else:
                    O[j,i] = -O[i,j]
    
    return iJx, iJy, iJz
    
def iJiJbf_wr(J):
    """
    Calculate the anti-commutators [iJ_a,iJ_b]_+ for
    body-fixed angular momentum components in the Wang-real
    representation.

    Parameters
    ----------
    J : int
        Total angular momentum, 0, 1, 2, ...

    Returns
    -------
    iJiJ : nested tuple of ndarrays
        iJiJ[a][b] is the [iJa, iJb]_+ anti-commutator ndarray
    
    """
    
    iJ = iJbf_wr(J)
    
    def ac(A,B):
        return A@B + B@A
    
    iJiJ = tuple(tuple(ac(iJ[a],iJ[b]) for b in range(3)) for a in range(3))
    
    return iJiJ


def X2I(X, mass):
    """
    Calculate the inertia tensor from Cartesian
    coordinates.

    Parameters
    ----------
    X : ndarray
        A (3*N,...) array containing the 
        x, y, and z Cartesian positions of N particles.
    mass : array_like
        The masses of the N particles.

    Returns
    -------
    I : ndarray
        A (3,3,...) array containing the 
        symmetric inertia tensor

    """

    if X.shape[0] % 3 != 0 :
        raise ValueError("The first dimension of X must be a multiple of 3")
    base_shape = X.shape[1:]
    N = X.shape[0] // 3 # floor division (shouldn't matter)
    
    X3 = X.copy()
    X3 = np.reshape(X3, (N,3) + base_shape)
    
    #########################################################
    # Calculate the Center-of-Mass
    COM = np.zeros((3,) + base_shape)
    for i in range(N):
        COM += mass[i] * X3[i,:]
    COM = COM / sum(mass)
    
    for i in range(N):
        X3[i,:] -= COM  # calculate X in COM frame
    #
    #########################################################
        
    ##########################################################
    # Calculate inertia tensor 
    I = np.zeros((3,3) + base_shape)
    for i in range(N):
        I[0,0] += mass[i] * (X3[i,1]**2 + X3[i,2]**2) # x,x
        I[1,1] += mass[i] * (X3[i,2]**2 + X3[i,0]**2) # y,y
        I[2,2] += mass[i] * (X3[i,0]**2 + X3[i,1]**2) # z,z
        I[0,1] += -mass[i] * X3[i,0] * X3[i,1]  # x,y
        I[0,2] += -mass[i] * X3[i,0] * X3[i,2]  # x,z
        I[1,2] += -mass[i] * X3[i,1] * X3[i,2]  # y,z
    # Copy symmetric elements
    np.copyto(I[1,0:1], I[0,1:2])
    np.copyto(I[2,0:1], I[0,2:3])
    np.copyto(I[2,1:2], I[1,2:3])
    #
    ##########################################################
    
    return I 
        
    
def X2ABC(X, mass):
    """
    Calculate rotational constants from
    Cartesian positions.

    Parameters
    ----------
    X : ndarray
        A (3*N,...) array containing the 
        x, y, and z Cartesian positions of N particles.
    mass : array_like
        The masses of the N particles.
        
    Returns
    -------
    ABC : ndarray
        A (3,...) array containing the
        A, B, and C rotational constants 
        (in energy units).

    """
    
    # Calculate moment of inertia tensor
    I = X2I(X, mass) # shape (3, 3, ...)
    # Move tensor indices to the last indices
    I = np.moveaxis(I, (0,1), (-2, -1)) # shape (..., 3, 3)
    
    # Now diagonalize
    w,_ = np.linalg.eigh(I) # w has shape (..., 3)
    w = np.moveaxis(w, -1, 0)  # move abc index to front
    
    # Calculate rotational constants
    # B = hbar**2 / (2 * I)
    ABC = constants.hbar**2 / (2.0 * w) # hc * cm^-1
    
    return ABC

def X2PAS(X, mass):
    """
    Rotate coordinates to the principal
    axis system.
    
    Parameters
    ----------
    X : ndarray
        A (3*N,...) array containing the
        x, y, and z Cartesian positions of N particles.
    mass : array_like
        The masses of the N particles.
        
    Returns
    -------
    XPAS : ndarray
        A (3*N,...) array of the positions in the
        PAS frame with axes ordered :math:`a`, :math:`b`, :math:`c`.
    R : ndarray
        A (3,3,...) orthogonal array containing the transformation
        matrix from the original axes to the principal
        axis system.
    COM : ndarray
        A (3,...) array of the center-of-mass position
        in the original frame.
        
    Notes
    -----
    The PAS coordinates are defined as
    
    ..  math::
        
        \\vec{x}_\\text{PAS} = \\mathbf{R}(\\vec{x} - \\vec{x}_\\text{COM}).
        
    The rows of :math:`\\mathbf{R}` equal the unit vectors of the principal
    axes with respect to the input coordinate frame.
    
    """
    
    # Calculate the inertia tensor
    I = X2I(X, mass) 
    
    # Calculate the eigenvectors
    I = np.moveaxis(I, (0,1), (-2,-1))
    _,U = np.linalg.eigh(I)
    
    # Check that they form a right-handed axis
    # system
    #
    # Compute (a x b) . c
    #
    # If this is +1, then the system is right-handed
    # If this is -1, then the system is left-handed
    # and one vector needs to change sign (or all three)
    trip = np.inner( np.cross(U[:,0], U[:,1]), U[:,2] )
    if U.ndim == 2:
        if trip < 0:
            U *= -1 
    else:
        U[:,:, trip < 0] *= -1
    
    # The array that transforms vectors in 
    # the original frame to the PAS is
    # the transpose of U
    #
    RPAS = U.T 
    RPAS = np.moveaxis(RPAS, (-2,-1), (0,1))
    
    
    # Finally, calculate the 
    # PAS coordinates
    XPAS = np.zeros_like(X)
    
    N = X.shape[0] // 3  # Number of atoms
    
    # First, calculate the center of mass
    COM = 0
    for k in range(N):
        COM += mass[k] * X[3*k:3*(k+1)]
    COM /= sum(mass) 
    
    for k in range(N):
        
        for i in range(3):
            for j in range(3):
                
                XPAS[3*k+i] += RPAS[i,j] * (X[3*k+j] - COM[j])
    
    return XPAS, RPAS, COM


def Nbf_matrix(N):
    """
    Calculate the body-fixed operators of :math:`\mathbf{N}` for
    a given :math:`N` quantum number,
    
    ..  math::
        \\langle N k' \\vert N_i \\vert N k \\rangle

    Parameters
    ----------
    N : integer
        The :math:`N` quantum number

    Returns
    -------
    Nx,Ny,Nz : ndarray
        The matrix representations.
        
    Notes
    -----
    The basis function order is :math:`k = 0, 1, \\ldots, N, -N, -N+1, \\ldots, -1`.
    This is different than :func:`Jbf_cs`.

    """
    
    # 
    # k = 0, 1, 2, ..., N, -N, ..., -1
    #
    # The final ndarrays will be indexable by the signed k quantum number
    #
    krange = [i for i in range(N+1)] + [ i-N for i in range(N)]
    n = 2*N+1 # the number of functions
    
    kI, kJ = np.meshgrid(krange, krange, indexing = 'ij')
    
    Nz = np.diag(krange) 
    
    # Calculate Np = Nx + i*Ny 
    # (this is the *lowering* operating in body-fixed frame
    #  because of the anamalous commutation sign of Nx, Ny, Nz)
    Np = np.zeros((n,n))
    Nm = np.zeros((n,n))
    
    idx = (kI == kJ - 1)
    Np[idx] = np.sqrt( N*(N+1) - kJ*(kJ-1) )[idx] # lowering operator
    
    
    # Raising operator, Nm = Nx - i*Ny 
    idx = (kI == kJ + 1)
    Nm[idx] = np.sqrt( N*(N+1) - kJ*(kJ+1) )[idx] # raising operator
    
    Nx = (Np + Nm) / 2
    Ny = (Np - Nm) / (2 * 1j) 
    
    return Nx, Ny, Nz

def L_matrix(L):
    """
    Calculate the Cartesian components of a general
    angular momentum operator with normal commutation 
    relations
    
    ..  math::
        \\langle L m' \\vert L_i \\vert L m \\rangle

    Parameters
    ----------
    L : integer
        The total angular momentum quantum number, :math:`L`.

    Returns
    -------
    LX, LY, LZ : ndarray
        The matrix representations.
        
    Notes
    -----
    The basis function order is :math:`m = 0, 1, \\ldots, L, -L, -L+1, \\ldots, -1`.

    """
    
    # 
    # k = 0, 1, 2, ..., L, -L, ..., -1
    #
    # The final ndarrays will be indexable by the signed m quantum number
    #
    mrange = [i for i in range(L+1)] + [ i-L for i in range(L)]
    n = 2*L+1 # the number of functions
    
    mI, mJ = np.meshgrid(mrange, mrange, indexing = 'ij')
    
    LZ = np.diag(mrange) 
    
    Lp = np.zeros((n,n))
    Lm = np.zeros((n,n))
    
    idx = (mI == mJ + 1)
    Lp[idx] = np.sqrt( L*(L+1) - mJ*(mJ + 1) )[idx] # raising operator
    
    
    idx = (mI == mJ - 1)
    Lm[idx] = np.sqrt( L*(L+1) - mJ*(mJ - 1) )[idx] # lowering operator
    
    LX = (Lp + Lm) / 2
    LY = (Lp - Lm) / (2 * 1j) 
    
    return LX, LY, LZ

def caseb_multistate_N(alpha, N, k, SS1, JJ1):
    """
    Calculate the body-fixed :math:`N_i` operators 
    for a multi-state case (b) basis set.

    Parameters
    ----------
    alpha : array_like
        The electronic (or other) state index.
    N : array_like
        The :math:`N` quantum number.
    k : array_like
        The signed :math:`k` quantum number.
    SS1 : array_like
        The value of :math:`2S+1`.
    JJ1 : array_like
        The value of :math:`2J+1`.

    Returns
    -------
    Nx, Ny, Nz : ndarray
        The matrix elements of the body-fixed components
        of :math:`\\mathbf{N}`. 
        
    Notes
    -----
    
    The case (b) basis function is
    
    ..  math::
        \\vert J m_J N k S; \\alpha \\rangle = \\sum_{m_N, m_S} 
        \\vert N k m_N \\rangle \\vert S m_S \\rangle \\vert \\alpha \\rangle
        \\langle N m_N, S m_S \\vert J m_J \\rangle
        
    The matrix elements of the body-fixed components :math:`N_i`, :math:`i = x,y,z`,
    are 
    
    ..  math::
        \\langle J' m_J' N' k' S' ;\\alpha' \\vert N_i \\vert J m_J N k S ; \\alpha\\rangle
        = \\delta_{\\alpha\\alpha'} \\delta_{JJ'} \\delta_{m_J m_J'} \\delta_{SS'}
          \\delta_{NN'} \\langle N' k' \\vert N_i \\vert N k \\rangle 
    """
    
    
    n = len(N) # The number of basis functions in the list 
    
    Nx = np.zeros((n,n), dtype = np.complex128)
    Ny = np.zeros((n,n), dtype = np.complex128)
    Nz = np.zeros((n,n), dtype = np.complex128)
    
    # Calculate N operators for every possible value of N
    Nops = [ (Nbf_matrix(i) if i in N else None) for i in range(max(N)+1)] 
    
    for i in range(n):
        for j in range(n):
            #
            # The matrix element is diagonal in 
            # alpha (electronic index or other), N, S and J
            #
            if alpha[i] != alpha[j]:
                continue 
            if N[i] != N[j]:
                continue 
            if SS1[i] != SS1[j]:
                continue 
            if JJ1[i] != JJ1[j]:
                continue
            
            Nx[i,j] = Nops[N[i]][0][k[i], k[j]]
            Ny[i,j] = Nops[N[i]][1][k[i], k[j]]
            Nz[i,j] = Nops[N[i]][2][k[i], k[j]]
    
    
    return Nx, Ny, Nz 

def caseb_multistate_S(alpha, N, k, SS1, JJ1):
    """
    Calculate the body-fixed :math:`S_i` operators 
    for a multi-state case (b) basis set.

    Parameters
    ----------
    alpha : array_like
        The electronic (or other) state index.
    N : array_like
        The :math:`N` quantum number.
    k : array_like
        The signed :math:`k` quantum number.
    SS1 : array_like
        The value of :math:`2S+1`.
    JJ1 : array_like
        The value of :math:`2J+1`.

    Returns
    -------
    Sx, Sy, Sz : ndarray
        The matrix elements of the body-fixed components
        of :math:`\\mathbf{S}`. 
        
    Notes
    -----
    
    The case (b) basis function is
    
    ..  math::
        \\vert J m_J N k S; \\alpha \\rangle = \\sum_{m_N, m_S} 
        \\vert N k m_N \\rangle \\vert S m_S \\rangle \\vert \\alpha \\rangle
        \\langle N m_N, S m_S \\vert J m_J \\rangle
        
    The matrix elements of the body-fixed components :math:`S_i`, :math:`i = x,y,z`,
    are calculated by first calculating the body-fixed  spherical tensor components
    
    ..  math::
        
        &\\langle J' m_J' N' k' S' ;\\alpha' \\vert S_q \\vert J m_J N k S ; \\alpha\\rangle
        = \\delta_{\\alpha\\alpha'} \\delta_{JJ'} \\delta_{m_J m_J'} \\delta_{SS'}  \\\\
        &\\qquad\\qquad \\times (-1)^{k + J + S + 1} \\sqrt{(2N+1)(2N'+1)(2S+1)S(S+1)} 
          \\left(\\begin{array}{ccc} N & 1 & N' \\\\ k & -q & -k' \\end{array} \\right)
          \\left\\{\\begin{array}{ccc} N & S & J \\\\ S & N' & 1 \\end{array} \\right\\}
                
    and then relating
    
    ..  math::
        
        S_x &= \\frac{1}{\\sqrt{2}} ( -S_{q = +1} + S_{q = -1}) \\\\
        S_y &= \\frac{+i}{\\sqrt{2}} ( S_{q = +1} + S_{q = -1}) \\\\
        S_z &= S_{q = 0} 
    
    """
    
    
    n = len(N) # The number of basis functions in the list 
    
    Sq = np.zeros((3,n,n))
    
    for q in [-1,0,1]:
        for i in range(n):
            for j in range(n):
                #
                # The matrix element is diagonal in 
                # alpha (electronic index or other), S and J
                #
                if alpha[i] != alpha[j]:
                    continue 
                if SS1[i] != SS1[j]:
                    continue 
                if JJ1[i] != JJ1[j]:
                    continue
                
                # (-1) ** (k + J + S + 1)
                coeff1 = (-1)**(k[j] + (SS1[j] + JJ1[j])/2 )
                # sqrt((2N+1) * (2N'+1) * (2S+1) * S(S+1) )
                ssp1 = (SS1[j] ** 2  - 1)/4 # the value of S(S+1) 
                coeff2 = np.sqrt( (2*N[j] + 1) * (2*N[i] + 1) * SS1[j] * ssp1) 
                
                threej = wigner3j(2*N[j], 2*1, 2*N[i],
                                  2*k[j], 2*(-q), 2*(-k[i])) 
                
                sixj = wigner6j(2*N[j], SS1[j]-1, JJ1[j]-1,
                                SS1[j]-1, 2*N[i], 2*1)
                
                Sq[q,i,j] = coeff1 * coeff2 * threej * sixj 
    #
    # Calculate the Cartesian components
    # from the spherical tensor components
    #
    Sx = (-Sq[1] + Sq[-1]) / np.sqrt(2) 
    Sy = 1j * (Sq[1] + Sq[-1]) / np.sqrt(2) 
    Sz = Sq[0].copy()  
    
    return Sx, Sy, Sz 

def caseb_multistate_dircos(Np, kp, SS1p, JJ1p, 
                            N, k, SS1, JJ1):
    """
    Calculate the (lab-)reduced matrix elements of the
    direction cosine tensor in a multi-state case (b) basis set.

    Parameters
    ----------
    Np : array_like
        The bra :math:`N` quantum number.
    kp : array_like
        The bra signed :math:`k` quantum number.
    SS1p : array_like
        The bra value of :math:`2S+1`.
    JJ1p : array_like
        The bra value of :math:`2J+1`.
    N : array_like
        The ket :math:`N` quantum number.
    k : array_like
        The ket signed :math:`k` quantum number.
    SS1 : array_like
        The ket value of :math:`2S+1`.
    JJ1 : array_like
        The ket value of :math:`2J+1`.

    Returns
    -------
    lamq : ndarray
        The body-frame spherical tensor components. 
        ``lamq[q]`` = :math:`\\lambda_q`` where :math:`q = 0,+1,-1`.
        
    Notes
    -----
    
    The reduced matrix element is
    
    ..  math::
        
        \\langle J' N' k' S' || \\lambda_q || J N k S \\rangle = 
            \\delta_{SS'} (-1)^{k + S + J + 1} 
            [(2J+1)(2N'+1)(2N+1)]^{1/2}
            \\left(\\begin{array}{ccc} N & 1 & N' \\\\ -k & q & k' \\end{array} \\right)
            \\left\\{\\begin{array}{ccc} N' & J' & S' \\\\ J & N & 1 \\end{array} \\right\\}
                
        
    The :math:`\\alpha` index is not used here, as this is usually
    absorbed into the factor that the direction cosine tensor
    multiplies.
    
    """
    
    mp = len(Np) # The number of bra functions (using mp because np is numpy ! Sorry ! >< )
    n = len(N) # The number of ket functions
    
    lamq = np.zeros((3,mp,n))
    
    for q in [0,1,-1]:
        for i in range(mp):
            for j in range(n):
                #
                # The matrix element is diagonal in 
                # S
                #
                if SS1p[i] != SS1[j]:
                    continue 
                
                # (-1) ** (k + J + S + 1)
                # Note (J + S + 1) = (JJ1 + SS1)/2 
                #
                coeff1 = (-1)**(k[j] + (SS1[j] + JJ1[j])/2 )
                # sqrt((2N+1) * (2N'+1) * (2J+1) )
                coeff2 = np.sqrt( (2*N[j] + 1) * (2*Np[i] + 1) * JJ1[j] ) 
                
                threej = wigner3j(2*N[j], 2*1, 2*Np[i],
                                 -2*k[j], 2*q, 2*kp[i]) 
                
                sixj = wigner6j(2*Np[i],   JJ1p[i]-1, SS1p[i]-1,
                                JJ1[j]-1,  2*N[j],    2*1)
                
                lamq[q,i,j] = coeff1 * coeff2 * threej * sixj 
 
    # Return the spherical components    
    return lamq
        
def caseb_multistate_L(Li_e, LiLj_ac_e, alpha, N, k, SS1, JJ1):
    """
    Calculate the body-fixed :math:`L_i` operators 
    for a multi-state case (b) basis set.

    Parameters
    ----------
    Li_e : (3,NE,NE) array_like
        The pure electronic matrix elements of 
        :math:`L_i`.
    LiLj_ac_e : (3,3,NE,NE) array_like
        The pure electronic matrix elements of
        the anti-commutators :math:`[L_i, L_j]_+ = L_i L_j + L_j L_i`.
    alpha : array_like
        The electronic state index, i.e. the values for
        indexing into `Li_e` and `LiLj_ac_e`.
    N : array_like
        The :math:`N` quantum number.
    k : array_like
        The signed :math:`k` quantum number.
    SS1 : array_like
        The value of :math:`2S+1`.
    JJ1 : array_like
        The value of :math:`2J+1`.

    Returns
    -------
    Li : (3,n,n) ndarray
        Li[i] is the :math:`L_i` operator in the case (b) representation.
    LiLj_ac : (3,3,n,n) ndarray
        LiLj_ac[i,j] is the :math:`[L_i, L_j]_+` anti-commutator in 
        the case (b) representation.

    """
    
    Li_e = np.array(Li_e)
    LiLj_ac_e = np.array(LiLj_ac_e) 
    
    # Construct Li in spin-rot-elec basis 
    n = len(N) # the number of case (b) basis functions 
    
    Li = np.zeros((3, n, n), dtype = np.complex128)
    LiLj_ac = np.zeros((3, 3, n, n), dtype = np.complex128)
    
    
    for i in range(n):
        for j in range(n):
            
                #
                # Diagonal in N, k, S and J
                #
                if N[i] != N[j] or k[i] != k[j] or \
                   SS1[i] != SS1[j] or \
                   JJ1[i] != JJ1[j]:
                    continue
                
                for a in range(3):
                    Li[a,i,j] = Li_e[a, alpha[i], alpha[j]]
                
                    for b in range(3):
                        LiLj_ac[a,b,i,j] = LiLj_ac_e[a,b, alpha[i], alpha[j]]
                        
    return Li, LiLj_ac 
    